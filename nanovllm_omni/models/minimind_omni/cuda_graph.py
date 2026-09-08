"""CUDA Graph streaming decode for MiniMind-O (plan §3.3).

Wraps the fixed-KV-buffer attention (``enable_fixed_kv_buffer``, report
§23-§25) in a per-step CUDA Graph capture/replay loop, collapsing the
~8 000 per-generate `cudaLaunchKernel` calls into a handful of graph
replays (~7.5x measured on the thinker decode primitive, RTX 3050).

Semantics
---------
- **``n_steps`` = ``max_new_tokens``**: the first token is produced by
  eager prefill; the remaining ``n_steps - 1`` are decode steps, each
  replaying one pre-captured CUDA graph.  Accordingly, ``_capture``
  builds exactly ``n_steps - 1`` graphs (one per decode position).
- **capture once, replay many** (§25): re-capturing per run folds
  capture-time RNG/state side effects into the loop and breaks
  determinism. A decoder captures ``n_steps - 1`` per-step graphs the
  first time it runs, then replays them on every later call that hits
  the same prompt length and token budget.
- **position-dependent graphs**: each graph bakes the KV write position
  (``_kv_pos``) via Python-int tensor slicing at capture time.  A
  single graph cannot serve multiple decode positions because the KV
  read range (``0 .. _kv_pos``) and write slot are frozen in the
  captured op stream.  Making graphs position-independent would require
  converting ``_kv_pos`` to a device-side scalar tensor and using
  ``torch.narrow`` with a tensor index — a significant attention rewrite
  for marginal VRAM savings (one fewer graph object per budget).  The
  current per-position approach is standard practice (vLLM, TGI) and is
  kept intentionally.
- Host-side multinomial sampling lives OUTSIDE the graphs (§24/§25 keep
  sampling's RNG host-side and deterministic).
- The two freqs host-reads in the upstream forward are the only capture
  blockers (§13/§18). This module neutralizes them by parsing a copy of
  the forward's source into an AST, structurally matching the
  ``<x>.freqs_cos[0, 0] == 0`` guard shape, and setting its test to
  ``if False:`` (warmup proves them dead) before compiling and calling
  that copy — it never rebinds the model class.

Opt-in: ``enable_cuda_graph`` returns ``None`` when CUDA is unavailable or
the model isn't the MiniMind-O upstream shape. The caller keeps the eager
path in that case.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Any

import torch

from nanovllm_omni.models.minimind_omni._sampling import (
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_P,
    NUM_AUDIO_LAYERS,
    sample_one_audio_layer,
    sample_text_token,
)
from nanovllm_omni.models.minimind_omni.attention import enable_fixed_kv_buffer

_log = logging.getLogger(__name__)


def _is_freqs_zero_guard(test: ast.expr) -> bool:
    """True when ``test`` is structurally ``<...>.freqs_cos[0, 0] == 0``.

    This is the meta-device RoPE-buffer recompute guard ("buffers lost
    during meta-device init") present twice in the upstream MiniMind-Omni
    forward -- once for ``self.thinker``, once for ``self.talker``. It is a
    host-read: comparing a GPU tensor element against a Python int forces a
    device sync, which ``torch.cuda.graph()`` capture forbids outright
    (raises rather than silently misbehaving). Since the model is always
    fully loaded (not meta) by the time capture runs, the guard is always
    False in practice -- but its *presence* in the captured op stream is
    still fatal, so it must be eliminated from the source, not just skipped
    at runtime.

    Matching structurally (AST shape) rather than by literal string means a
    whitespace or variable-naming change upstream (``self.thinker`` vs
    ``model.thinker``, spacing, etc.) does not silently break detection --
    only a genuine change in what the guard *checks* would.
    """
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    right = test.comparators[0]
    if not (isinstance(right, ast.Constant) and right.value == 0):
        return False
    left = test.left
    if not isinstance(left, ast.Subscript):
        return False
    # Left side must end in a `.freqs_cos` attribute access -- any receiver
    # chain (`self.thinker`, `model.talker`, etc.) is accepted.
    if not (isinstance(left.value, ast.Attribute) and left.value.attr == "freqs_cos"):
        return False
    # Slice must be the literal tuple `[0, 0]`.
    sl = left.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else None
    if elts is None or len(elts) != 2:
        return False
    return all(isinstance(e, ast.Constant) and e.value == 0 for e in elts)


class _NeutralizeFreqsGuards(ast.NodeTransformer):
    """Replaces every ``if <freqs_zero_guard>:`` test with ``if False:``.

    Counts replacements on ``self.replaced`` so the caller can assert the
    expected number of capture blockers were actually found (drift in the
    upstream source should fail loud, not silently skip a host-read).
    """

    def __init__(self) -> None:
        self.replaced = 0

    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        if _is_freqs_zero_guard(node.test):
            self.replaced += 1
            node.test = ast.Constant(value=False)
        return node


def _patched_forward(cls: Any, src: str) -> Any:
    """Compile a copy of cls.forward with the two freqs `[0,0]` host-read
    checks neutralized (proven capture blockers, §13/§18). Returns a plain
    function `fwd(model, **kwargs)` -- never rebinds the class.

    Neutralization happens at the AST level (structural match on
    ``_is_freqs_zero_guard``), not by literal string replacement: the
    guard's exact spelling/whitespace can drift across upstream model
    revisions without silently breaking detection.
    """
    tree = ast.parse(textwrap.dedent(src))
    transformer = _NeutralizeFreqsGuards()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    # Require exactly the two known capture blockers to be neutralized, so
    # a drift in the upstream forward's freqs-check shape fails loud
    # instead of silently skipping one host-read during capture.
    if transformer.replaced != 2:
        _log.warning(
            "enable_cuda_graph: expected 2 freqs host-reads to neutralize, "
            "found %s; skipping graph path",
            transformer.replaced,
        )
        return None
    namespace = dict(cls.forward.__globals__)
    namespace["__name__"] = cls.__module__
    namespace["__qualname__"] = cls.__qualname__ + ".forward_graph"
    exec(compile(tree, "<enable_cuda_graph>", "exec"), namespace)
    return namespace["forward"]


def _build_omni_input(row: torch.Tensor, audio_pad: int) -> torch.Tensor:
    """Build the [1, 9, X] joint text+audio input that stream_generate's
    first forward expects (8 audio-pad rows + 1 token row).

    Used twice:
      - decode step (X=1): ``row`` is shape ``[1, 1]`` (the next token).
      - prefill (X=seq): ``row`` is shape ``[1, seq]`` (the text prompt).

    Fact #3: this 8+1 layout is load-bearing for the bit-exact contract —
    without the audio rows, decode logits diverge from index 1.
    """
    width = row.shape[1]
    # 8 audio-pad rows + 1 token row = the stream_generate joint layout
    buf = torch.full((1, 8, width), audio_pad, dtype=torch.long, device=row.device)
    return torch.cat((buf, row.unsqueeze(1)), dim=1)


class CudaGraphDecoder:
    """Capture-once / replay-many CUDA-Graphed decoder over the buffer-ized
    model's per-instance KV buffers.

    Each ``self.steps[k]`` is ``(graph, static_input, output, bridge)``: the
    captured CUDAGraph, the input tensor pinned to the capture-time address,
    the output reference held by the graph, and its replay-owned bridge output.
    """

    def __init__(
        self,
        model: Any,
        fwd: Any,
        n_steps: int,
        *,
        eos_token_id: int | None = None,
        audio_stop_token: int | None = None,
    ) -> None:
        self.model = model
        self.fwd = fwd
        self.n_steps = n_steps
        self.temperature = DEFAULT_TEXT_TEMPERATURE
        self.top_p = DEFAULT_TEXT_TOP_P
        self.rp = 1.0
        self.audio_pad = int(model.config.audio_pad_token)
        # Stop parity with BatchedThinkerRunner.step_finished (defect B fix):
        # graphed decode halts when text EOS observed AND the last audio
        # layer emits audio_stop. ``None`` falls back to model attributes
        # (matches eager's ``int(getattr(self.model, 'audio_stop_token', 0))``
        # pattern at batched_generation.py:119).
        self.eos_token_id = (
            int(eos_token_id)
            if eos_token_id is not None
            else int(getattr(model, "eos_token_id_2", 2))
        )
        self.audio_stop_token = (
            int(audio_stop_token)
            if audio_stop_token is not None
            else int(getattr(model, "audio_stop_token", 0))
        )
        self.attns = [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]
        self.steps: list[tuple[Any, torch.Tensor, Any, torch.Tensor | None]] = []
        self._bridge_layer = -1
        self._prefill_bridge: torch.Tensor | None = None
        self._captured = False
        self._captured_len = -1
        # The per-step graph COUNT is also baked at the first-requested
        # ``n_steps`` (one graph per decode step). A different
        # ``max_tokens`` budget on a later request must rebuild the step
        # set; the fixed-budget serve path captures exactly once.
        self._captured_n_steps = -1
        self._prefill_len = -1
        self._last_input_ids = None
        self._text_finished = False

    def _reset_pos(self) -> None:
        for a in self.attns:
            a._kv_pos = 0

    def _zero_kv_contents(self) -> None:
        """Zero the full per-attention KV buffer. Each generate must start a
        fresh KV session (like a new production conversation): stale KV from
        a prior prompt must never survive into the current one. In-place zero
        keeps buffer addresses stable for the captured graphs."""
        for a in self.attns:
            if hasattr(a, "_kv_past_key"):
                a._kv_past_key.zero_()
                a._kv_past_value.zero_()

    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run production-shape prefill and return the last text-row logits.
        Fact #3: prefill with the [1, 9, seq] input (audio pads + text) so
        the KV state matches stream_generate's; token0 is sampled from the
        text row's logits.

        Always zero the KV buffer first: a fresh prompt must not inherit
        stale K/V from a previous request at the same buffer addresses
        (otherwise the new prefill's logits diverge as soon as a layer reads
        past the freshly-written slot range). The in-place zero keeps the
        graph-captured tensor addresses stable.
        """
        self._zero_kv_contents()
        self._prefill_len = input_ids.shape[1]
        self._last_input_ids = input_ids
        self._reset_pos()
        prefill_in = _build_omni_input(input_ids, self.audio_pad)
        with torch.no_grad():
            out = self.fwd(self.model, input_ids=prefill_in, past_key_values=None, use_cache=True)
        if self._bridge_layer >= 0:
            from nanovllm_omni.models.minimind_omni.batched_generation import _read_bridge_capture

            bridge = _read_bridge_capture(self.model, self._bridge_layer)
            self._prefill_bridge = bridge[0].detach().to(device="cpu", dtype=torch.float32).clone()
        return out.logits[0, -1].clone()

    def _enable_bridge_capture(self) -> None:
        if self._bridge_layer >= 0:
            return
        from nanovllm_omni.models.minimind_omni.batched_generation import enable_bridge_capture

        self._bridge_layer = enable_bridge_capture(self.model)
        if self._bridge_layer >= 0:
            # Existing graphs omit the new hook and must be rebuilt.
            self._captured = False

    def _needs_recapture(self) -> bool:
        """Pure decision: must the per-step graphs be rebuilt?

        True when not yet captured OR the current prompt length differs from
        the length the existing graphs were captured at (defect #5: offsets
        are baked at that length; a different-length prompt must rebuild)
        OR the requested generation budget differs from the budget the
        graphs were captured at (the per-step graph COUNT is baked at the
        first-requested ``n_steps``; a changed ``max_tokens`` must rebuild).
        Kept as its own method so the CPU contract test exercises the REAL
        code path (not a hand-replicated copy).
        """
        return not (
            self._captured
            and self._captured_len == self._prefill_len
            and self._captured_n_steps == self.n_steps
        )

    def _capture(self, next_token: torch.Tensor) -> None:
        """Capture one graph per AR step over the (prefill-loaded) KV buffers.

        Warmup forwards are rewound before capture so they do not consume a
        decode position. Defect #5: re-capture when the prefill length changed since the last
        capture. Per-step graph offsets are baked at ``_kv_pos = prefill_len
        + k``; reusing them for a different-length prompt misaligns KV, so a
        length change must drop the stale graphs and rebuild.
        """
        if not self._needs_recapture():
            return
        # Prompt length changed -> stale per-step graphs are unsafe. The
        # buffer still holds the PREVIOUS prompt's KV in [min_len : max_len]
        # where the new prefill didn't overwrite. Clear it here (re-capture
        # branch only, same-length reuse stays un-cleared) then re-prefill the
        # current prompt so the rebuilt graphs start from clean KV.
        had_capture = self._captured
        self.steps = []
        self._captured = False
        self._zero_kv_contents()
        # ``generate_tokens`` already prefills the current request before
        # entering _capture.  Only recapture needs another prefill because
        # zeroing the shared buffer would otherwise erase the fresh prompt.
        if had_capture:
            self._prefill(self._last_input_ids)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # Warmup on the side stream FIRST so cuBLAS/cuDNN workspace picks an
        # algorithm and the capture-context fwd only records (no JIT/alloc).
        # torch.cuda.graph(g) internally captures on its own default capture
        # stream and joins back after, so warmup-then-capture on a side stream
        # keeps the recorded kernels' stream bindings consistent. Moving the
        # warmup INSIDE torch.cuda.graph(g) breaks capture (allocator cache
        # pollution).
        # Prefill produces token 0; the decode loop replays at most
        # ``n_steps - 1`` graphs (indices 0 .. n_steps-2).  Capture exactly
        # that many — the old code captured ``n_steps`` and left the last
        # graph unreplayed (wasted warmup + capture + VRAM).
        num_decode_graphs = max(self.n_steps - 1, 0)
        with torch.cuda.stream(side):
            for _ in range(num_decode_graphs):
                inp = _build_omni_input(next_token, self.audio_pad).clone()
                with torch.no_grad():
                    self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
                # The warmup forward above advances the Python KV cursor, but
                # its position must not count toward the graph's replay
                # position.  Rewind to the same cursor before capture so graph
                # k is baked at prefill_len + k, not prefill_len + 2*k + 1.
                for attn in self.attns:
                    attn._kv_pos -= inp.shape[-1]
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g), torch.no_grad():
                    out = self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
                bridge = None
                if self._bridge_layer >= 0:
                    from nanovllm_omni.models.minimind_omni.batched_generation import (
                        _read_bridge_capture,
                    )

                    bridge = _read_bridge_capture(self.model, self._bridge_layer)
                self.steps.append((g, inp, out, bridge))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self._captured = True
        self._captured_len = self._prefill_len
        # Bake the budget the graphs were captured at so a later request
        # with a different ``n_steps`` (per-request ``max_tokens``) forces
        # a rebuild instead of replaying a stale step set.
        self._captured_n_steps = self.n_steps

    def _reset_request_state(
        self,
        *,
        post_eos_padding_count: int = 0,
        internal_stop_token_id: int | None = None,
    ) -> None:
        self._text_finished = False
        self._post_eos_padding_count = post_eos_padding_count
        self._post_eos_started = False
        self._post_eos_remaining = 0
        self._internal_stop_emitted = False
        if internal_stop_token_id is None:
            internal_stop_token_id = getattr(
                getattr(self, "model", None), "internal_stop_token_id", 17
            )
        self._internal_stop_token_id = int(internal_stop_token_id)

    def _next_post_eos_token(self) -> int:
        """Mirror ``BatchedThinkerRunner._next_post_eos_token``."""
        if not self._post_eos_started:
            self._post_eos_started = True
            self._post_eos_remaining = self._post_eos_padding_count
            return int(getattr(getattr(self, "model", None), "enter_token_id", 201))
        if self._post_eos_remaining > 0:
            self._post_eos_remaining -= 1
            return int(getattr(getattr(self, "model", None), "pad_token_id", 0))
        self._internal_stop_emitted = True
        return self._internal_stop_token_id

    def _should_stop(self, token_id: int, audio_codes: list[list[int]]) -> bool:
        """Defect B fix: parity with ``BatchedThinkerRunner.step_finished``.

        Flips ``self._text_finished`` the first time a sampled text token
        equals ``eos_token_id``; returns True once that flag is set AND
        the most recent code on the last audio channel equals
        ``audio_stop_token``. Pure decision method -- kept separate so the
        CPU contract test exercises the real predicate (no hand-replicated
        copy).
        """
        if not self._text_finished and token_id == self.eos_token_id:
            self._text_finished = True
        if getattr(self, "_post_eos_padding_count", 0) > 0:
            return self._internal_stop_emitted
        return (
            self._text_finished and audio_codes[NUM_AUDIO_LAYERS - 1][-1] == self.audio_stop_token
        )

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        seed: int | None = None,
        return_audio: bool = False,
        return_bridge: bool = False,
        post_eos_padding_count: int = 0,
        internal_stop_token_id: int | None = None,
    ) -> Any:
        """Prefill + n-step joint text/audio decode; returns generated text
        token ids (excludes input), and (if ``return_audio``) the 8-channel
        Mimi audio codes.

        Fact #4 mechanism: production draws TEXT then 8 AUDIO codes per step
        through ONE shared per-request ``torch.Generator`` (runner
        ``_sample_text``/``_sample_audio_row`` both pass ``gen=st.gen``).
        To keep the text sequence identical to ``stream_generate``, this
        decoder draws the audio codes through the same ``gen`` (advancing
        its state exactly like production) even when the caller only wants
        text. For WAV output the audio codes are returned as well.

        Stopping (defect B fix): the main loop checks
        ``_should_stop(token_id, audio_codes)`` after each decode step. With
        post-EOS padding enabled, it emits enter, PAD, and internal-stop
        tokens through the same state machine as
        ``BatchedThinkerRunner``; otherwise it stops at text EOS plus
        audio-stop.
        """
        self._reset_request_state(
            post_eos_padding_count=post_eos_padding_count,
            internal_stop_token_id=internal_stop_token_id,
        )
        if return_bridge:
            self._enable_bridge_capture()
        num_layers = NUM_AUDIO_LAYERS
        prefill_logits = self._prefill(input_ids)
        bridge_states: list[torch.Tensor] = []
        history = list(input_ids.reshape(-1).tolist())
        audio_history: list[list[int]] = [[] for _ in range(num_layers)]
        gen = torch.Generator(device=prefill_logits.device)
        gen.manual_seed(seed if seed is not None else torch.initial_seed())

        # seed0: sampled from prefill logits; audio_step -1 => all audio pad
        token_id = sample_text_token(
            prefill_logits,
            history_ids=history,
            temperature=self.temperature,
            top_p=self.top_p,
            rp=self.rp,
            gen=gen,
        )
        text_codes = [token_id]
        audio_codes = [[self.audio_pad] for _ in range(num_layers)]
        history = history + [token_id]
        next_token = torch.tensor([[token_id]], device=prefill_logits.device, dtype=torch.long)
        self._capture(next_token)
        # seed0 stop check: text_finished may flip here, but no audio code
        # was sampled yet; the post-EOS state machine starts on the next step.
        if self._should_stop(token_id, audio_codes):
            return self._result(text_codes, audio_codes, return_audio, return_bridge, bridge_states)
        for k in range(self.n_steps - 1):
            step_index = k + 1  # text token index at this decode step
            # Match ``stream_generate``: the k-th decode step samples audio
            # from layer k (i.e. audio_start = step_index - 0). Eager writes
            # audio starting at the first decode step, so we must too --
            # audio_step == step_index here would drop one audio frame
            # relative to eager, breaking _should_stop parity.
            audio_step = step_index
            g, inp, out, bridge = self.steps[k]
            try:
                inp.copy_(_build_omni_input(next_token, self.audio_pad))
                g.replay()
            except RuntimeError as exc:
                # Drop stale per-step graphs so a later request recaptures,
                # then surface a clear error (parity: talker_cuda_graph.decode
                # invalidates + falls back; here there is no clean eager path
                # because the model is already buffer-ized).
                self.steps = []
                self._captured = False
                raise RuntimeError(
                    f"CUDA Graph replay failed at step {step_index}: {exc}; "
                    "capture state reset — retry will recapture."
                ) from exc
            was_text_finished = self._text_finished
            token_id = sample_text_token(
                out.logits[0, -1],
                history_ids=history,
                temperature=self.temperature,
                top_p=self.top_p,
                rp=self.rp,
                gen=gen,
            )
            if was_text_finished:
                token_id = self._next_post_eos_token()
            text_codes.append(token_id)
            history = history + [token_id]
            # audio draws (same gen => identical RNG advance to production)
            audio_logits = getattr(out, "audio_logits", None)
            for layer in range(num_layers):
                if audio_step >= layer and audio_logits is not None:
                    code = sample_one_audio_layer(
                        audio_logits[layer][0, -1],
                        audio_history[layer],
                        gen=gen,
                    )
                    audio_codes[layer].append(code)
                    audio_history[layer].append(code)
                else:
                    audio_codes[layer].append(self.audio_pad)
            next_token = torch.tensor([[token_id]], device=out.logits.device, dtype=torch.long)
            if return_bridge and bridge is not None:
                bridge_states.append(
                    bridge[0, 0].detach().to(device="cpu", dtype=torch.float32).clone()
                )
            if self._should_stop(token_id, audio_codes):
                break
        return self._result(text_codes, audio_codes, return_audio, return_bridge, bridge_states)

    def _result(
        self,
        text_codes: list[int],
        audio_codes: list[list[int]],
        return_audio: bool,
        return_bridge: bool,
        bridge_states: list[torch.Tensor],
    ) -> Any:
        if return_bridge:
            if self._prefill_bridge is None:
                bridge = torch.empty((0, 0), dtype=torch.float32)
            elif bridge_states:
                bridge = torch.cat((self._prefill_bridge, torch.stack(bridge_states)), dim=0)
            else:
                bridge = self._prefill_bridge
            return text_codes, audio_codes, bridge
        if return_audio:
            return text_codes, audio_codes
        return text_codes


def enable_cuda_graph(
    model: Any,
    n_steps: int = 16,
    max_len: int | None = None,
    *,
    eos_token_id: int | None = None,
    audio_stop_token: int | None = None,
) -> CudaGraphDecoder | None:
    """Install fixed-KV-buffer attention + return a CUDA-Graphed decoder.

    Defaults:
      - ``n_steps=16``: the e2e acceptance figure from report §14/§16.
        Pass a smaller number to capture fewer per-step graphs (the decoder
        captures ``n_steps - 1`` graphs: prefill produces the first token,
        then each graph handles one decode step, stops earlier when text
        EOS + audio stop align).
      - ``max_len=None``: read ``model.config.max_position_embeddings``
        via ``enable_fixed_kv_buffer``; pass an int to override the buffer.
      - ``eos_token_id`` / ``audio_stop_token`` (default None): threaded
        into ``CudaGraphDecoder`` so graphed decode halts at content-natural
        end (parity with eager ``BatchedThinkerRunner.step_finished``).
        None falls back to ``model.eos_token_id_2`` /
        ``model.audio_stop_token`` at decoder construction time.

    Returns ``None`` (not enabled) when CUDA is unavailable, the model
    doesn't carry the MiniMind-O freqs host-reads this integration targets,
    or no attention instance bound a fixed KV buffer. Opt-in: does not
    touch the bundle default path.
    """
    if not torch.cuda.is_available():
        _log.info("enable_cuda_graph: no CUDA; keeping eager path")
        return None
    if not hasattr(model.config, "audio_pad_token"):
        _log.warning("enable_cuda_graph: no audio_pad_token; skipping")
        return None

    # A previously-captured decoder's step set is baked at the first-requested
    # ``n_steps``; if a later request asks for a different budget, update the
    # decoder's budget so the next ``generate_tokens`` triggers a re-capture
    # via ``_needs_recapture``. Fixed-budget serve path (same ``n_steps``
    # every request) leaves the decoder unchanged and stays in capture-once
    # mode.
    existing = getattr(model, "_nanovllm_graph_decoder", None)
    if existing is not None:
        if existing.n_steps != n_steps:
            _log.info(
                "enable_cuda_graph: budget %d -> %d (re-capture on next generate)",
                existing.n_steps,
                n_steps,
            )
            existing.n_steps = n_steps
        return existing

    cls = type(model)
    try:
        src = inspect.getsource(cls.forward)
    except (OSError, TypeError) as exc:
        _log.warning("enable_cuda_graph: cannot read forward source: %s", exc)
        return None
    fwd = _patched_forward(cls, src)
    if fwd is None:
        return None

    enable_fixed_kv_buffer(model, max_len)
    decoder = CudaGraphDecoder(
        model=model,
        fwd=fwd,
        n_steps=n_steps,
        eos_token_id=eos_token_id,
        audio_stop_token=audio_stop_token,
    )
    if not decoder.attns:
        _log.warning("enable_cuda_graph: no attention instances buffer-ized; skipping")
        return None
    _log.info("enable_cuda_graph: %d attention instances buffer-ized", len(decoder.attns))

    model._nanovllm_graph_decoder = decoder
    return decoder
