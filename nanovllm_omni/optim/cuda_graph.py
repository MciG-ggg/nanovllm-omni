"""CUDA Graph streaming decode for MiniMind-O (plan §3.3).

Wraps the fixed-KV-buffer attention (``enable_fixed_kv_buffer``, report
§23-§25) in a per-step CUDA Graph capture/replay loop, collapsing the
~8 000 per-generate `cudaLaunchKernel` calls into a handful of graph
replays (~7.5x measured on RTX 3050, report §24).

Protocol (locked by GPU experiments in
`docs/perf/ncu-generate-kernels-2026-09-01.md`):

- **capture once, replay many** (§25): re-capturing per run folds
  capture-time RNG/state side effects into the loop and breaks
  determinism. A decoder captures `n_steps` per-step graphs the first
  time it runs, then replays them on every later call.
- host-side multinomial sampling lives OUTSIDE the graphs (§24/§25 keep
  sampling's RNG host-side and deterministic).
- the two freqs host-reads in the upstream forward are the only capture
  blockers (§13/§18). This module neutralizes them by compiling a copy of
  the forward with the checks set to ``if False`` (warmup proves them dead)
  and calling that copy — it never rebinds the model class.

Opt-in: ``enable_cuda_graph`` returns ``None`` when CUDA is unavailable or
the model isn't the MiniMind-O upstream shape. The caller keeps the eager
path in that case.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
from typing import Any

import torch

from nanovllm_omni.models.minimind_omni._sampling import (
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_P,
    sample_one_audio_layer,
    sample_text_token,
)
from nanovllm_omni.models.minimind_omni.attention import enable_fixed_kv_buffer

_log = logging.getLogger(__name__)


def _patched_forward(cls: Any, src: str) -> Any:
    """Compile a copy of cls.forward with the two freqs `[0,0]` host-read
    checks neutralized (proven capture blockers, §13/§18). Returns a plain
    function `fwd(model, **kwargs)` — never rebinds the class."""
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    if "if False:" not in patched:
        _log.warning("enable_cuda_graph: no freqs host-reads to neutralize; skipping")
        return None
    namespace = dict(cls.forward.__globals__)
    namespace["__name__"] = cls.__module__
    namespace["__qualname__"] = cls.__qualname__ + ".forward_graph"
    exec(compile(textwrap.dedent(patched), "<enable_cuda_graph>", "exec"), namespace)
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

    Each ``self.steps[k]`` is ``(graph, static_input, output_output)``: the
    captured CUDAGraph, the input tensor pinned to the capture-time address,
    and the output reference held by the graph.
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
        # rp = repetition_penalty (1.0 = no penalty; matches SamplingParams.repetition_penalty).
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
        self.steps: list[tuple[Any, torch.Tensor, Any]] = []
        self._captured = False
        self._captured_len = -1
        self._prefill_len = -1
        self._last_input_ids = None
        # defect B: text EOS state flag, flips on first sampled EOS token.
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
        text row's logits."""
        self._prefill_len = input_ids.shape[1]
        self._last_input_ids = input_ids
        self._reset_pos()
        prefill_in = _build_omni_input(input_ids, self.audio_pad)
        with torch.no_grad():
            out = self.fwd(self.model, input_ids=prefill_in, past_key_values=None, use_cache=True)
        # text row is the last channel: logits[0, -1] is the last text position
        return out.logits[0, -1].clone()

    def _needs_recapture(self) -> bool:
        """Pure decision: must the per-step graphs be rebuilt?

        True when not yet captured OR the current prompt length differs from
        the length the existing graphs were captured at (defect #5: offsets
        are baked at that length; a different-length prompt must rebuild).
        Kept as its own method so the CPU contract test exercises the REAL
        code path (not a hand-replicated copy).
        """
        return not (self._captured and self._captured_len == self._prefill_len)

    def _capture(self, next_token: torch.Tensor) -> None:
        """Capture one graph per AR step over the (prefill-loaded) KV buffers.

        Defect #5: re-capture when the prefill length changed since the last
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
        self.steps = []
        self._captured = False
        self._zero_kv_contents()
        self._prefill(self._last_input_ids)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # Warmup on the side stream FIRST so cuBLAS/cuDNN workspace picks an
        # algorithm and the capture-context fwd only records (no JIT/alloc).
        # Both warmup and capture live on the same stream so the captured
        # kernels inherit the side stream's bindings. Moving the warmup INSIDE
        # torch.cuda.graph(g) breaks capture (allocator cache pollution).
        with torch.cuda.stream(side):
            for _ in range(self.n_steps):
                inp = _build_omni_input(next_token, self.audio_pad).clone()
                with torch.no_grad():
                    self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g), torch.no_grad():
                    out = self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
                self.steps.append((g, inp, out))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self._captured = True
        self._captured_len = self._prefill_len

    # -- programmatic API -------------------------------------------------
    def _should_stop(self, tok: int, audio_codes: list[list[int]]) -> bool:
        """Defect B fix: parity with ``BatchedThinkerRunner.step_finished``.

        Flips ``self._text_finished`` the first time a sampled text token
        equals ``eos_token_id``; returns True once that flag is set AND
        the most recent code on the last audio channel equals
        ``audio_stop_token``. Pure decision method -- kept separate so the
        CPU contract test exercises the real predicate (no hand-replicated
        copy).
        """
        if not self._text_finished and tok == self.eos_token_id:
            self._text_finished = True
        return self._text_finished and audio_codes[7][-1] == self.audio_stop_token

    def generate_tokens(
        self, input_ids: torch.Tensor, *, seed: int | None = None, return_audio: bool = False
    ) -> list[int] | tuple[list[int], list[list[int]]]:
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
        ``_should_stop(tok, audio_codes)`` after each decode step and
        breaks on match -- matches ``BatchedThinkerRunner.step_finished``.
        The ``torch.Generator`` advance stops at the content-natural end
        (no dummy draws), matching eager parity up to the stop step.
        """
        num_layers = 8
        prefill_logits = self._prefill(input_ids)
        history = list(input_ids.reshape(-1).tolist())
        audio_history: list[list[int]] = [[] for _ in range(num_layers)]
        gen = torch.Generator(device=prefill_logits.device)
        gen.manual_seed(seed if seed is not None else torch.initial_seed())

        # seed0: sampled from prefill logits; audio_step -1 => all audio pad
        tok = sample_text_token(
            prefill_logits,
            history_ids=history,
            temperature=self.temperature,
            top_p=self.top_p,
            rp=self.rp,
            gen=gen,
        )
        text_codes = [tok]
        audio_codes = [[self.audio_pad] for _ in range(num_layers)]
        history = history + [tok]
        next_token = torch.tensor([[tok]], device=prefill_logits.device, dtype=torch.long)
        self._capture(next_token)
        # seed0 stop check: text_finished flag may flip here, but no audio
        # code sampled yet -> break only triggers when both gates fire.
        if self._should_stop(tok, audio_codes):
            if return_audio:
                return text_codes, audio_codes
            return text_codes
        for k in range(self.n_steps - 1):
            step_index = k + 1  # text token index at this decode step
            audio_step = step_index - 1  # audio lags text by one position
            g, inp, out = self.steps[k]
            inp.copy_(_build_omni_input(next_token, self.audio_pad))
            g.replay()
            tok = sample_text_token(
                out.logits[0, -1],
                history_ids=history,
                temperature=self.temperature,
                top_p=self.top_p,
                rp=self.rp,
                gen=gen,
            )
            text_codes.append(tok)
            history = history + [tok]
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
            next_token = torch.tensor([[tok]], device=out.logits.device, dtype=torch.long)
            # defect B: halt at content-natural end; ``_should_stop`` flips
            # _text_finished on EOS, returns True once last-layer audio_stop.
            if self._should_stop(tok, audio_codes):
                break
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
        Pass a smaller number to capture fewer per-step graphs (less
        cold-start cost; the decoder emits at most that many decode
        tokens -- defect B fix: stops earlier when text EOS + audio stop
        align).
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
    existing = getattr(model, "_nanovllm_graph_decoder", None)
    if existing is not None:
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
