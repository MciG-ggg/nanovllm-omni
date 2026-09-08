"""Single-graph paged CUDA Graph decode (one graph, replayed many times).

This is the position-independent counterpart to ``cuda_graph``.

The difference in one line
--------------------------
``optim/cuda_graph`` captures ``n_steps - 1`` graphs -- **one graph per
decode position** -- because the fixed-KV buffer's history slice
``[:, :_kv_pos]`` changes shape every AR step, and CUDA Graphs freeze
tensor shapes at capture time.

This module captures **exactly one** decode graph and replays it for
every AR step of a request. That is possible because the paged cache
(``paged_attention``) keeps every tensor shape constant:

- new K/V is written through ``slot_mapping`` (fixed length)
- history is read through ``block_tables`` + ``context_lens``
  (fixed shape; only the *values* change per step)

So a decode step becomes: write new values into the persistent metadata
tensors, then ``graph.replay()``. Same graph object, every step.

Consequences vs the per-position decoder
----------------------------------------
- capture cost is paid once, not ``n_steps - 1`` times
- VRAM holds one graph, not ``n_steps - 1`` graphs
- ``n_steps`` is no longer baked into capture, so changing
  ``max_tokens`` between requests does **not** force a recapture
- prompt length is no longer baked into capture either; it only matters
  if it changes the *window width* (number of block-table columns)

Parity
------
Sampling, the post-EOS state machine and the stop predicate are
inherited verbatim from ``CudaGraphDecoder`` so text/audio stopping
behaviour is identical between the two graph paths. Only KV storage and
graph capture differ.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Any

import torch

from nanovllm_omni.models.minimind_omni._sampling import (
    NUM_AUDIO_LAYERS,
    sample_one_audio_layer,
    sample_text_token,
)
from nanovllm_omni.models.minimind_omni.cuda_graph import (
    CudaGraphDecoder,
    _build_omni_input,
    _patched_forward,
)
from nanovllm_omni.models.minimind_omni.paged_attention import enable_paged_kv_cache

_log = logging.getLogger(__name__)


class PagedCudaGraphDecoder(CudaGraphDecoder):
    """Capture-once / replay-many decoder over a paged KV cache.

    Subclasses ``CudaGraphDecoder`` purely to inherit the request-state
    machine (``_reset_request_state`` / ``_next_post_eos_token`` /
    ``_should_stop`` / ``_result``). Storage, capture and the decode
    loop are all replaced.
    """

    def __init__(
        self,
        model: Any,
        fwd: Any,
        n_steps: int,
        *,
        eos_token_id: int | None = None,
        audio_stop_token: int | None = None,
        # block_size=256 is forced by flash-attn's varlen kernel: the kernel
        # asserts `block_size % 256 == 0` for the paged layout. The fork's
        # ``Sequence.block_size`` default is also 256. Smaller values are
        # rejected at runtime when flash-attn is on the import path.
        block_size: int = 256,
        max_batch_size: int = 1,
    ) -> None:
        # NOTE: deliberately does NOT call super().__init__ -- that one
        # collects fixed-KV-buffer attentions, which this path replaces.
        from nanovllm_omni.models.minimind_omni._sampling import (
            DEFAULT_TEXT_TEMPERATURE,
            DEFAULT_TEXT_TOP_P,
        )

        self.model = model
        self.fwd = fwd
        self.n_steps = n_steps
        self.temperature = DEFAULT_TEXT_TEMPERATURE
        self.top_p = DEFAULT_TEXT_TOP_P
        self.rp = 1.0
        self.audio_pad = int(model.config.audio_pad_token)
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
        self.block_size = block_size
        self.max_batch_size = max_batch_size

        self.cache: Any = None
        # THE graph -- singular, on purpose.
        self.graph: Any = None
        self.graph_input: Any = None
        self.graph_output: Any = None
        self.graph_bridge: Any = None

        self._captured = False
        # Capture is invalidated only by a change in the *window width*
        # (block-table columns). Not by prompt length, not by n_steps --
        # that is the whole point of the paged layout.
        self._captured_window = -1
        self._max_blocks_per_seq = -1

        self._bridge_layer = -1
        self._prefill_bridge: Any = None
        self._text_finished = False
        self._seq: Any = None

    def _required_blocks(self, prompt_len: int) -> int:
        """Block-table columns needed for ``prompt_len + n_steps`` tokens.

        This is the only quantity that can invalidate the captured graph,
        since it sets the gathered-window shape.
        """
        total = prompt_len + self.n_steps + 1
        return (total + self.block_size - 1) // self.block_size

    def _ensure_cache(self, prompt_len: int) -> None:
        """Install / resize the paged cache for this prompt length.

        Sizes the pool tightly (``max_batch_size * blocks_per_seq``)
        rather than by free-VRAM share: one of the reasons to move off
        per-position graphs is to *reduce* VRAM, so we do not want a
        greedy pool here.
        """
        need = self._required_blocks(prompt_len)
        if self.cache is not None and need <= self._max_blocks_per_seq:
            return
        # +1: block 0 is reserved as the padding/sentinel block.
        num_blocks = self.max_batch_size * need + 1
        self.cache = enable_paged_kv_cache(
            self.model,
            num_blocks=num_blocks,
            block_size=self.block_size,
            max_batch_size=self.max_batch_size,
            max_num_new_tokens=need * self.block_size,
        )
        if self.cache is None:
            raise RuntimeError("PagedCudaGraphDecoder: no paged-capable attention found")
        self._max_blocks_per_seq = need
        # Window shape changed -> the captured graph is stale.
        self._captured = False
        self.graph = None

    def _reset_blocks(self) -> None:
        """Return this request's blocks and zero the pool.

        In-place zero keeps the graph-captured tensor addresses stable
        while guaranteeing a fresh prompt never reads a previous
        request's K/V.
        """
        if self._seq is not None:
            # already-freed is not fatal
            with contextlib.suppress(RuntimeError):
                self.cache.release(self._seq)
            self._seq = None
        if self.cache is not None and self.cache.kv_cache is not None:
            self.cache.kv_cache.zero_()

    def _sync_prefill_ctx(self, prompt_len: int) -> None:
        context = self.cache._ctx
        context.is_prefill = True
        bt = self._seq.block_table
        slots = [
            bt[i // self.block_size] * self.block_size + (i % self.block_size)
            for i in range(prompt_len)
        ]
        context.slot_mapping.fill_(-1)
        context.slot_mapping[:prompt_len] = torch.tensor(
            slots, dtype=context.slot_mapping.dtype, device=context.slot_mapping.device
        )
        context.context_lens.zero_()
        context.context_lens[0] = prompt_len
        # The decoder currently runs one request per graph. FlashAttention's
        # varlen prefill still needs explicit [0, prompt_len] boundaries.
        context.cu_seqlens_q.zero_()
        context.cu_seqlens_q[1] = prompt_len
        context.cu_seqlens_k.zero_()
        context.cu_seqlens_k[1] = prompt_len
        context.max_seqlen_q = prompt_len
        context.max_seqlen_k = prompt_len
        context.block_tables.zero_()
        context.block_tables[0, : len(bt)] = torch.tensor(
            bt, dtype=context.block_tables.dtype, device=context.block_tables.device
        )

    def _sync_decode_ctx(self) -> None:
        """Point the metadata tensors at the newest token.

        Called immediately before ``graph.replay()``. Writes land in the
        same buffers the captured graph reads, so the single graph sees
        fresh values every step -- this is what makes replay-many work.
        """
        seq = self._seq
        context = self.cache._ctx
        context.is_prefill = False
        slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        context.slot_mapping.fill_(-1)
        context.slot_mapping[0] = slot
        context.context_lens.zero_()
        context.context_lens[0] = len(seq)
        bt = seq.block_table
        context.block_tables.zero_()
        context.block_tables[0, : len(bt)] = torch.tensor(
            bt, dtype=context.block_tables.dtype, device=context.block_tables.device
        )

    def _prefill(self, input_ids: Any) -> Any:
        prompt = input_ids.reshape(-1).tolist()
        prompt_len = len(prompt)
        self._ensure_cache(prompt_len)
        self._reset_blocks()
        self._seq = self.cache.new_sequence(prompt)
        self._sync_prefill_ctx(prompt_len)

        prefill_in = _build_omni_input(input_ids, self.audio_pad)
        with torch.no_grad():
            out = self.fwd(self.model, input_ids=prefill_in, past_key_values=None, use_cache=True)
        if self._bridge_layer >= 0:
            from nanovllm_omni.models.minimind_omni.batched_generation import _read_bridge_capture

            bridge = _read_bridge_capture(self.model, self._bridge_layer)
            self._prefill_bridge = bridge[0].detach().to(device="cpu", dtype=torch.float32).clone()
        return out.logits[0, -1].clone()

    def _needs_recapture(self) -> bool:
        """Only a window-width change invalidates the single graph.

        Contrast with the per-position decoder, which must also compare
        prompt length AND ``n_steps``.
        """
        return not (self._captured and self._captured_window == self._max_blocks_per_seq)

    def _capture(self, next_token: Any) -> None:
        """Capture ONE decode graph.

        The captured op stream reads ``slot_mapping`` / ``context_lens``
        / ``block_tables`` from fixed addresses, so replaying it after
        rewriting those values advances the request by one token -- for
        any position, without recapture.
        """
        if not self._needs_recapture():
            return
        self.graph = None
        self._captured = False

        # Decode-shaped context so the graph bakes the decode branch.
        self._sync_decode_ctx()
        inp = _build_omni_input(next_token, self.audio_pad).clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # Warmup so cuBLAS/cuDNN pick algorithms + allocate workspace
            # before capture (capture must only record, never allocate).
            with torch.no_grad():
                self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                out = self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
            bridge = None
            if self._bridge_layer >= 0:
                from nanovllm_omni.models.minimind_omni.batched_generation import (
                    _read_bridge_capture,
                )

                bridge = _read_bridge_capture(self.model, self._bridge_layer)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = graph
        self.graph_input = inp
        self.graph_output = out
        self.graph_bridge = bridge
        self._captured = True
        self._captured_window = self._max_blocks_per_seq
        _log.info(
            "paged CUDA Graph captured: 1 graph, window=%d blocks (replayed every AR step)",
            self._max_blocks_per_seq,
        )

    def generate_tokens(
        self,
        input_ids: Any,
        *,
        seed: int | None = None,
        return_audio: bool = False,
        return_bridge: bool = False,
        post_eos_padding_count: int = 0,
        internal_stop_token_id: int | None = None,
    ) -> Any:
        """Prefill eagerly, then replay the SAME graph for every step.

        Public contract is identical to
        ``CudaGraphDecoder.generate_tokens`` so callers (thinker
        ``run_generate``, benches) are unchanged.
        """
        self._reset_request_state(
            post_eos_padding_count=post_eos_padding_count,
            internal_stop_token_id=internal_stop_token_id,
        )
        if return_bridge:
            self._enable_bridge_capture()

        num_layers = NUM_AUDIO_LAYERS
        prefill_logits = self._prefill(input_ids)
        bridge_states: list[Any] = []
        history = list(input_ids.reshape(-1).tolist())
        audio_history: list[list[int]] = [[] for _ in range(num_layers)]
        gen = torch.Generator(device=prefill_logits.device)
        gen.manual_seed(seed if seed is not None else torch.initial_seed())

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

        # Token 0 joins the sequence before capture so the captured
        # context describes a real decode step.
        self.cache.append_token(self._seq, token_id)
        self._capture(next_token)

        if self._should_stop(token_id, audio_codes):
            return self._result(text_codes, audio_codes, return_audio, return_bridge, bridge_states)

        out = self.graph_output
        for k in range(self.n_steps - 1):
            step_index = k + 1
            audio_step = step_index
            try:
                # Same graph object every iteration; only the values in
                # the persistent input/metadata tensors change.
                self.graph_input.copy_(_build_omni_input(next_token, self.audio_pad))
                self._sync_decode_ctx()
                self.graph.replay()
            except RuntimeError as exc:
                self.graph = None
                self._captured = False
                raise RuntimeError(
                    f"paged CUDA Graph replay failed at step {step_index}: {exc}; "
                    "capture state reset - retry will recapture."
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
            if return_bridge and self.graph_bridge is not None:
                bridge_states.append(
                    self.graph_bridge[0, 0].detach().to(device="cpu", dtype=torch.float32).clone()
                )
            # Grow the sequence so the next _sync_decode_ctx points at the
            # right slot (and allocates a block on boundary crossings).
            self.cache.append_token(self._seq, token_id)

            if self._should_stop(token_id, audio_codes):
                break

        return self._result(text_codes, audio_codes, return_audio, return_bridge, bridge_states)


def enable_paged_cuda_graph(
    model: Any,
    n_steps: int = 16,
    *,
    eos_token_id: int | None = None,
    audio_stop_token: int | None = None,
    block_size: int = 256,
    max_batch_size: int = 1,
) -> PagedCudaGraphDecoder | None:
    """Install the paged single-graph decoder.

    Mirrors ``enable_cuda_graph``'s signature and gating so the two graph
    backends are drop-in swappable at the ``run_generate`` seam.

    Unlike ``enable_cuda_graph``, a later request with a different
    ``n_steps`` does not invalidate the capture -- ``n_steps`` is not a
    graph shape here.

    Returns ``None`` when CUDA is unavailable or the model is not the
    MiniMind-O shape this integration targets.
    """
    if not torch.cuda.is_available():
        _log.info("enable_paged_cuda_graph: no CUDA; keeping eager path")
        return None
    if not hasattr(model.config, "audio_pad_token"):
        _log.warning("enable_paged_cuda_graph: no audio_pad_token; skipping")
        return None

    existing = getattr(model, "_nanovllm_paged_graph_decoder", None)
    if existing is not None:
        # n_steps is NOT a capture shape in the paged layout, so this is a
        # plain field update -- no recapture implied.
        existing.n_steps = n_steps
        return existing

    cls = type(model)
    try:
        src = inspect.getsource(cls.forward)
    except (OSError, TypeError) as exc:
        _log.warning("enable_paged_cuda_graph: cannot read forward source: %s", exc)
        return None
    fwd = _patched_forward(cls, src)
    if fwd is None:
        return None

    decoder = PagedCudaGraphDecoder(
        model=model,
        fwd=fwd,
        n_steps=n_steps,
        eos_token_id=eos_token_id,
        audio_stop_token=audio_stop_token,
        block_size=block_size,
        max_batch_size=max_batch_size,
    )
    model._nanovllm_paged_graph_decoder = decoder
    return decoder


__all__ = ["PagedCudaGraphDecoder", "enable_paged_cuda_graph"]
