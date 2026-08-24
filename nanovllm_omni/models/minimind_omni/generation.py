"""Project-owned streaming MiniMind-O generation loop.

The single-request path (``stream_generate``) is a thin wrapper that drives a
``BatchedThinkerRunner`` with ``max_batch=1`` and yields one
``(text_chunk, audio_frame)`` pair per decode step. All per-step work
(forward, sampling, EOS bookkeeping, frame emission, open_thinking audio
gating) is shared with the batched engine path used by
``engine.run_batched_generate``.

The earlier hand-rolled ``stream_generate_optimized`` was retired because it
duplicated ~120 lines of per-step logic with ``BatchedThinkerRunner``; now
both paths route through one MiniMind generation loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from .batched_generation import BatchedThinkerRunner

__all__ = ["stream_generate"]


def stream_generate(
    model: Any,
    input_ids: Any,
    *,
    eos_token_id: int | None = 2,
    max_new_tokens: int = 1024,
    temperature: float = 0.75,
    top_p: float = 0.90,
    rp: float = 1.0,
    use_cache: bool = True,  # accepted for API parity; runner always uses cache
    return_audio_codes: bool = False,  # accepted for API parity; runner always returns
    open_thinking: bool = False,
    **_kwargs: Any,
) -> Iterator[tuple[Any, Any]]:
    """Stream MiniMind-O output one decode step at a time.

    Equivalent to ``run_batched_generate([prompt], max_batch=1)`` driven by
    a one-slot ``OmniScheduler``. ``text_chunk`` is a ``[1, N]`` tensor of
    all generated text tokens (or ``None`` after EOS); ``audio_frame`` is a
    list of 8 ints (Mimi codebook frame) or ``None`` until the 8th decode
    step. The generator terminates once ``BatchedThinkerRunner.step_finished``
    flips True (text EOS + last audio layer stopped, or ``max_new_tokens``).
    """
    import torch

    from nanovllm_omni.engine.sched import OmniScheduler

    cfg = getattr(model, "config", None)
    max_seq = int(getattr(cfg, "max_position_embeddings", 4096))
    sched = OmniScheduler(max_batch=1, max_seq=max_seq)
    runner = BatchedThinkerRunner(
        SimpleNamespace(model=model),
        sched,
        temperature=temperature,
        top_p=top_p,
        rp=rp,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id if eos_token_id is not None else 2,
        open_thinking=open_thinking,
    )
    rid = runner.add_request(input_ids[0].tolist())

    seen_frames = 0
    while sched.has_requests():
        out = sched.schedule()
        if out.is_empty:
            break
        prefilled: set[str] = set()
        finished: set[str] = set()
        for group in out.prefill_groups:
            if rid in group.req_ids:
                runner.prefill_group(group)
                prefilled.add(rid)
        for group in out.decode_groups:
            if rid not in group.req_ids:
                continue
            runner.decode_group(group)
            st = runner.states[rid]
            text_chunk = torch.as_tensor(
                st.text_tokens,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).unsqueeze(
                0
            )  # [1, N], matches the old text_buffer[:, start_pos:current_len]
            audio_frame = st.frames[seen_frames] if len(st.frames) > seen_frames else None
            seen_frames = len(st.frames)
            if st.text_finished:
                yield None, audio_frame
            else:
                yield text_chunk, audio_frame
            if runner.step_finished(rid):
                finished.add(rid)
        sched.update_from_output(prefilled=prefilled, finished=finished)
        if finished:
            return
