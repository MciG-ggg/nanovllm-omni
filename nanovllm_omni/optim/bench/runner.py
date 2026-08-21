"""Stage timing + result dataclasses for the four-helper pipeline (TK-011)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .prompts import BenchPrompt


@dataclass(frozen=True)
class StageTimes:
    """Per-stage wall-clock milliseconds for one ``run_one`` call."""

    tokenize_ms: float = 0.0
    generate_ms: float = 0.0
    decode_ms: float = 0.0
    wav_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tokenize_ms + self.generate_ms + self.decode_ms + self.wav_ms


@dataclass(frozen=True)
class RunResult:
    """One timed ``run_one`` invocation: timings, frame count, VRAM peak, audio bytes."""

    prompt_id: str
    seed: int
    run_idx: int
    times: StageTimes
    frames: int
    vram_peak_mb: float
    audio_bytes: bytes = b""


def _maybe_reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_mb() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def _normalize_prompt(prompt: BenchPrompt | str) -> BenchPrompt:
    """Accept a BenchPrompt or a raw string; return BenchPrompt."""
    if isinstance(prompt, BenchPrompt):
        return prompt
    return BenchPrompt(id="ad-hoc", text=str(prompt))


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def run_one(
    bundle: Any,
    prompt: BenchPrompt | str,
    *,
    seed: int = 42,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    open_thinking: bool = False,
    run_idx: int = 0,
    max_tokens_tolerance: int = 4,
) -> RunResult:
    """Drive one prompt through the four helpers and record per-stage times.

    Seeds torch's RNG so repeated calls with the same ``seed`` produce the
    same audio bytes (the upstream MiniMind generate loop consults torch
    RNG via the temperature/top_p sampling). Resets and reads the CUDA peak
    memory counter when CUDA is available.

    Raises ``ValueError`` if the model yields more than ``max_tokens +
    max_tokens_tolerance`` audio frames (i.e. ``max_tokens`` was not honored).
    """
    import torch

    from nanovllm_omni.models.minimind_omni.stages import (
        decode_audio,
        encode_wav,
        run_generate,
        tokenize_for_generate,
    )

    p = _normalize_prompt(prompt)
    torch.manual_seed(seed)
    _maybe_reset_cuda_peak()

    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)

    samples: Any = None
    with torch.no_grad():
        t0 = time.perf_counter()
        if p.system is not None:
            input_ids = tokenize_for_generate(
                bundle.tokenizer,
                p.text,
                open_thinking,
                messages=[
                    {"role": "system", "content": p.system},
                    {"role": "user", "content": p.text},
                ],
            )
        else:
            input_ids = tokenize_for_generate(bundle.tokenizer, p.text, open_thinking)
        input_ids = input_ids.to(bundle.device)
        t_tokenize_ms = _ms_since(t0)

        t0 = time.perf_counter()
        frames = run_generate(
            bundle.model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            open_thinking=open_thinking,
        )
        t_generate_ms = _ms_since(t0)

        if not frames:
            t_decode_ms = 0.0
        else:
            t0 = time.perf_counter()
            samples = decode_audio(bundle.mimi, frames, bundle.device)
            t_decode_ms = _ms_since(t0)

    if len(frames) > max_tokens + max_tokens_tolerance:
        raise ValueError(
            f"max_tokens={max_tokens} not honored: "
            f"got {len(frames)} frames (tolerance {max_tokens_tolerance})"
        )

    if frames and samples is not None:
        t0 = time.perf_counter()
        wav_bytes = encode_wav(samples)
        t_wav_ms = _ms_since(t0)
    else:
        wav_bytes, t_wav_ms = b"", 0.0

    return RunResult(
        prompt_id=p.id,
        seed=seed,
        run_idx=run_idx,
        times=StageTimes(
            tokenize_ms=t_tokenize_ms,
            generate_ms=t_generate_ms,
            decode_ms=t_decode_ms,
            wav_ms=t_wav_ms,
        ),
        frames=len(frames),
        vram_peak_mb=_vram_peak_mb(),
        audio_bytes=wav_bytes,
    )


def run_n(
    bundle: Any,
    prompt: BenchPrompt | str,
    *,
    n: int = 5,
    warmup: int = 1,
    **kwargs: Any,
) -> list[RunResult]:
    """Run ``warmup`` discarded iterations then ``n`` timed iterations of one prompt."""
    for _ in range(max(warmup, 0)):
        run_one(bundle, prompt, **kwargs)
    return [run_one(bundle, prompt, run_idx=i, **kwargs) for i in range(n)]
