"""Stage timing + result dataclasses for the four-helper pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .prompts import BENCH_PROMPTS


@dataclass(frozen=True)
class StageTimes:
    """Wall-clock seconds for the four stages of one ``generate_audio`` call."""

    tokenize: float = 0.0
    generate: float = 0.0
    decode: float = 0.0
    wav: float = 0.0

    @property
    def total(self) -> float:
        return self.tokenize + self.generate + self.decode + self.wav

    def as_dict(self) -> dict[str, float]:
        return {
            "tokenize_ms": self.tokenize * 1000.0,
            "generate_ms": self.generate * 1000.0,
            "decode_ms": self.decode * 1000.0,
            "wav_ms": self.wav * 1000.0,
            "total_ms": self.total * 1000.0,
        }


@dataclass(frozen=True)
class RunResult:
    """One ``generate_audio`` run: timings, sizes, max CUDA memory, audio bytes."""

    prompt: str
    stages: StageTimes
    n_tokens: int
    n_samples: int
    max_mem_bytes: int
    wav_bytes: bytes = b""

    def as_csv_row(self) -> dict[str, Any]:
        row = {"prompt": self.prompt}
        row.update(self.stages.as_dict())
        row["n_tokens"] = self.n_tokens
        row["n_samples"] = self.n_samples
        row["max_mem_bytes"] = self.max_mem_bytes
        row["wav_bytes"] = len(self.wav_bytes)
        return row


# Ponytail: per-run max_memory is cheap; reset only if CUDA is reachable.
def _maybe_reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_mem_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return 0


def _call_with_timing(
    label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, float]:
    """Time ``fn`` with ``time.perf_counter`` and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_one(
    bundle: Any,
    prompt: str,
    *,
    max_tokens: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    open_thinking: bool = False,
    run_id: int = 0,
) -> RunResult:
    """Drive one prompt through the four helpers and record timings."""
    # Local imports keep the bench package importable on CPU-only CI.
    import torch

    from nanovllm_omni.models.minimind_omni.stages import (
        decode_audio,
        encode_wav,
        run_generate,
        tokenize_for_generate,
    )

    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    _maybe_reset_cuda_peak()

    t = StageTimes()
    samples: Any = None
    frames: list[list[int]] = []

    with torch.no_grad():
        input_ids, t_tokenize = _call_with_timing(
            "tokenize",
            tokenize_for_generate,
            bundle.tokenizer,
            prompt,
            open_thinking,
        )
        input_ids = input_ids.to(bundle.device)

        frames, t_generate = _call_with_timing(
            "generate",
            run_generate,
            bundle.model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            open_thinking=open_thinking,
        )

        if frames:
            samples, t_decode = _call_with_timing(
                "decode", decode_audio, bundle.mimi, frames, bundle.device
            )
        else:
            t_decode = 0.0

    if frames and samples is not None:
        wav_bytes, t_wav = _call_with_timing("wav", encode_wav, samples)
    else:
        wav_bytes, t_wav = b"", 0.0

    t = StageTimes(
        tokenize=t_tokenize,
        generate=t_generate,
        decode=t_decode,
        wav=t_wav,
    )
    return RunResult(
        prompt=prompt,
        stages=t,
        n_tokens=sum(len(f) for f in frames),
        n_samples=int(samples.shape[0]) if samples is not None else 0,
        max_mem_bytes=_peak_mem_bytes(),
        wav_bytes=wav_bytes,
    )


def run_n(
    bundle: Any,
    prompts: tuple[str, ...] | None = None,
    *,
    n: int = 5,
    warmup: int = 1,
    **kwargs: Any,
) -> list[RunResult]:
    """Run ``n`` cold iterations over ``prompts`` (default :data:`BENCH_PROMPTS`).

    ``warmup`` iterations are run before timing and discarded; ``run_id`` is
    stamped onto each result for downstream CSV de-duplication. Returns
    ``n * len(prompts)`` results in prompt-major order.
    """
    prompts = prompts if prompts is not None else BENCH_PROMPTS
    for _ in range(max(warmup, 0)):
        for prompt in prompts:
            run_one(bundle, prompt, **kwargs)
    results: list[RunResult] = []
    run_id = 0
    for _ in range(n):
        for prompt in prompts:
            results.append(run_one(bundle, prompt, run_id=run_id, **kwargs))
            run_id += 1
    return results
