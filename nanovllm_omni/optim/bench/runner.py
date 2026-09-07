"""Stage timing + result dataclasses for the four-helper pipeline (TK-011)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .prompts import BenchPrompt


@dataclass(frozen=True)
class StageTimes:
    """Per-stage wall-clock + GPU-only milliseconds for one ``run_one`` call."""

    tokenize_ms: float = 0.0
    generate_ms: float = 0.0
    decode_ms: float = 0.0
    wav_ms: float = 0.0
    # GPU-only time per stage (0 if CUDA unavailable or the stage is CPU-only).
    generate_cuda_ms: float = 0.0
    decode_cuda_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tokenize_ms + self.generate_ms + self.decode_ms + self.wav_ms

    @property
    def total_cuda_ms(self) -> float:
        return self.generate_cuda_ms + self.decode_cuda_ms

    @property
    def cpu_dispatch_ms(self) -> float:
        """Wall time minus CUDA time for gen+dec stages -- CPU dispatch + sync overhead."""
        cpu = self.generate_ms - self.generate_cuda_ms
        cpu += self.decode_ms - self.decode_cuda_ms
        return max(cpu, 0.0)


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

    def as_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "run_idx": self.run_idx,
            "tokenize_ms": f"{self.times.tokenize_ms:.6f}",
            "generate_ms": f"{self.times.generate_ms:.6f}",
            "decode_ms": f"{self.times.decode_ms:.6f}",
            "wav_ms": f"{self.times.wav_ms:.6f}",
            "total_ms": f"{self.times.total_ms:.6f}",
            "frames": self.frames,
            "vram_mb": f"{self.vram_peak_mb:.3f}",
        }
        # Per-stage detail (TK-011 follow-up; defaults to 0.0 on CPU).
        row["generate_cuda_ms"] = f"{self.times.generate_cuda_ms:.6f}"
        row["decode_cuda_ms"] = f"{self.times.decode_cuda_ms:.6f}"
        row["cpu_dispatch_ms"] = f"{self.times.cpu_dispatch_ms:.6f}"
        row["generate_per_step_ms"] = (
            f"{self.times.generate_ms / self.frames:.6f}" if self.frames else "0.000000"
        )
        return row


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
    if isinstance(prompt, BenchPrompt):
        return prompt
    return BenchPrompt(id="ad-hoc", text=str(prompt))


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _cuda_event_pair() -> tuple[Any, Any]:
    """Return ``(start_event, end_event)`` or ``(None, None)`` if CUDA is unavailable."""
    import torch

    if not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _measure_cuda_ms(start: Any, end: Any) -> float:
    """Return milliseconds between two CUDA events; 0 if either is None."""
    import torch

    if start is None or end is None:
        return 0.0
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def run_one(
    bundle: Any,
    prompt: BenchPrompt | str,
    *,
    seed: int = 42,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    open_thinking: bool = False,
    use_thinker_cuda_graph: bool = False,
    run_idx: int = 0,
    max_tokens_tolerance: int = 4,
) -> RunResult:
    """Drive one prompt through the four helpers and record per-stage times.

    Seeds torch's RNG so repeated calls with the same ``seed`` produce the
    same audio bytes. Resets and reads the CUDA peak memory counter when
    CUDA is available. Wraps the ``generate`` and ``decode`` helper calls
    with ``torch.cuda.Event`` to capture GPU-only time per stage.

    Raises ``ValueError`` if the model yields more than ``max_tokens +
    max_tokens_tolerance`` audio frames.
    """
    import torch

    from nanovllm_omni.models.minimind_omni import (
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
    frames: list[list[int]] = []
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

        gen_start, gen_end = _cuda_event_pair()
        if gen_start is not None:
            gen_start.record()
        t0 = time.perf_counter()
        frames = run_generate(
            bundle.model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            open_thinking=open_thinking,
            use_thinker_cuda_graph=use_thinker_cuda_graph,
            seed=seed,
        )
        t_generate_ms = _ms_since(t0)
        if gen_end is not None:
            gen_end.record()
        t_generate_cuda_ms = _measure_cuda_ms(gen_start, gen_end)

        if not frames:
            t_decode_ms = 0.0
            t_decode_cuda_ms = 0.0
        else:
            dec_start, dec_end = _cuda_event_pair()
            if dec_start is not None:
                dec_start.record()
            t0 = time.perf_counter()
            samples = decode_audio(bundle.mimi, frames, bundle.device)
            t_decode_ms = _ms_since(t0)
            if dec_end is not None:
                dec_end.record()
            t_decode_cuda_ms = _measure_cuda_ms(dec_start, dec_end)

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
            generate_cuda_ms=t_generate_cuda_ms,
            decode_cuda_ms=t_decode_cuda_ms,
        ),
        frames=len(frames),
        vram_peak_mb=_vram_peak_mb(),
        audio_bytes=wav_bytes,
    )


def run_one_full(
    omni: Any,
    prompt: BenchPrompt | str,
    *,
    seed: int = 42,
    max_tokens: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    run_idx: int = 0,
) -> RunResult:
    """Time the full three-stage E2E through ``Omni.generate`` (real weights).

    The whole pipeline (thinker -> talker -> MTP -> code2wav) is one timed
    unit; wall time goes into ``StageTimes.generate_ms`` (the per-stage
    tokenize/decode split is thinker-bench-only). ``frames`` is derived from
    the emitted WAV length (Mimi is 12.5 Hz, 24 kHz mono -> 1920 samples /
    16-bit frame). CPU fallback: returns a well-formed zero-ish row.
    """
    import torch

    from nanovllm_omni import SamplingParams

    p = _normalize_prompt(prompt)
    torch.manual_seed(seed)
    _maybe_reset_cuda_peak()

    gen_start, gen_end = _cuda_event_pair()
    if gen_start is not None:
        gen_start.record()
    t0 = time.perf_counter()
    try:
        outs = omni.generate(
            p.text,
            SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        t_generate_ms = _ms_since(t0)
    except Exception:
        t_generate_ms = _ms_since(t0)
        return RunResult(
            prompt_id=p.id,
            seed=seed,
            run_idx=run_idx,
            times=StageTimes(generate_ms=t_generate_ms),
            frames=0,
            vram_peak_mb=_vram_peak_mb(),
        )
    if gen_end is not None:
        gen_end.record()
    t_generate_cuda_ms = _measure_cuda_ms(gen_start, gen_end)

    out = outs[0] if isinstance(outs, list) else outs
    mm = getattr(out, "multimodal_output", None) or {}
    try:
        audio = mm["audio"] if isinstance(mm, dict) else mm.get("audio")
    except (KeyError, AttributeError):
        audio = getattr(mm, "audio", None)
    wav_bytes: bytes = b""
    frames = 0
    if audio is not None and getattr(audio, "data", None):
        wav_bytes = bytes(audio.data)
        # WAV header is 44 bytes; mono 16-bit PCM -> bytes_per_frame = 1920 * 2.
        data_bytes = max(len(wav_bytes) - 44, 0)
        frames = data_bytes // (1920 * 2)

    return RunResult(
        prompt_id=p.id,
        seed=seed,
        run_idx=run_idx,
        times=StageTimes(
            generate_ms=t_generate_ms,
            generate_cuda_ms=t_generate_cuda_ms,
        ),
        frames=frames,
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


def run_n_full(
    omni: Any,
    prompt: BenchPrompt | str,
    *,
    n: int = 5,
    warmup: int = 1,
    **kwargs: Any,
) -> list[RunResult]:
    """Full-E2E analogue of ``run_n``: warmup then n timed ``Omni.generate``."""
    for _ in range(max(warmup, 0)):
        run_one_full(omni, prompt, **kwargs)
    return [run_one_full(omni, prompt, run_idx=i, **kwargs) for i in range(n)]
