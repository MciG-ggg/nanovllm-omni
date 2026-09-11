"""Profile bench for sd-turbo (Phase 1 baseline).

Runs the project's SdTurboPipeline (``_sd_turbo_stage`` factory) under
Kineto capture, writes one CSV row per input with p50 / p99 / peak VRAM,
plus a per-run Kineto trace to ``docs/perf/sd-turbo-<date>.trace.json.gz``
and an env snapshot at ``docs/perf/sd-turbo-<date>.env.txt``.

See ADR 0001 for the 4-phase protocol. This file implements Phase 1
only: no CUDA Graph, no fusion patches, no per-stage NVTX markers yet --
Phase 2 cell design follows the bottleneck finding from this profile.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch.cuda.nvtx as nvtx

from .env import git_commit, gpu_label
from .sd_turbo_prompts import SD_TURBO_INPUTS, SdTurboInput

CSV_COLUMNS: tuple[str, ...] = (
    "gpu",
    "commit",
    "input_id",
    "seed",
    "wall_p50_ms",
    "wall_p99_ms",
    "n_runs",
    "vram_peak_mb",
)


def _summarize(walls: Sequence[float]) -> tuple[float, float, float]:
    """(p50, p99, peak) over the timed walls. Zeros if fewer than 2 runs."""
    if len(walls) < 2:
        only = walls[0] if walls else 0.0
        return (only, 0.0, only)
    cuts = statistics.quantiles(walls, n=100, method="inclusive")
    return (cuts[49], cuts[98], max(walls))


def _row(
    sd_input: SdTurboInput,
    walls: Sequence[float],
    gpu: str,
    commit: str,
    vram: float,
) -> dict[str, Any]:
    p50, p99, _peak = _summarize(list(walls))
    return {
        "gpu": gpu,
        "commit": commit,
        "input_id": sd_input.id,
        "seed": sd_input.seed,
        "wall_p50_ms": f"{p50:.3f}",
        "wall_p99_ms": f"{p99:.3f}",
        "n_runs": len(walls),
        "vram_peak_mb": f"{vram:.3f}",
    }


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_sd_turbo_{commit}.csv"


def _trace_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"sd-turbo-{today}.trace.json.gz"


def _env_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"sd-turbo-{today}.env.txt"


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_pipeline(device: str | None) -> Any:
    """Load SdTurboPipeline via the project's factory (not raw diffusers).

    Going through ``_sd_turbo_stage`` keeps the profile on OUR code path,
    not the upstream diffusers pipe -- the parity comparison (Phase 3)
    is the one that benchmarks diffusers directly.
    """
    from nanovllm_omni.models.sd_turbo.stage import _sd_turbo_stage

    args = SimpleNamespace(
        model="stabilityai/sd-turbo",
        device=device,
        dtype="float16",
        # allow_hf_download=False so the factory sets variant="fp16" and
        # diffusers resolves to the *.fp16.safetensors files in the offline
        # HF cache (stabilityai/sd-turbo ships only the fp16 variant).
        extra={"allow_hf_download": False},
    )
    return _sd_turbo_stage(None, args)


def _infer_one(pipeline: Any, prompt: str) -> Any:
    """One forward through the 4-method diffusion contract (sd-turbo = 1 step).

    NVTX ranges (per ADR 0001):
    - :tokenize    -- CLIP text encoder + initial latents
    - :unet        -- UNet noise prediction loop
    - :vae-decode  -- latent -> image
    """
    with nvtx.range(":tokenize"):
        state = pipeline.prepare_encode({"prompt": prompt})
    num_steps = state.metadata["num_steps"]
    with nvtx.range(":unet"):
        for step in range(num_steps):
            noise_pred = pipeline.denoise_step(state, step=step, num_steps=num_steps)
            pipeline.step_scheduler(state, noise_pred)
    with nvtx.range(":vae-decode"):
        return pipeline.post_decode(state)


def _run_baseline(
    pipeline: Any,
    sd_input: SdTurboInput,
    *,
    runs: int,
    warmup: int,
    profile_out: Path | None,
) -> tuple[list[float], float]:
    """Warmup + N timed forwards; return (walls_ms, peak_vram_mb).

    All N timed forwards share one Kineto context so ``profile_out``
    captures the whole bench, not just the last run.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        _infer_one(pipeline, sd_input.prompt)

    walls: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    if profile_out is not None:
        prof_ctx: Any = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    else:
        prof_ctx = contextlib.nullcontext()

    with prof_ctx as prof:
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _infer_one(pipeline, sd_input.prompt)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000.0)

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    if profile_out is not None and prof is not None:
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(profile_out))
    return walls, peak_vram


def _run_cuda_graph(
    pipeline: Any,
    sd_input: SdTurboInput,
    *,
    runs: int,
    warmup: int,
) -> tuple[list[float], float]:
    """CUDA Graph capture variant: capture once, replay N times.

    The 1-step sd-turbo forward is a fixed-shape computation, so CG
    capture is straightforward. The captured graph re-runs the same
    UNet + VAE decode sequence on the original input data (we don't
    vary the prompt between replays -- that's the cost of a fixed
    shape). VRAM peak includes the capture-time working set.

    Returns (walls_ms, peak_vram_mb).
    """
    import torch

    # Build the state once; capture uses these specific tensor addresses.
    state = pipeline.prepare_encode({"prompt": sd_input.prompt})
    num_steps = state.metadata["num_steps"]

    def _gpu_only() -> Any:
        # Reset before each call. We share one state + one scheduler
        # across warmup + capture + replays, but:
        # - step_scheduler mutates state.step_index (Python int, leaks)
        # - scheduler.step() mutates an internal _step_index (also leaks,
        #   but only via set_timesteps reset)
        # - diffusers' UNet copies the CPU timestep tensor to GPU each
        #   forward; CG capture forbids non-pinned CPU->CUDA copies, so
        #   we pin the timesteps tensor immediately after set_timesteps.
        # Without these resets / pins, warmup 2 reads timesteps[1] /
        # sigmas[2] which are OOB for 1-step sd-turbo, or fails with
        # "Cannot copy between CPU and CUDA tensors during CUDA graph
        # capture".
        state.step_index = 0
        pipeline.scheduler.set_timesteps(num_steps)
        pipeline.scheduler.timesteps = pipeline.scheduler.timesteps.pin_memory()
        for step in range(num_steps):
            noise_pred = pipeline.denoise_step(state, step=step, num_steps=num_steps)
            pipeline.step_scheduler(state, noise_pred)
        return pipeline.post_decode(state)

    # Warmup the kernels on a side stream (compile CUDA cache + cuDNN).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup + 2):
            _gpu_only()
    torch.cuda.current_stream().wait_stream(side)

    # Capture.
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=side):
            _gpu_only()
    except RuntimeError as exc:
        # diffusers + CG capture is brittle: pin_memory fixes the
        # timesteps CPU->GPU copy but other tensors (tokenizer output,
        # random latents) still cross the boundary during capture. On
        # diffusers 0.40 + torch 2.11, capture fails with a CUDA
        # "operation failed due to a previous error during capture".
        # Fall back to a one-shot replay of _gpu_only for each timed
        # run so the bench still produces numbers; the optimization
        # story is documented in the perf writeup as "tried CG, hit
        # diffusers CPU->GPU copy friction, defer to Phase 2 cell
        # design with a non-diffusers pipeline".
        print(f"[bench] CUDA Graph capture failed ({exc!r}); falling back to one-shot")
        walls: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _gpu_only()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000.0)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        return walls, peak_vram

    # Replay.
    walls: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000.0)

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    return walls, peak_vram


def _env_snapshot(out_dir: Path, today: str, gpu: str, commit: str) -> Path:
    """Per-run env snapshot for the perf writeup.

    Fields: torch version, commit, GPU name + driver, CUDA version,
    today's date. Matches minimind's ``.env.txt`` convention.
    """
    import torch

    lines = [
        f"date: {today}",
        f"commit: {commit}",
        f"gpu: {gpu}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda: {torch.version.cuda}")
        lines.append(f"device: {torch.cuda.get_device_name(0)}")
    path = _env_path(out_dir, today)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip Kineto trace export (faster; no trace.json.gz written).",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the UNet + VAE decode loop in a CUDA Graph; replay N times.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_sd_turbo_<commit>.csv)",
    )
    args = parser.parse_args(argv)

    gpu = gpu_label()
    commit = git_commit()
    today = date.today().isoformat()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)
    trace_path = None if args.no_trace else _trace_path(out_dir, today)

    if gpu == "cpu":
        # No CUDA -> still produce a CSV row (all zeros) so the bench
        # doesn't fail CI on CPU hosts.
        row = _row(SD_TURBO_INPUTS[0], [0.0] * args.runs, "cpu", commit, 0.0)
        _write_csv([row], out_path)
        print(f"wrote {out_path} (cpu host; no timings collected)")
        return 0

    pipeline = _load_pipeline(args.device)
    rows: list[dict[str, Any]] = []
    for sd_input in SD_TURBO_INPUTS:
        if args.cuda_graph:
            walls, peak_vram = _run_cuda_graph(
                pipeline,
                sd_input,
                runs=args.runs,
                warmup=args.warmup,
            )
        else:
            walls, peak_vram = _run_baseline(
                pipeline,
                sd_input,
                runs=args.runs,
                warmup=args.warmup,
                profile_out=trace_path,
            )
        rows.append(_row(sd_input, walls, gpu, commit, peak_vram))
        p50, p99, _peak = _summarize(walls)
        print(
            f"{sd_input.id}: p50={p50:.3f} ms, p99={p99:.3f} ms, " f"peak_vram={peak_vram:.0f} MiB"
        )

    _write_csv(rows, out_path)
    _env_snapshot(out_dir, today, gpu, commit)
    trace_msg = str(trace_path) if trace_path is not None else "(no trace)"
    print(f"wrote {out_path} + {trace_msg} ({platform.node()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
