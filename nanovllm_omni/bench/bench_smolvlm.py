"""Profile bench for smolvlm (Phase 1 baseline).

Runs the project's SmolVLM vlm_forward callable (returned by
``_vlm_stage`` factory) under Kineto capture, writes one CSV row per
input with p50 / p99 / peak VRAM, plus a per-run Kineto trace to
``docs/perf/smolvlm-<date>.trace.json.gz`` and an env snapshot at
``docs/perf/smolvlm-<date>.env.txt``.

Text-only mode (no images) for parity with the minimind thinker bench.
Three prompt lengths (short / medium / long) isolate prefill vs decode
cost split.

See ADR 0001 for the 4-phase protocol. Phase 1 only: no CUDA Graph, no
fusion patches, no per-stage NVTX markers yet -- Phase 2 cell design
follows the bottleneck finding from this profile.
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
from .smolvlm_prompts import SMOLVLM_INPUTS, SmolVLMInput

CSV_COLUMNS: tuple[str, ...] = (
    "gpu",
    "commit",
    "input_id",
    "max_new_tokens",
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
    sm_input: SmolVLMInput,
    walls: Sequence[float],
    gpu: str,
    commit: str,
    vram: float,
) -> dict[str, Any]:
    p50, p99, _peak = _summarize(list(walls))
    return {
        "gpu": gpu,
        "commit": commit,
        "input_id": sm_input.id,
        "max_new_tokens": sm_input.max_new_tokens,
        "wall_p50_ms": f"{p50:.3f}",
        "wall_p99_ms": f"{p99:.3f}",
        "n_runs": len(walls),
        "vram_peak_mb": f"{vram:.3f}",
    }


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_smolvlm_{commit}.csv"


def _trace_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"smolvlm-{today}.trace.json.gz"


def _env_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"smolvlm-{today}.env.txt"


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_pipeline(device: str | None) -> Any:
    """Load SmolVLM vlm_forward via the project's factory (not raw transformers).

    Going through ``_vlm_stage`` keeps the profile on OUR code path,
    not the upstream transformers generate -- Phase 3 parity is the one
    that benchmarks transformers directly.
    """
    from nanovllm_omni.models.smolvlm.stage import _vlm_stage

    args = SimpleNamespace(
        model="HuggingFaceTB/SmolVLM-500M-Instruct",
        device=device,
        dtype="bfloat16",
        extra={"allow_hf_download": False},
    )
    return _vlm_stage(None, args)


def _infer_one(pipeline: Any, prompt: str, max_new_tokens: int) -> str:
    """One text-only forward through the SmolVLM stage contract.

    NVTX ranges (per ADR 0001):
    - :vlm-decode  -- SmolVLM processor + AR forward + per-token decode

    The ADR schema lists 3 separate ranges (tokenize / vlm-prefill /
    vlm-decode) but we wrap the whole upstream ``model.generate`` as
    :vlm-decode here because splitting them requires intrusive changes
    to transformers' generate. For text-only short prompts (smolvlm_short),
    decode is the dominant cost; for medium/long prompts the implicit
    prefill + decode split is roughly proportional to AR length.

    Returns the decoded text (informational; not part of the timing).
    """
    sampling = SimpleNamespace(extra={"max_new_tokens": max_new_tokens, "images": []})
    with nvtx.range(":vlm-decode"):
        return pipeline({"prompt": prompt}, sampling)


def _run_baseline(
    pipeline: Any,
    sm_input: SmolVLMInput,
    *,
    runs: int,
    warmup: int,
    profile_out: Path | None,
) -> tuple[list[float], float]:
    """Warmup + N timed forwards; return (walls_ms, peak_vram_mb)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        _infer_one(pipeline, sm_input.prompt, sm_input.max_new_tokens)

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
            _infer_one(pipeline, sm_input.prompt, sm_input.max_new_tokens)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000.0)

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    if profile_out is not None and prof is not None:
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(profile_out))
    return walls, peak_vram


def _env_snapshot(out_dir: Path, today: str, gpu: str, commit: str) -> Path:
    """Per-run env snapshot for the perf writeup."""
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
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device (default: cuda; sd-turbo / minimind default to cuda too)",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip Kineto trace export (faster; no trace.json.gz written).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_smolvlm_<commit>.csv)",
    )
    args = parser.parse_args(argv)

    gpu = gpu_label()
    commit = git_commit()
    today = date.today().isoformat()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)
    trace_path = None if args.no_trace else _trace_path(out_dir, today)

    if gpu == "cpu":
        # No CUDA -> still produce a CSV row per input (all zeros) so the
        # bench doesn't fail CI on CPU hosts.
        rows = [
            _row(sm_input, [0.0] * args.runs, "cpu", commit, 0.0) for sm_input in SMOLVLM_INPUTS
        ]
        _write_csv(rows, out_path)
        print(f"wrote {out_path} (cpu host; no timings collected)")
        return 0

    pipeline = _load_pipeline(args.device)
    rows: list[dict[str, Any]] = []
    for sm_input in SMOLVLM_INPUTS:
        walls, peak_vram = _run_baseline(
            pipeline,
            sm_input,
            runs=args.runs,
            warmup=args.warmup,
            profile_out=trace_path,
        )
        rows.append(_row(sm_input, walls, gpu, commit, peak_vram))
        p50, p99, _peak = _summarize(walls)
        print(
            f"{sm_input.id}: p50={p50:.3f} ms, p99={p99:.3f} ms, " f"peak_vram={peak_vram:.0f} MiB"
        )

    _write_csv(rows, out_path)
    _env_snapshot(out_dir, today, gpu, commit)
    trace_msg = str(trace_path) if trace_path is not None else "(no trace)"
    print(f"wrote {out_path} + {trace_msg} ({platform.node()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
