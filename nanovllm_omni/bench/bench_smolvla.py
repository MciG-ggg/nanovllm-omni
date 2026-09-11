"""Profile bench for smolvla (Phase 1 baseline).

Runs the two-stage pipeline (vlm AR + action flow matching) under
Kineto capture, writes one CSV row per input with p50 / p99 / peak VRAM,
plus a per-run Kineto trace to ``docs/perf/smolvla-<date>.trace.json.gz``
and an env snapshot at ``docs/perf/smolvla-<date>.env.txt``.

NVTX ranges (per ADR 0001):
- :tokenize      -- preprocessor (image + state normalization)
- :vlm-prefill   -- VLM forward (SigLIP + SmolVLM2 prefix embed + KV cache)
- :flow-step     -- flow-matching loop (denoise_step + step_scheduler x N)
- :action-decode -- truncation to original_action_dim

Synthetic observation (256x256 RGB + 7-dim zero state); real LIBERO
data is Phase 2 follow-up.

See ADR 0001 for the 4-phase protocol. Phase 1 only: no CUDA Graph, no
fusion patches.
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
from .smolvla_prompts import SMOLVLA_INPUTS, SmolVLAInput

CSV_COLUMNS: tuple[str, ...] = (
    "gpu",
    "commit",
    "input_id",
    "num_inference_steps",
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
    sm_input: SmolVLAInput,
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
        "num_inference_steps": sm_input.num_inference_steps,
        "wall_p50_ms": f"{p50:.3f}",
        "wall_p99_ms": f"{p99:.3f}",
        "n_runs": len(walls),
        "vram_peak_mb": f"{vram:.3f}",
    }


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_smolvla_{commit}.csv"


def _trace_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"smolvla-{today}.trace.json.gz"


def _env_path(out_dir: Path, today: str) -> Path:
    return out_dir / f"smolvla-{today}.env.txt"


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_pipeline(device: str | None) -> tuple[Any, Any]:
    """Load both SmolVLA stages via the project's factories.

    vlm_stage and action_stage share a process-level policy cache
    (see ``action_stage._POLICY_CACHE``), so calling both loads the
    lerobot SmolVLAPolicy exactly once.
    """
    from nanovllm_omni.models.smolvla.action_stage import _action_stage
    from nanovllm_omni.models.smolvla.vlm_stage import _vlm_stage

    args = SimpleNamespace(
        model="HuggingFaceVLA/smolvla_libero",
        device=device,
        extra={"allow_hf_download": False},
    )
    vlm = _vlm_stage(None, args)
    action = _action_stage(None, args)
    return vlm, action


def _infer_one(
    vlm: Any,
    action: Any,
    sm_input: SmolVLAInput,
) -> Any:
    """Full smolvla forward: vlm AR prefill + action flow matching."""
    from nanovllm_omni.models.smolvla.stage_processors import vlm2action

    sampling = SimpleNamespace(
        extra={
            "image": sm_input.image,
            "state": list(sm_input.state),
            "num_inference_steps": sm_input.num_inference_steps,
        }
    )

    with nvtx.range(":tokenize"):
        vlm_out = vlm(sm_input.instruction, sampling)

    with nvtx.range(":vlm-prefill"):
        action_payload = vlm2action(vlm_out, sm_input.instruction)

    state_obj = action.prepare_encode(action_payload)
    num_steps = state_obj.metadata["num_steps"]
    with nvtx.range(":flow-step"):
        for step in range(num_steps):
            noise_pred = action.denoise_step(state_obj, step=step, num_steps=num_steps)
            action.step_scheduler(state_obj, noise_pred)

    with nvtx.range(":action-decode"):
        return action.post_decode(state_obj)


def _run_baseline(
    vlm: Any,
    action: Any,
    sm_input: SmolVLAInput,
    *,
    runs: int,
    warmup: int,
    profile_out: Path | None,
) -> tuple[list[float], float]:
    """Warmup + N timed forwards; return (walls_ms, peak_vram_mb)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        _infer_one(vlm, action, sm_input)

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
            _infer_one(vlm, action, sm_input)
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
    parser.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip Kineto trace export (faster; no trace.json.gz written).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_smolvla_<commit>.csv)",
    )
    args = parser.parse_args(argv)

    gpu = gpu_label()
    commit = git_commit()
    today = date.today().isoformat()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)
    trace_path = None if args.no_trace else _trace_path(out_dir, today)

    if gpu == "cpu":
        rows = [
            _row(sm_input, [0.0] * args.runs, "cpu", commit, 0.0) for sm_input in SMOLVLA_INPUTS
        ]
        _write_csv(rows, out_path)
        print(f"wrote {out_path} (cpu host; no timings collected)")
        return 0

    vlm, action = _load_pipeline(args.device)
    rows: list[dict[str, Any]] = []
    for sm_input in SMOLVLA_INPUTS:
        walls, peak_vram = _run_baseline(
            vlm,
            action,
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
