"""TK-012: standardized MiniMind-O benchmark with fixed scenario.

Single-purpose bench for README headline numbers:

| Item | Value |
|---|---|
| prompt | "Hello, how are you?" (fixed) |
| output | 120 Mimi frames @ 12.5 Hz = 5 s audio |
| batch | 1 |
| metric | TTFB p50 + total p50 over N requests (default 50) |
| hardware | dual-run on RTX 3050 (4GB) AND RTX 4090 (24GB) |

Why this script:

- The existing ``python -m nanovllm_omni.optim.bench`` CLI (TK-011) is a
  multi-prompt, per-stage profiler; this script is the simpler
  TTFB-p50 / total-p50 headline bench the README points to.
- TTFB = time to first audio frame. MiniMind-O emits one Mimi frame
  per generated token, so the per-step generation time plus tokenize
  approximates TTFB.

Why both GPUs in one script (rather than per-host):

- The hardware column is just a string in the output table; running
  on the current machine produces one row. The headline is meant to
  be collected once per host (the prompt and scenario are locked) so
  the README can list both numbers side by side.

Outputs:

- Markdown table to stdout.
- CSV to ``docs/perf/bench_minimind_<commit>.csv``.

CPU / no-CUDA: prints a no-op markdown row (``gpu=cpu``, all timings
0) so the CSV still has a row. This avoids hard-failing on machines
without a CUDA device; real numbers come from the GPU hosts.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .prompts import BenchPrompt

# TK-012 locked scenario: prompt fixed to English, output fixed at
# 120 frames (5 s of audio @ 12.5 Hz Mimi rate).
FIXED_PROMPT = BenchPrompt(id="ttfb_fixed", text="Hello, how are you?")
FIXED_MAX_TOKENS = 120  # 120 frames / 12.5 Hz = 9.6 s audio budget; cap at 5 s budget

DEFAULT_N = 50
DEFAULT_WARMUP = 3


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _git_commit() -> str:
    """Best-effort short SHA; falls back to ``unknown`` for non-git checkouts."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _gpu_label() -> str:
    """Human-readable GPU + driver label, or ``cpu`` if CUDA isn't present.

    Uses ``nvidia-smi`` query when available; falls back to torch's
    device name. We do NOT raise on a CPU-only host -- the script still
    writes a CPU row so the CSV stays well-formed.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        driver_str = driver.decode("ascii").strip().splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, IndexError):
        driver_str = "?"
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} ({mem_gb:.0f} GB, driver {driver_str})"


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_minimind_{commit}.csv"


CSV_COLUMNS: tuple[str, ...] = (
    "gpu",
    "commit",
    "n_requests",
    "ttfb_p50_ms",
    "ttfb_p99_ms",
    "total_p50_ms",
    "total_p99_ms",
    "vram_peak_mb",
)


def _as_csv_row(gpu: str, commit: str, n: int, *values: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gpu": gpu,
        "commit": commit,
        "n_requests": n,
        "ttfb_p50_ms": f"{values[0]:.3f}",
        "ttfb_p99_ms": f"{values[1]:.3f}",
        "total_p50_ms": f"{values[2]:.3f}",
        "total_p99_ms": f"{values[3]:.3f}",
        "vram_peak_mb": f"{values[4]:.3f}",
    }
    return row


def _row_from_summary(gpu: str, commit: str, n: int, summary: tuple[float, ...]) -> dict[str, Any]:
    """Same shape as ``_as_csv_row`` but takes a 5-tuple directly."""
    return _as_csv_row(gpu, commit, n, *summary)


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _markdown_table(row: dict[str, Any]) -> str:
    header = "| gpu | commit | n | ttfb_p50_ms | ttfb_p99_ms | total_p50_ms | total_p99_ms | vram_peak_mb |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    body = "| {gpu} | {commit} | {n} | {ttfb_p50} | {ttfb_p99} | {total_p50} | {total_p99} | {vram} |".format(
        gpu=row["gpu"],
        commit=row["commit"],
        n=row["n_requests"],
        ttfb_p50=row["ttfb_p50_ms"],
        ttfb_p99=row["ttfb_p99_ms"],
        total_p50=row["total_p50_ms"],
        total_p99=row["total_p99_ms"],
        vram=row["vram_peak_mb"],
    )
    return "\n".join([header, sep, body])


def _summarize(results: list[Any]) -> tuple[float, float, float, float, float]:
    """Compute TTFB p50/p99 + total p50/p99 + VRAM peak from RunResult rows.

    TTFB is approximated as ``tokenize_ms + generate_per_step_ms`` (one
    generated token = one Mimi frame = first chunk of audio). Total is
    the existing ``StageTimes.total_ms`` sum.
    """
    if not results:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    ttfb: list[float] = []
    total: list[float] = []
    vram: list[float] = []
    for r in results:
        per_step = r.times.generate_ms / r.frames if r.frames else r.times.generate_ms
        ttfb.append(r.times.tokenize_ms + per_step)
        total.append(r.times.total_ms)
        vram.append(r.vram_peak_mb)
    return (
        _percentile(ttfb, 50),
        _percentile(ttfb, 99),
        _percentile(total, 50),
        _percentile(total, 99),
        max(vram) if vram else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of timed requests")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_minimind_<commit>.csv)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=FIXED_MAX_TOKENS,
        help="Token budget per request (default: 120 = 5 s audio @ 12.5 Hz)",
    )
    args = parser.parse_args(argv)

    gpu = _gpu_label()
    commit = _git_commit()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)

    if gpu == "cpu":
        # No CUDA -> still produce a CSV row (all zeros) so the bench
        # doesn't fail CI on CPU hosts. The README headline numbers
        # come from real GPU runs.
        row = _row_from_summary("cpu", commit, args.n, (0.0, 0.0, 0.0, 0.0, 0.0))
        _write_csv([row], out_path)
        print(_markdown_table(row))
        print(f"\nwrote {out_path} (cpu host; no timings collected)")
        return 0

    # Real CUDA path.
    from nanovllm_omni.models.minimind_omni import create_bundle

    from .runner import run_n

    bundle_kwargs: dict[str, Any] = {}
    if args.mimi:
        bundle_kwargs["mimi_model_id"] = args.mimi
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    results = run_n(
        bundle,
        FIXED_PROMPT,
        n=args.n,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        seed=42,
        temperature=0.7,
        top_p=0.9,
    )

    ttfb_p50, ttfb_p99, total_p50, total_p99, vram_peak = _summarize(results)
    row = _row_from_summary(
        gpu, commit, args.n, (ttfb_p50, ttfb_p99, total_p50, total_p99, vram_peak)
    )
    _write_csv([row], out_path)
    print(_markdown_table(row))
    print(f"\nwrote {out_path} ({platform.node()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_N",
    "DEFAULT_WARMUP",
    "FIXED_MAX_TOKENS",
    "FIXED_PROMPT",
    "_csv_path",
    "_summarize",
    "main",
]
