"""TK-012: standardized MiniMind-O benchmark with fixed scenario.

Single-purpose bench for README headline numbers:

| Item | Value |
|---|---|
| prompt | "Hello, how are you?" (fixed) |
| output | 120 Mimi frames @ 12.5 Hz = 5 s audio |
| metric | TTFB p50 + total p50 over N requests (default 50) |

Outputs a CSV row (gpu, commit, n, ttfb_p50/p99_ms, total_p50/p99_ms,
vram_peak_mb) plus a markdown table to stdout.

CPU / no-CUDA: still writes a zero-valued row so the CSV stays well-formed.
"""

from __future__ import annotations

import argparse
import platform
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .env import git_commit, gpu_label
from .prompts import BenchPrompt
from .runner import run_n

# TK-012 locked scenario: prompt + token budget are part of the headline
# contract. Don't drift these without an explicit ticket.
FIXED_PROMPT = BenchPrompt(id="ttfb_fixed", text="Hello, how are you?")
FIXED_MAX_TOKENS = 120  # 120 frames / 12.5 Hz ≈ 9.6 s audio budget; cap at 5 s

DEFAULT_N = 50
DEFAULT_WARMUP = 3

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


def _summarize(results: Sequence[Any]) -> tuple[float, float, float, float, float]:
    """(ttfb_p50, ttfb_p99, total_p50, total_p99, vram_peak) from RunResult rows.

    TTFB is approximated as ``tokenize_ms + per-frame generate_ms`` (one
    generated token = one Mimi frame = first chunk of audio).
    """
    if not results:
        return (0.0,) * 5
    ttfbs = [r.times.tokenize_ms + r.times.generate_ms / r.frames for r in results]
    totals = [r.times.total_ms for r in results]
    ttfb_cuts = statistics.quantiles(ttfbs, n=100, method="inclusive")
    total_cuts = statistics.quantiles(totals, n=100, method="inclusive")
    return (
        ttfb_cuts[49],
        ttfb_cuts[98],
        total_cuts[49],
        total_cuts[98],
        max((r.vram_peak_mb for r in results), default=0.0),
    )


def _row_from_summary(gpu: str, commit: str, n: int, summary: tuple[float, ...]) -> dict[str, Any]:
    ttfb_p50, ttfb_p99, total_p50, total_p99, vram_peak = summary
    return {
        "gpu": gpu,
        "commit": commit,
        "n_requests": n,
        "ttfb_p50_ms": f"{ttfb_p50:.3f}",
        "ttfb_p99_ms": f"{ttfb_p99:.3f}",
        "total_p50_ms": f"{total_p50:.3f}",
        "total_p99_ms": f"{total_p99:.3f}",
        "vram_peak_mb": f"{vram_peak:.3f}",
    }


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_minimind_{commit}.csv"


def _write_csv(row: dict[str, Any], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerow(row)
    return path


def _markdown_table(row: dict[str, Any]) -> str:
    return (
        "| gpu | commit | n | ttfb_p50_ms | ttfb_p99_ms | "
        "total_p50_ms | total_p99_ms | vram_peak_mb |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
        "| {gpu} | {commit} | {n} | {ttfb_p50} | {ttfb_p99} | "
        "{total_p50} | {total_p99} | {vram} |"
    ).format(
        gpu=row["gpu"],
        commit=row["commit"],
        n=row["n_requests"],
        ttfb_p50=row["ttfb_p50_ms"],
        ttfb_p99=row["ttfb_p99_ms"],
        total_p50=row["total_p50_ms"],
        total_p99=row["total_p99_ms"],
        vram=row["vram_peak_mb"],
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

    gpu = gpu_label()
    commit = git_commit()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)

    if gpu == "cpu":
        # No CUDA -> still produce a CSV row (all zeros) so the bench
        # doesn't fail CI on CPU hosts. The README headline numbers
        # come from real GPU runs.
        row = _row_from_summary("cpu", commit, args.n, (0.0,) * 5)
        _write_csv(row, out_path)
        print(_markdown_table(row))
        print(f"\nwrote {out_path} (cpu host; no timings collected)")
        return 0

    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    bundle_kwargs: dict[str, Any] = {}
    if args.mimi:
        bundle_kwargs["mimi_model_id"] = args.mimi
    bundle = load_minimind_omni_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

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

    row = _row_from_summary(gpu, commit, args.n, _summarize(results))
    _write_csv(row, out_path)
    print(_markdown_table(row))
    print(f"\nwrote {out_path} ({platform.node()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
