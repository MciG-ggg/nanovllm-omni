"""Profile bench skeleton for sd-turbo (Phase 1 baseline).

# Phase 1: baseline-only wall time over SD_TURBO_INPUTS. No Kineto
# capture or per-stage breakdown yet -- those land when StageProfile
# gains model-aware stage names (tokenize / unet / vae-decode).
# See ADR 0001 for the protocol.
"""

from __future__ import annotations

import argparse
import platform
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
        return (walls[0] if walls else 0.0, 0.0, walls[0] if walls else 0.0)
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


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_sd_turbo_<commit>.csv)",
    )
    args = parser.parse_args(argv)

    gpu = gpu_label()
    commit = git_commit()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)

    if gpu == "cpu":
        # No CUDA -> still produce a CSV row (all zeros) so the bench
        # doesn't fail CI on CPU hosts.
        row = _row(SD_TURBO_INPUTS[0], [0.0] * args.runs, "cpu", commit, 0.0)
        _write_csv([row], out_path)
        print(f"wrote {out_path} (cpu host; no timings collected)")
        return 0

    # TODO(wire): load the sd-turbo diffusers pipeline + run N timed
    # forwards per input. Until wired, write empty rows so the CSV
    # contract is preserved.
    rows = [_row(sd_input, [0.0] * args.runs, gpu, commit, 0.0) for sd_input in SD_TURBO_INPUTS]
    _write_csv(rows, out_path)
    print(f"wrote {out_path} ({platform.node()}) [skeleton: not yet wired]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
