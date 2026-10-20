"""Run the three wall-clock benches on the Colab GPU session.

One command produces every number that ADR-0003 needs for MiniMind-O
and SmolVLA:

1. ``nanovllm_omni.bench sweep-graphs`` -- the MiniMind-O eager cell
   and the thinker-graph cell, each in a fresh process
2. ``nanovllm_omni.bench.bench_smolvla`` -- the SmolVLA two-stage path
3. ``nanovllm_omni.bench.bench_smolvla_flow`` -- the flow loop alone,
   the target that the ncu launch window needs

Usage:

    colab exec -s omni-t4 --timeout 3600 -f tools/measure_benches.py
"""

import subprocess
import sys
from pathlib import Path

DEST = Path("/content/nanovllm-omni")
MEASURE = Path("/content/measure")

COMMANDS = [
    [
        sys.executable,
        "-m",
        "nanovllm_omni.bench",
        "sweep-graphs",
        "--pipeline",
        "full",
        "--model",
        "jingyaogong/minimind-3o",
        "--device",
        "cuda",
        "--runs",
        "5",
        "--warmup",
        "1",
        "--out-dir",
        str(MEASURE / "minimind"),
    ],
    [
        sys.executable,
        "-m",
        "nanovllm_omni.bench.bench_smolvla",
        "--runs",
        "5",
        "--warmup",
        "1",
        "--no-trace",
        "--out",
        str(MEASURE / "bench_smolvla.csv"),
    ],
    [
        sys.executable,
        "-m",
        "nanovllm_omni.bench.bench_smolvla_flow",
        "--repeats",
        "20",
        "--steps",
        "10",
    ],
]


def main() -> int:
    MEASURE.mkdir(parents=True, exist_ok=True)
    failed = []
    for command in COMMANDS:
        print("\n$", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=str(DEST), text=True, capture_output=True)
        print(result.stdout[-4000:], flush=True)
        print(result.stderr[-3000:], flush=True)
        if result.returncode:
            failed.append(" ".join(command[3:6]))

    print("\n=== csv files ===", flush=True)
    for path in sorted(MEASURE.rglob("*.csv")):
        print(f"--- {path} ---", flush=True)
        print(path.read_text(encoding="utf-8"), flush=True)

    if failed:
        print("failed:", ", ".join(failed), flush=True)
        return 1
    print("MEASURE_BENCHES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
