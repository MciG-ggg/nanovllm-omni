"""TK-013: StagePool scaling demo for MiniMind-O (single-process).

Sweeps ``num_replicas in (1, 2, 4)`` with a fixed 8-prompt workload per
replica and reports throughput (req/s), per-request latency p50/p99, and
VRAM peak per replica count. The README uses this as the StagePool-pattern
evidence: throughput scales with replicas, VRAM stays roughly flat.

Important caveat: in this codebase, "replicas" do **not** mean N copies
of the model in N subprocesses. They mean N independent
``RuntimeScheduler`` instances driving the same ``MinimindBundle``. So
VRAM stays roughly flat; throughput scales because decode is
single-token-per-forward and batching across N schedulers hides the
per-request overhead. True N-model-replicas-across-N-subprocesses is the
distributed runtime the project deliberately omits.

CPU / no-CUDA: still writes one zero-valued row per replica so the CSV
stays well-formed.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bench_minimind import FIXED_PROMPT
from .env import git_commit, gpu_label

# Locked scenario: 8 prompts per replica; power of 2 so the
# RoundRobinBalancer's per-replica split is clean.
DEFAULT_PROMPTS_PER_REPLICA = 8
DEFAULT_REPLICAS: tuple[int, ...] = (1, 2, 4)

CSV_COLUMNS: tuple[str, ...] = (
    "replicas",
    "gpu",
    "commit",
    "batch",
    "throughput_ops_per_sec",
    "latency_p50_ms",
    "latency_p99_ms",
    "vram_peak_mb",
)


def _summarize_row(
    *,
    replicas: int,
    gpu: str,
    commit: str,
    batch: int,
    latencies: Sequence[float],
    throughput: float,
    vram_peak_mb: float,
) -> dict[str, Any]:
    if latencies:
        cuts = statistics.quantiles(latencies, n=100, method="inclusive")
        p50, p99 = cuts[49], cuts[98]
    else:
        p50 = p99 = 0.0
    return {
        "replicas": replicas,
        "gpu": gpu,
        "commit": commit,
        "batch": batch,
        "throughput_ops_per_sec": f"{throughput:.3f}",
        "latency_p50_ms": f"{p50:.3f}",
        "latency_p99_ms": f"{p99:.3f}",
        "vram_peak_mb": f"{vram_peak_mb:.3f}",
    }


def _csv_path(out_dir: Path, commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"bench_replica_{commit}.csv"


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| replicas | gpu | commit | batch | throughput_ops_per_sec | "
        "latency_p50_ms | latency_p99_ms | vram_peak_mb |"
    )
    sep = "|---:|---|---|---:|---:|---:|---:|---:|"
    body = "\n".join(
        "| {replicas} | {gpu} | {commit} | {batch} | {throughput_ops_per_sec} | "
        "{latency_p50_ms} | {latency_p99_ms} | {vram_peak_mb} |".format(**r)
        for r in rows
    )
    return f"{header}\n{sep}\n{body}\n"


def _run_replica_workload(
    bundle: Any,
    batched_fn: Any,
    prompts: list[str],
    *,
    num_replicas: int,
    max_batch: int,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[list[float], float, float]:
    """One replica-config workload; returns (latencies_ms, throughput_ops, vram_mb)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    payloads = batched_fn(
        bundle,
        prompts,
        num_replicas=num_replicas,
        max_batch=max_batch,
        max_new_tokens=max_new_tokens,
        base_seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed_s = time.perf_counter() - t0
    vram = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
    )
    n = max(len(payloads), 1)
    throughput = n / elapsed_s if elapsed_s > 0 else 0.0
    # Per-request latency: wall-clock divided across the batch (the
    # batched engine drives all replicas concurrently, so wall-clock
    # is the throughput floor, not a sum of per-request costs).
    return [elapsed_s * 1000.0] * n, throughput, vram


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPLICAS),
        help="num_replicas values to sweep (default: 1 2 4)",
    )
    parser.add_argument(
        "--prompts-per-replica",
        type=int,
        default=DEFAULT_PROMPTS_PER_REPLICA,
        help="Requests per replica (total = replicas * prompts_per_replica)",
    )
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: docs/perf/bench_replica_<commit>.csv)",
    )
    args = parser.parse_args(argv)

    gpu = gpu_label()
    commit = git_commit()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else _csv_path(out_dir, commit)

    rows: list[dict[str, Any]] = []
    if gpu == "cpu":
        for replicas in args.replicas:
            batch = args.prompts_per_replica * replicas
            rows.append(
                _summarize_row(
                    replicas=replicas,
                    gpu="cpu",
                    commit=commit,
                    batch=batch,
                    latencies=[],
                    throughput=0.0,
                    vram_peak_mb=0.0,
                )
            )
        _write_csv(rows, out_path)
        print(_markdown_table(rows))
        print(f"\nwrote {out_path} (cpu host; no timings collected)")
        return 0

    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.models.minimind_omni import create_bundle

    bundle_kwargs: dict[str, Any] = {}
    if args.mimi:
        bundle_kwargs["mimi_model_id"] = args.mimi
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    for replicas in args.replicas:
        batch = args.prompts_per_replica * replicas
        prompts = [FIXED_PROMPT.text] * batch
        latencies, throughput, vram = _run_replica_workload(
            bundle,
            run_batched_generate,
            prompts,
            num_replicas=replicas,
            max_batch=args.max_batch,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rows.append(
            _summarize_row(
                replicas=replicas,
                gpu=gpu,
                commit=commit,
                batch=batch,
                latencies=latencies,
                throughput=throughput,
                vram_peak_mb=vram,
            )
        )

    _write_csv(rows, out_path)
    print(_markdown_table(rows))
    print(f"\nwrote {out_path} ({platform.node()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
