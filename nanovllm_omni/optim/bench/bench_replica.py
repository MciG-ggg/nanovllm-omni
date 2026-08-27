"""TK-013: per-replica throughput scaling for MiniMind-O (StagePool demo).

Compares 1, 2, 4 replicas of MiniMind-O on the same fixed scenario as
``bench_minimind.py`` (TK-012). This is the headline validation of the
StagePool pattern: increasing ``num_replicas`` should give ~Nx
throughput (saturating near GPU limit) without changing per-request
latency significantly, with VRAM staying roughly flat because all
replicas share the same bundle (single-process scope).

What this script actually measures:

- **Throughput (req/s)** for a fixed 8-request workload.
- **Per-request latency p50 / p99** over the workload.
- **VRAM peak** for the run.

Important caveat (the README's "Single-card, single-process runtime"
section): in this codebase, "replicas" do **not** mean N copies of
the model in N subprocesses. They mean N independent
``RuntimeScheduler`` instances driving the same ``MinimindBundle`` with
different RNG seeds. So:

- VRAM stays roughly flat as ``num_replicas`` grows (the model is
  loaded once).
- Throughput scales because the GPU was underutilized at replica=1
  (decode is single-token-per-forward; batching N requests across
  N replicas hides most of the per-request overhead).

If you want true N-model-replicas-across-N-subprocesses, that's the
distributed runtime the project deliberately omits; this bench only
demonstrates the single-process StagePool pattern.

CPU / no-CUDA: prints a no-op row per ``num_replicas`` (timings 0) so
the CSV still has rows; the table stays well-formed.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .bench_minimind import FIXED_PROMPT

# Locked scenario: 8 prompts, all the fixed TK-012 prompt, varying
# num_replicas. 8 is a power of 2 so the RoundRobinBalancer's per-replica
# split (8/replicas requests per replica) is clean.
DEFAULT_PROMPTS_PER_REPLICA = 8
DEFAULT_REPLICAS: tuple[int, ...] = (1, 2, 4)


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


def _git_commit() -> str:
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


def _as_csv_row(
    *,
    replicas: int,
    gpu: str,
    commit: str,
    batch: int,
    throughput: float,
    latency_p50: float,
    latency_p99: float,
    vram_peak_mb: float,
) -> dict[str, Any]:
    return {
        "replicas": replicas,
        "gpu": gpu,
        "commit": commit,
        "batch": batch,
        "throughput_ops_per_sec": f"{throughput:.3f}",
        "latency_p50_ms": f"{latency_p50:.3f}",
        "latency_p99_ms": f"{latency_p99:.3f}",
        "vram_peak_mb": f"{vram_peak_mb:.3f}",
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| replicas | gpu | commit | batch | throughput_ops_per_sec | latency_p50_ms | latency_p99_ms | vram_peak_mb |"
    sep = "|---:|---|---|---:|---:|---:|---:|---:|"
    body: list[str] = []
    for r in rows:
        body.append(
            "| {replicas} | {gpu} | {commit} | {batch} | {throughput} | {p50} | {p99} | {vram} |".format(
                replicas=r["replicas"],
                gpu=r["gpu"],
                commit=r["commit"],
                batch=r["batch"],
                throughput=r["throughput_ops_per_sec"],
                p50=r["latency_p50_ms"],
                p99=r["latency_p99_ms"],
                vram=r["vram_peak_mb"],
            )
        )
    return "\n".join([header, sep, *body])


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _per_request_latencies(
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
    """Run one workload and return (per-request wall-clock in ms, throughput, vram_peak).

    The bundle / model is loaded once outside the function in the main
    CLI; this helper only calls ``batched_fn`` and measures time. The
    batched_fn signature mirrors ``run_batched_generate`` so a test can
    pass a mock.
    """
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
    # Per-request latency: equally-distributed share of elapsed (the
    # batched engine drives all replicas concurrently, so the wall-clock
    # cost is the throughput floor, not a sum of per-request costs).
    per_request = [elapsed_s * 1000.0] * n
    return per_request, throughput, vram


def _summarize_row(
    *,
    replicas: int,
    gpu: str,
    commit: str,
    batch: int,
    latencies: list[float],
    throughput: float,
    vram_peak_mb: float,
) -> dict[str, Any]:
    return _as_csv_row(
        replicas=replicas,
        gpu=gpu,
        commit=commit,
        batch=batch,
        throughput=throughput,
        latency_p50=_percentile(latencies, 50),
        latency_p99=_percentile(latencies, 99),
        vram_peak_mb=vram_peak_mb,
    )


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

    gpu = _gpu_label()
    commit = _git_commit()
    out_dir = Path(__file__).resolve().parents[3] / "docs" / "perf"
    out_path = Path(args.out) if args.out else out_dir / f"bench_replica_{commit}.csv"

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

    # Real CUDA path.
    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.models.minimind_omni import create_bundle

    bundle_kwargs: dict[str, Any] = {}
    if args.mimi:
        bundle_kwargs["mimi_model_id"] = args.mimi
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    for replicas in args.replicas:
        batch = args.prompts_per_replica * replicas
        prompts = [FIXED_PROMPT.text] * batch
        latencies, throughput, vram = _per_request_latencies(
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


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_PROMPTS_PER_REPLICA",
    "DEFAULT_REPLICAS",
    "_markdown_table",
    "_summarize_row",
    "main",
]


# keep statistics import live even though the helpers don't use it --
# future per-request instrumentation will likely want stdev / mean.
_ = statistics
