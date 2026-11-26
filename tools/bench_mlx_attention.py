"""Compare the MLX attention arms for one MiniMind-O thinker decode step.

The measurement lives in ``nanovllm_omni.bench.mlx_attention``; this script
is the CLI around it: it runs the padding experiment and the three arms, then
writes the CSV and the JSON metadata next to the results.

Usage:
    uv run python tools/bench_mlx_attention.py --out docs/perf/mlx-attention-2026-11-11
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nanovllm_omni.bench.env import git_commit
from nanovllm_omni.bench.mlx_attention import (
    BUCKET_RATIO,
    LAYERS,
    MEAN_LENGTH,
    PATTERNS,
    POOL_CEILING_BLOCKS,
    REAL_TOKENS,
    Arm,
    check_agreement,
    decode_shape,
    eval_floor,
    lengths_for,
    measure_bucketed,
    measure_dense,
    measure_paged,
    padding_experiment,
    padding_waste,
)


def _best(measure, repeat: int, *args, **kwargs) -> Arm:
    """The best of ``repeat`` measurements of one arm.

    The Mac is shared, so a cell measured while another process runs reads
    high. Repeating the cell and keeping the best sample filters that out.
    """
    results = [measure(*args, **kwargs) for _ in range(repeat)]
    return Arm(
        name=results[0].name,
        attention_ms=min(result.attention_ms for result in results),
        write_ms=min(result.write_ms for result in results),
        buckets=results[0].buckets,
    )


def _arms_table(rows: list[dict]) -> str:
    """The three-arm table, one row per batch and pattern."""
    lines = [
        "| batch | pattern | waste | dense | bucketed | paged (pool=batch) "
        "| paged (pool=64 blocks) | dense ns/head-key | paged ns/head-key | winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        winner = min(
            ("dense", row["dense_total_ms"]),
            ("bucketed", row["bucketed_total_ms"]),
            ("paged_batch_pool", row["paged_total_small_ms"]),
            ("paged_ceiling_pool", row["paged_total_ceiling_ms"]),
            key=lambda item: item[1],
        )[0]
        lines.append(
            f"| {row['batch']} | {row['pattern']} | {row['waste_pct']}% | "
            f"{row['dense_total_ms']} ms | {row['bucketed_total_ms']} ms | "
            f"{row['paged_total_small_ms']} ms | {row['paged_total_ceiling_ms']} ms | "
            f"{row['dense_ns_per_head_key']} | {row['paged_ns_per_head_key']} | {winner} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX attention arms comparison.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    shape = decode_shape()
    floor_ms = eval_floor(args.runs, args.warmup)
    print(f"evaluation floor: {floor_ms:.3f} ms per measured call\n")

    # A fixed cell, measured at the start and at the end of the session: the
    # gap between the two is the drift a shared Mac produces while the run
    # heats up.
    reference = lengths_for("uniform", 8)
    start_reference = _best(
        measure_dense, args.repeat, reference, shape, args.runs, args.warmup
    ).attention_ms

    print("padding experiment: 8 sequences, 256 real tokens each, growing pad")
    print("| padded tokens | waste | dense attn | step (8 layers) |")
    print("|---:|---:|---:|---:|")
    padding_runs = [
        padding_experiment(shape, 8, args.runs, args.warmup) for _ in range(args.repeat)
    ]
    padding_rows = []
    for index, row in enumerate(padding_runs[0]):
        best = min(run[index]["dense_attn_ms"] for run in padding_runs)
        padding_rows.append({**row, "dense_attn_ms": best})
    for row in padding_rows:
        print(
            f"| {row['padded_tokens']} | {row['waste_pct']}% | "
            f"{row['dense_attn_ms']} ms | {row['step_dense_ms']} ms |"
        )
    print()

    rows = []
    for batch in args.batches:
        for pattern in PATTERNS:
            lengths = lengths_for(pattern, batch)
            dense = _best(measure_dense, args.repeat, lengths, shape, args.runs, args.warmup)
            bucketed = _best(measure_bucketed, args.repeat, lengths, shape, args.runs, args.warmup)
            paged = _best(
                measure_paged,
                args.repeat,
                lengths,
                shape,
                args.runs,
                args.warmup,
                pool_blocks=POOL_CEILING_BLOCKS,
            )
            batch_blocks = sum(max(1, -(-length // 256)) for length in lengths)
            paged_small = _best(
                measure_paged,
                args.repeat,
                lengths,
                shape,
                args.runs,
                args.warmup,
                pool_blocks=batch_blocks,
            )
            # Cost per head-key touched: the two arms scan different key sets,
            # so this is the number that compares them.
            dense_keys = shape.num_heads * len(lengths) * max(lengths)
            paged_keys = shape.num_heads * sum(lengths)
            rows.append(
                {
                    "batch": batch,
                    "pattern": pattern,
                    "lengths": ",".join(str(length) for length in lengths),
                    "waste_pct": round(padding_waste(lengths) * 100, 1),
                    "dense_attn_ms": round(dense.attention_ms, 3),
                    "dense_write_ms": round(dense.write_ms, 3),
                    "dense_total_ms": round(dense.total_ms, 3),
                    "buckets": bucketed.buckets,
                    "bucketed_attn_ms": round(bucketed.attention_ms, 3),
                    "bucketed_total_ms": round(bucketed.total_ms, 3),
                    "paged_attn_ms": round(paged.attention_ms, 3),
                    "paged_write_ceiling_ms": round(paged.write_ms, 3),
                    "paged_total_ceiling_ms": round(paged.total_ms, 3),
                    "paged_write_small_ms": round(paged_small.write_ms, 3),
                    "paged_total_small_ms": round(paged_small.total_ms, 3),
                    "step_dense_ms": round(dense.total_ms * LAYERS, 3),
                    "step_bucketed_ms": round(bucketed.total_ms * LAYERS, 3),
                    "step_paged_ceiling_ms": round(paged.total_ms * LAYERS, 3),
                    "step_paged_small_ms": round(paged_small.total_ms * LAYERS, 3),
                    "dense_ns_per_head_key": round(dense.attention_ms * 1e6 / dense_keys, 3),
                    "paged_ns_per_head_key": round(paged.attention_ms * 1e6 / paged_keys, 3),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False))

    print()
    print(_arms_table(rows))
    end_reference = _best(
        measure_dense, args.repeat, reference, shape, args.runs, args.warmup
    ).attention_ms
    print(
        f"\nreference cell (8 x 1024, dense attn): {start_reference:.3f} ms at start, "
        f"{end_reference:.3f} ms at end"
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "dense-vs-paged.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(args.out / "padding.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(padding_rows[0]))
            writer.writeheader()
            writer.writerows(padding_rows)
        meta = {
            "git_commit": git_commit(),
            "mlx_version": "0.32.4",
            "shape": {
                "num_heads": shape.num_heads,
                "num_kv_heads": shape.num_kv_heads,
                "head_dim": shape.head_dim,
                "layers": LAYERS,
            },
            "mean_length": MEAN_LENGTH,
            "real_tokens": REAL_TOKENS,
            "bucket_ratio": BUCKET_RATIO,
            "pool_ceiling_blocks": POOL_CEILING_BLOCKS,
            "runs": args.runs,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "eval_floor_ms": round(floor_ms, 4),
            "reference_start_ms": round(start_reference, 3),
            "reference_end_ms": round(end_reference, 3),
            "agreement_max_abs_diff": check_agreement(lengths_for("spread4", 8), shape),
        }
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out / 'dense-vs-paged.csv'}")


if __name__ == "__main__":
    main()
