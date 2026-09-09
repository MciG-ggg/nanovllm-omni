#!/usr/bin/env python3
"""Bench: single-graph paged decode vs eager.

Two cells, one process, same prompts / seed / token budget:

1. ``eager``   -- no CUDA Graph at all
2. ``paged``   -- ``optim/paged_cuda_graph``: ONE graph, replayed for
   every decode step, over a paged KV cache

What this is actually measuring
-------------------------------
The interesting number is the **cost of getting into** the hot state:

- ``capture_ms``: paged pays one capture
- ``graphs``: VRAM holds 1 graph
- ``recapture on n_steps change``: paged must not recapture when the
  token budget changes (n_steps is not a capture shape)

Cell isolation
--------------
``enable_paged_kv_cache`` swaps wrapper modules into the model, so it
must be uninstalled between cells or the "eager" cell silently measures
the paged path. ``disable_paged_kv_cache`` is called before every cell
for that reason (a no-op on a fresh model).

Usage:
    python tools/bench_paged_single_graph.py \\
        --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \\
        --max-new-tokens 16 --repeats 20 --out docs/perf/aligned/paged-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

PROMPTS = ["Hi", "Hello there", "Tell me a short story"]
SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9


def _time_ms(fn: Any) -> tuple[float, Any]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": statistics.median(s),
        "p95": s[max(int(0.95 * len(s)) - 1, 0)],
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mimi", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="docs/perf/aligned/paged-v1")
    ap.add_argument(
        "--no-flash",
        action="store_true",
        help="force torch-native SDPA even when flash-attn is installed; "
        "lets you compare kernel paths on the same torch version",
    )
    args = ap.parse_args()
    if args.no_flash:
        os.environ["NANOVLLM_DISABLE_FLASH"] = "1"

    from nanovllm_omni.models.minimind_omni.bundle import load_minimind_omni_bundle
    from nanovllm_omni.models.minimind_omni.paged_attention import disable_paged_kv_cache
    from nanovllm_omni.models.minimind_omni.paged_cuda_graph import enable_paged_cuda_graph
    from nanovllm_omni.models.minimind_omni.thinker import (
        run_generate,
        tokenize_for_generate,
    )

    bundle = load_minimind_omni_bundle(model_id=args.model, device="cuda", mimi_model_id=args.mimi)
    model, tok = bundle.model, bundle.tokenizer
    eos = tok.eos_token_id
    n = args.max_new_tokens

    ids = [tokenize_for_generate(tok, p, False).to("cuda") for p in PROMPTS]
    rows: list[dict[str, Any]] = []

    def _eager(x: Any) -> Any:
        return run_generate(
            model,
            x,
            max_new_tokens=n,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=eos,
            open_thinking=False,
            seed=SEED,
            use_thinker_cuda_graph=False,
        )

    # ---- cell 1: eager -------------------------------------------------
    disable_paged_kv_cache(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _time_ms(lambda: _eager(ids[0]))  # warm the allocator, discard
    eager_samples, eager_frames = [], 0
    for x in ids:
        for _ in range(args.repeats):
            t, frames = _time_ms(lambda x=x: _eager(x))
            eager_samples.append(t)
            eager_frames = len(frames)
    rows.append(
        {
            "cell": "eager",
            "graphs": 0,
            "capture_ms": 0.0,
            "cold_ms": 0.0,
            "frames": eager_frames,
            "vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            **_stats(eager_samples),
        }
    )

    # ---- cell 2: paged single graph ------------------------------------
    disable_paged_kv_cache(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dec = enable_paged_cuda_graph(model, n_steps=n, eos_token_id=eos)
    if dec is None:
        raise SystemExit("paged decoder unavailable")
    dec.temperature, dec.top_p = TEMPERATURE, TOP_P
    cold_paged, (_, audio) = _time_ms(
        lambda: dec.generate_tokens(ids[0], seed=SEED, return_audio=True)
    )
    pg_samples = []
    for x in ids:
        for _ in range(args.repeats):
            t, (_, audio) = _time_ms(
                lambda x=x: dec.generate_tokens(x, seed=SEED, return_audio=True)
            )
            pg_samples.append(t)
    rows.append(
        {
            "cell": "paged",
            "graphs": 1,
            "capture_ms": cold_paged - statistics.median(pg_samples),
            "cold_ms": cold_paged,
            "frames": len(audio[0]),
            "vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            **_stats(pg_samples),
        }
    )

    # ---- capture invalidation: n_steps must not be a capture shape -----
    window_before = dec._captured_window
    dec.n_steps = max(n - 4, 2)
    dec.generate_tokens(ids[0], seed=SEED, return_audio=True)
    recaptured = dec._captured_window != window_before

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (out_dir / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    eager_p50 = rows[0]["p50"]
    meta = {
        "max_new_tokens": n,
        "repeats": args.repeats,
        "prompts": PROMPTS,
        "seed": SEED,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "recapture_on_n_steps_change": recaptured,
        "speedup_vs_eager": {r["cell"]: round(eager_p50 / r["p50"], 3) for r in rows},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"{'cell':8} {'graphs':>7} {'p50':>9} {'cold':>9} {'capture':>9} {'vram_mb':>8} {'frames':>7}"
    )
    for r in rows:
        print(
            f"{r['cell']:8} {r['graphs']:7d} {r['p50']:9.1f} {r['cold_ms']:9.1f} "
            f"{r['capture_ms']:9.1f} {r['vram_mb']:8.0f} {r['frames']:7d}"
        )
    print(f"\nspeedup vs eager: {meta['speedup_vs_eager']}")
    print(f"recapture when n_steps changes: {recaptured} (paged must be False)")
    print(f"wrote {out_dir}/cells.csv + meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
