#!/usr/bin/env python3
"""Bench: single-token decode loop.

Minimal prefill-once / decode-N-times measurement on the loaded bundle.
Records per-step decode latency and VRAM peak.

This is intentionally simpler than ``bench_paged_single_graph.py``: one
decode path, one cell, no graph capture. Use ``verify_phase4_wire.py``
when you want shape checks, bit-equal, and the E2E WAV.

Usage:
    python tools/bench_decode_loop.py \\
        --model-path /home/mcig/minimind-3o \\
        --batch-size 4 --num-repeats 20 --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_HF_PATH = (
    "/home/mcig/.cache/huggingface/hub/models--jingyaogong--minimind-3o/"
    "snapshots/ee3febbd08cc5b2bd41c039c825a8934232fee33/"
)


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": statistics.median(s),
        "p95": s[max(int(0.95 * len(s)) - 1, 0)],
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "n": len(s),
    }


def _sync_ms(fn: Any) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _load_thinker(model_path: str) -> tuple[Any, torch.device]:
    """Load bundle, return MiniMindThinker on cuda (fp16)."""
    from nanovllm_omni.models.minimind_omni.bundle import load_minimind_omni_bundle
    from nanovllm_omni.models.minimind_omni.thinker import MiniMindThinker

    bundle = load_minimind_omni_bundle(model_id=model_path, device="cuda")
    thinker = bundle.thinker
    if not isinstance(thinker, MiniMindThinker):
        # Phase 3 wires the bundle to expose MiniMindThinker/Talker directly;
        # when it hasn't run yet this tool is useless, so say so plainly.
        raise SystemExit(
            "bundle.thinker is not MiniMindThinker "
            f"(got {type(thinker).__name__}); Phase 3 wiring incomplete"
        )
    if not next(thinker.parameters()).is_cuda:
        thinker = thinker.cuda().half().eval()
    return thinker, torch.device("cuda")


def _bench_decode(
    thinker: Any,
    batch_size: int,
    num_repeats: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Run ``num_repeats`` trials; each trial = prefill + ``max_tokens`` decodes.

    Per-step decode latency is collected across all trials; the VRAM peak
    is the max across trials. ``num_repeats`` trials exist so first-call
    init costs in trial 0 don't dominate p50/p95 — the warm-up of one
    decode step inside each trial is the dominant init cost we throw away.
    """
    device = next(thinker.parameters()).device
    vocab_size = getattr(thinker, "vocab_size", 6400)
    prefill_len = 8

    samples: list[float] = []
    peak_vram_mb = 0.0
    for _ in range(num_repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        input_ids = torch.randint(0, vocab_size, (batch_size, prefill_len), device=device)
        positions = torch.arange(prefill_len, device=device).expand(batch_size, prefill_len)

        with torch.no_grad():
            _ = thinker(input_ids, positions)
            torch.cuda.synchronize()

            pos = prefill_len
            for step in range(max_tokens):
                ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
                ps = torch.full((batch_size, 1), pos, device=device)
                ms = _sync_ms(lambda ids=ids, ps=ps: thinker(ids, ps))
                # Drop the first step of the very first trial — that's where
                # the JIT / first-touch warm-up lands.
                if not (step == 0 and len(samples) == 0):
                    samples.append(ms)
                pos += 1

        peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / 2**20)

    return {**_stats(samples), "vram_mb": peak_vram_mb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_HF_PATH)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-repeats", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this bench needs a real GPU")

    thinker, device = _load_thinker(args.model_path)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(
        f"batch_size={args.batch_size} num_repeats={args.num_repeats} "
        f"max_tokens={args.max_tokens}"
    )

    result = _bench_decode(
        thinker=thinker,
        batch_size=args.batch_size,
        num_repeats=args.num_repeats,
        max_tokens=args.max_tokens,
    )

    print(
        f"{'metric':>8} {'value':>10}\n"
        f"{'p50':>8} {result['p50']:>10.2f}\n"
        f"{'p95':>8} {result['p95']:>10.2f}\n"
        f"{'min':>8} {result['min']:>10.2f}\n"
        f"{'max':>8} {result['max']:>10.2f}\n"
        f"{'mean':>8} {result['mean']:>10.2f}\n"
        f"{'n':>8} {result['n']:>10d}\n"
        f"{'vram_mb':>8} {result['vram_mb']:>10.1f}"
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
