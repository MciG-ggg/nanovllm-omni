#!/usr/bin/env python3
"""Isolate ONE talker.talker_mtp call to count ops the CUDA Graph would capture.

The stage-1 talker_mtp path is what enable_talker_mtp_cuda_graph captures
per-step. This script measures JUST that call (no prefill, no surrounding
driver loop) to get a clean ops/step number for the graphed surface.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from _talker_fixtures import make_fake_bundle  # noqa: E402

from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402


def main() -> int:
    bundle = make_fake_bundle(
        hidden_size=8, vocab_size=16, num_hidden_layers=2, max_position_embeddings=32
    )
    talker = wrap_talker(bundle)

    hidden_size = 8
    num_code_layers = 8
    # Fixed shapes — these are what the decode loop passes every step.
    input_ids = torch.zeros(1, dtype=torch.long)
    input_embeds = torch.zeros(1, 1, hidden_size)
    last_talker_hidden = torch.zeros(1, hidden_size)
    text_step = torch.zeros(1, hidden_size)
    active_mask = torch.ones(1, num_code_layers, dtype=torch.bool)

    # Warmup (also patches talker_mtp through the wrapper's __init__ hooks)
    for _ in range(3):
        talker.talker_mtp(
            input_ids=input_ids,
            input_embeds=input_embeds,
            last_talker_hidden=last_talker_hidden,
            text_step=text_step,
            active_mask=active_mask,
            temperature=0.2,
            top_k=50,
            do_sample=True,
        )

    # Profile
    n = 100
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(n):
            with record_function("talker_mtp_one_call"):
                talker.talker_mtp(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    last_talker_hidden=last_talker_hidden,
                    text_step=text_step,
                    active_mask=active_mask,
                    temperature=0.2,
                    top_k=50,
                    do_sample=True,
                )

    table = prof.key_averages()
    total_ops = sum(getattr(it, "count", 0) for it in table)
    total_us = sum(getattr(it, "cpu_time_total", 0) for it in table)
    ops_per_call = total_ops // n
    us_per_call = total_us / n
    top = sorted(
        ((getattr(it, "key", ""), getattr(it, "count", 0)) for it in table),
        key=lambda x: -x[1],
    )[:15]

    # Wall time
    t0 = time.perf_counter()
    for _ in range(200):
        talker.talker_mtp(
            input_ids=input_ids,
            input_embeds=input_embeds,
            last_talker_hidden=last_talker_hidden,
            text_step=text_step,
            active_mask=active_mask,
            temperature=0.2,
            top_k=50,
            do_sample=True,
        )
    wall_ms = (time.perf_counter() - t0) / 200 * 1000

    print("=" * 60)
    print("ONE talker.talker_mtp() call (CPU, fake fixture)")
    print("=" * 60)
    print("  Shape: input_ids=[1], input_embeds=[1,1,H=8],")
    print("          last_talker_hidden=[1,H=8], active_mask=[1,8]")
    print(f"  ops/call:           {ops_per_call}")
    print(f"  us/call (profiler): {us_per_call:.2f}")
    print(f"  ms/call (wall):     {wall_ms:.3f}")
    print(f"  top 10 ops (over {n} calls, total counts):")
    for op, count in top[:10]:
        avg = count // n
        print(f"    {count:>6}  (~{avg:>3}/call)  {op[:70]}")

    # Compare to graph replay cost (single replay = ~1 op)
    saved_per_step = max(0, ops_per_call - 1)
    saved_per_request_192 = saved_per_step * 192
    print()
    print("--- CUDA Graph impact (assume replay = 1 op) ---")
    print(f"  Per-step reduction:  {ops_per_call} -> 1 ({saved_per_step} saved)")
    print(
        f"  Per-request (192 steps): {ops_per_call * 192} -> 193 ops "
        f"({saved_per_request_192} saved = "
        f"{100 * saved_per_request_192 / (ops_per_call * 192):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
