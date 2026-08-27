#!/usr/bin/env python
"""Measure run_batched_generate throughput at batch=1/2/4 (decision input).

Goal: does continuous batching (the engine batched loop) improve serving
throughput for N minimind-o prompts vs running them one-by-one? Prints per
submission-batch wall median + req/s + total wav bytes, and a
"batched(X) vs X*unbatched" comparison to decide whether the batched
engine path earns its keep.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from nanovllm_omni.engine.batched_runner import run_batched_generate
from nanovllm_omni.models.minimind_omni import create_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jingyaogong/minimind-3o")
    ap.add_argument("--mimi", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    bundle = create_bundle(model_id=args.model, device=args.device, mimi=args.mimi)
    base = [args.prompt]

    medians: dict[int, float] = {}
    for batch in (1, 2, 4):
        for _ in range(args.warmup):
            run_batched_generate(bundle, base * batch, max_new_tokens=args.max_tokens)
        ws: list[float] = []
        nbytes = 0
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = run_batched_generate(bundle, base * batch, max_new_tokens=args.max_tokens)
            torch.cuda.synchronize()
            ws.append((time.perf_counter() - t0) * 1e3)
            nbytes = max(nbytes, sum(len(o.data) for o in outs))
        med = statistics.median(ws)
        medians[batch] = med
        print(
            f"batch={batch} median_wall_ms={med:.0f} req/s={batch / (med / 1e3):.2f} wav_bytes={nbytes}"
        )

    t1 = medians[1]
    for batch in (2, 4):
        tb = medians[batch]
        gain = (batch * t1) / tb - 1.0
        verdict = "BATCHING_WINS" if tb < batch * t1 else "NO_WIN"
        print(
            f"cmp batch{batch}: total_ms={tb:.0f} vs {batch}*batch1={batch * t1:.0f} "
            f"-> {verdict} (+{gain * 100:.0f}% vs serial)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
