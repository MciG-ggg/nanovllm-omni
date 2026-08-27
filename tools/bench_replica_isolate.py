#!/usr/bin/env python
"""Isolate whether multi-replica / StagePool ceremony costs anything.

Same batch=4 workload through run_batched_generate with num_replicas=1 vs 2.
If they're equal, the multi-replica layer is pure ceremony (drop it).
"""

from __future__ import annotations

import statistics
import time

import torch

from nanovllm_omni.engine.batched_runner import run_batched_generate
from nanovllm_omni.models.minimind_omni import create_bundle


def main() -> int:
    bundle = create_bundle(model_id="jingyaogong/minimind-3o", device="cuda")
    prompt = "你好，请用一句话介绍你自己。"
    for num_replicas in (1, 2):
        ws: list[float] = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_batched_generate(bundle, [prompt] * 4, max_new_tokens=32, num_replicas=num_replicas)
            torch.cuda.synchronize()
            ws.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(ws)
        print(
            f"num_replicas={num_replicas} batch4 median_wall_ms={med:.0f} req/s={4 / (med / 1e3):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
