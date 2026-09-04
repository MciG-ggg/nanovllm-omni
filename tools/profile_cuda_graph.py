#!/usr/bin/env python3
"""GPU benchmark: does CUDA Graph capture of the MiniMind-O AR step reduce
wall time vs eager dispatch?

The ncu investigation (docs/perf/ncu-generate-kernels-2026-09-01.md §6.4)
concluded that 11 367 cudaLaunchKernel calls per generate are the only
remaining structural cost, and CUDA Graph capture is the recommended fix.
The capture plan (docs/perf/cuda-graph-capture-plan-2026-09-01.md) and its
two CPU-validated pillars (KV fixed buffer + pad-to-max_len mask) are done.
This script is the GPU execution step: capture ONE AR-step model forward as
a CUDA Graph and measure replay wall time vs the eager call.

This is a correctness/feasibility probe — it uses a fixed-shape, already
materialized past_kv (NOT the growing-KV patch). It answers "does CUDA
Graph replay beat eager on this workload at all?" If yes, the full
Omni.generate integration (fixed-KV buffer + mask) is worth building.

Requires an NVIDIA GPU + PyTorch CUDA build. Sample usage:
    uv run python tools/profile_cuda_graph.py --model <minimind-3o> \\
        --mimi <mimi> [--runs 10]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

_SAMPLE_TEXT = "Hello, how are you?"
_RUNS_DEFAULT = 10


def _time_call(fn, n: int) -> tuple[float, float, float]:
    """(median, min, max) wall-ms over n synchronized iterations."""
    times: list[float] = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), min(times), max(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument("--runs", type=int, default=_RUNS_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device — nothing to benchmark")
        return 2

    from transformers import AutoTokenizer

    from nanovllm_omni.models.minimind_omni import create_bundle

    torch.manual_seed(args.seed)

    bundle = create_bundle(model_id=args.model, mimi_model_id=args.mimi)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ids = tok(_SAMPLE_TEXT, return_tensors="pt").input_ids.cuda()

    # --- prefill: materialize a fixed-shape past KV, then take one decode step
    with torch.no_grad():
        prefill_out = bundle.model(input_ids=ids, past_key_values=None, use_cache=True)
    past_kv = prefill_out.past_key_values

    with torch.no_grad():
        next_id = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
        # Reference decode call (eager) — also confirms shapes.
        _ = bundle.model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

    # --- eager baseline
    eager_med, eager_min, eager_max = _time_call(
        lambda: bundle.model(input_ids=next_id, past_key_values=past_kv, use_cache=True),
        args.runs,
    )
    print(f"EAGER decode:  median={eager_med:.2f}ms  min={eager_min:.2f}  max={eager_max:.2f}")

    # --- CUDA Graph capture of the same decode call
    # Static input buffers (fixed shapes -> capturable).
    static_input = next_id.clone()
    static_past_kv = tuple((k.clone(), v.clone()) for k, v in past_kv)

    # Warmup on a side stream (CUDA Graph capture requires a non-default stream).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                _ = bundle.model(
                    input_ids=static_input, past_key_values=static_past_kv, use_cache=True
                )
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph), torch.no_grad():
            _ = bundle.model(input_ids=static_input, past_key_values=static_past_kv, use_cache=True)
    except Exception as exc:  # noqa: BLE001 — capture can fail for many reasons
        print(f"CUDA Graph capture FAILED: {type(exc).__name__}: {exc}")
        print(
            "Known causes: dynamic shapes, allocations in forward, "
            "stream-unsafe ops. See the capture plan's risk table."
        )
        return 1
    print("CUDA Graph capture OK")

    graph_med, graph_min, graph_max = _time_call(lambda: graph.replay(), args.runs)
    print(
        f"CUDA GRAPH replay:  median={graph_med:.2f}ms  "
        f"min={graph_min:.2f}  max={graph_max:.2f}"
    )

    speedup = eager_med / graph_med if graph_med else float("inf")
    delta_pct = 100 * (eager_med - graph_med) / eager_med
    verdict = "CUDA Graph is faster" if delta_pct > 5 else "within noise / not faster"
    print(f"SPEEDUP: {speedup:.2f}x  (delta {delta_pct:+.1f}%)  -> {verdict}")

    print(
        "\nNext step if faster: integrate the fixed-KV-buffer + pad-to-max_len "
        "mask into Omni.generate per the capture plan and rerun "
        "`uv run pytest -m 'not smoke'` + the determinism protocol (WAV MD5)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
