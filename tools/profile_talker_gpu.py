#!/usr/bin/env python3
"""Profile stage 0 vs stage 1 on REAL GPU weights (RTX 3050).

Counts cudaLaunchKernel events per stage per step, wall time per step, and
compares eager vs graphed stage 0. The numbers from tools/profile_talker_vs_thinker.py
(CPU fake fixtures) suggested stage 1 has ~2064x more dispatch overhead than
graphed stage 0 -- this script measures that on real hardware.

Requires: NVIDIA GPU + torch CUDA + real minimind-3o weights locally.

Sample:
    HF_HUB_OFFLINE=1 python tools/profile_talker_gpu.py \
        --model ~/minimind-3o --mimi ~/mimi --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

_SAMPLE_TEXT = "Hello, how are you?"


def _time_cuda(fn, n: int) -> tuple[float, float, float]:
    """(median, min, max) ms over n calls with cuda.synchronize()."""
    times: list[float] = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), min(times), max(times)


def _count_kernels(fn, n: int = 1) -> tuple[int, list[tuple[str, int]]]:
    """Run fn once under torch.profiler (CUDA), return total kernel launches + top kernels.

    Returns (total_launches, [(kernel_name, count), ...]).
    """
    fn()  # warmup
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn()
    # Aggregate by CUDA kernel name
    key_avgs = prof.key_averages()
    cuda_kernels = {
        it.key: it.count
        for it in key_avgs
        if it.device_type == torch._C._autograd.DeviceType.CUDA and it.count > 0
    }
    top = sorted(cuda_kernels.items(), key=lambda x: -x[1])[:10]
    return sum(cuda_kernels.values()), top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/mcig/minimind-3o")
    parser.add_argument("--mimi", default="/home/mcig/mimi")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2

    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    print(f"model={args.model}, mimi={args.mimi}")
    print()

    from transformers import AutoTokenizer

    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle
    from nanovllm_omni.models.minimind_omni.bundle import MinimindBundle
    from nanovllm_omni.models.minimind_omni.talker import (
        MiniMindOmniTalkerForConditionalGeneration,
    )

    torch.manual_seed(args.seed)
    bundle: MinimindBundle = load_minimind_omni_bundle(model_id=args.model, mimi_model_id=args.mimi)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ids = tok(_SAMPLE_TEXT, return_tensors="pt").input_ids.cuda()
    print(f"prompt tokens: {ids.shape[1]}")

    # ===== Stage 0: joint model decode step =====
    # Eager prefill then one decode step
    with torch.no_grad():
        prefill_out = bundle.model(input_ids=ids, past_key_values=None, use_cache=True)
    past_kv = prefill_out.past_key_values
    next_id = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)

    def stage0_eager_decode():
        with torch.no_grad():
            return bundle.model(
                input_ids=next_id, past_key_values=past_kv, use_cache=True, logits_to_keep=1
            )

    # Time eager
    s0_eager_med, s0_eager_min, s0_eager_max = _time_cuda(stage0_eager_decode, args.runs)
    s0_eager_kernels, s0_eager_top = _count_kernels(stage0_eager_decode, n=1)
    print()
    print("--- Stage 0 EAGER decode step (joint model [1,1]) ---")
    print(f"  ms/call (median over {args.runs}): {s0_eager_med:.3f}")
    print(f"  ms/call (min/max):                {s0_eager_min:.3f} / {s0_eager_max:.3f}")
    print(f"  cudaLaunchKernel events/call:     {s0_eager_kernels}")
    print("  top kernels (per call):")
    for name, count in s0_eager_top[:5]:
        print(f"    {count:>4}  {name[:80]}")

    # Stage 0 graphed (warm up on side stream + capture)
    print()
    print("--- Stage 0 CUDA-GRAPH replay ---")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                _ = bundle.model(
                    input_ids=next_id, past_key_values=past_kv, use_cache=True, logits_to_keep=1
                )
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    s0_graph_med = float("nan")
    s0_graph_kernels = 0
    try:
        with torch.cuda.graph(graph), torch.no_grad():
            _ = bundle.model(
                input_ids=next_id, past_key_values=past_kv, use_cache=True, logits_to_keep=1
            )
        s0_graph_med, s0_graph_min, s0_graph_max = _time_cuda(lambda: graph.replay(), args.runs)
        s0_graph_kernels, s0_graph_top = _count_kernels(lambda: graph.replay(), n=1)
        print("  Graph capture OK")
        print(f"  ms/call (median over {args.runs}): {s0_graph_med:.3f}")
        print(f"  ms/call (min/max):                {s0_graph_min:.3f} / {s0_graph_max:.3f}")
        print(f"  cudaLaunchKernel events/call:     {s0_graph_kernels}")
    except Exception as exc:
        print(f"  Graph capture FAILED: {type(exc).__name__}: {exc}")
        print("  (Continuing to stage 1 measurement; stage 0 graph data unavailable)")

    # ===== Stage 1: talker wrapper decode step =====
    # Drive the talker wrapper directly with fabricated inputs that match
    # the real-model shapes. This isolates the cost of one stage-1 step
    # (preprocess + main forward + sample + talker_mtp) without needing
    # a full pipeline run.
    print()
    print("--- Stage 1 TALKER decode step (one _drive_talker_generation call) ---")

    from nanovllm_omni.models.minimind_omni.stage_processors import TalkerInputPayload
    from nanovllm_omni.models.minimind_omni.talker import (
        _drive_talker_generation,
        wrap_talker,
    )

    talker = bundle.talker
    if not isinstance(talker, MiniMindOmniTalkerForConditionalGeneration):
        talker = wrap_talker(bundle)

    text_hidden_size = talker.text_hidden_size
    prompt_len = 4
    num_decode_steps = 8  # enough to actually exercise the decode loop
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype

    # Fabricated bridge that matches what thinker2talker would produce.
    # Shape: [prompt_len + num_decode_steps - 1, text_hidden_size]
    bridge = torch.randn(
        prompt_len + num_decode_steps - 1,
        text_hidden_size,
        device=device,
        dtype=dtype,
    )
    payload = TalkerInputPayload(
        input_ids=torch.full(
            (prompt_len,), talker.audio_pad_token, dtype=torch.long, device=device
        ),
        bridge_states=bridge,
        text_token_ids=tuple(range(prompt_len + num_decode_steps)),
        prompt_token_ids=tuple(range(prompt_len)),
        output_token_ids=tuple(range(prompt_len, prompt_len + num_decode_steps)),
        request_id="gpu-profile",
        metadata={},
    )

    def stage1_eager():
        return _drive_talker_generation(
            talker,
            payload,
            temperature=0.2,
            top_k=50,
            do_sample=False,
        )

    # Warm + time
    stage1_eager()
    s1_med, s1_min, s1_max = _time_cuda(stage1_eager, args.runs)
    s1_kernels, s1_top = _count_kernels(stage1_eager, n=1)
    print(f"  ms/call (median over {args.runs}): {s1_med:.3f}")
    print(f"  ms/call (min/max):                {s1_min:.3f} / {s1_max:.3f}")
    print(f"  cudaLaunchKernel events/call:     {s1_kernels}")
    print("  top kernels (per call):")
    for name, count in s1_top[:5]:
        print(f"    {count:>4}  {name[:80]}")
    # Per-step breakdown
    s1_total_ops = s1_kernels
    s1_per_step = s1_total_ops / max(num_decode_steps, 1)
    print(f"  ({num_decode_steps} decode steps in call; {s1_per_step:.0f} kernels per decode step)")

    # ===== Per-request extrapolation =====
    print()
    print("--- Per-request extrapolation (typical N_AR=16, M_TALK=192) ---")
    n_ar = 16
    m_talk = 192
    # If graph failed, fall back to eager numbers + a rough estimate
    if s0_graph_med != s0_graph_med:  # NaN check
        s0_prod_per_req = s0_eager_med * n_ar
        s0_graph_label = "(eager, since graph capture failed)"
        print()
        print("  NOTE: stage 0 graph capture failed on this model -- using eager numbers")
        print("        as the per-stage cost. Real production graph is handled by the")
        print("        project's enable_cuda_graph module (see optim/cuda_graph.py).")
    else:
        s0_prod_per_req = s0_graph_med * n_ar
        s0_graph_label = f"(graphed: {s0_graph_med:.2f} ms/step)"
    s1_eager_per_req = s1_med * m_talk
    print(f"  Stage 0 {s0_graph_label}: {s0_prod_per_req:.1f} ms x {n_ar})")
    print(f"  Stage 1 eager:    {s1_eager_per_req:.1f} ms ({s1_med:.2f} ms x {m_talk})")
    print(f"  Total eager:      {s0_prod_per_req + s1_eager_per_req:.1f} ms")
    print(f"  Stage 1 / Stage 0 ratio: {s1_eager_per_req / max(s0_prod_per_req, 0.001):.1f}x")
    print()
    # With CUDA Graph on stage 1, per-step cost would be similar to stage 0 graph:
    # ~1 replay + capture overhead. Estimate from stage 0 graph numbers.
    s1_graph_est_per_step = s0_graph_med if s0_graph_med == s0_graph_med else s1_med * 0.05
    s1_graph_est_per_req = s1_graph_est_per_step * m_talk
    saved = s1_eager_per_req - s1_graph_est_per_req
    print(f"  Stage 1 GRAPHED (est): {s1_graph_est_per_req:.1f} ms")
    print(
        f"  Estimated saving: {saved:.1f} ms ({100 * saved / (s0_prod_per_req + s1_eager_per_req):.1f}% of total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
