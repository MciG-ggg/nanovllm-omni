#!/usr/bin/env python3
"""Profile stage 0 (joint model forward) vs stage 1 (talker wrapper).

Answers: is stage 1 (the talker CUDA-Graph candidate) actually the bottleneck
vs stage 0 (already CUDA-Graphed)?

Uses fake fixtures (CPU-only). Numbers are dispatch-overhead proxies -- the
real wall-time win on GPU is proportional to per-op dispatch cost (~5-10us
cudaLaunchKernel each). We measure:
  - ops launched per stage 0 decode step (joint model forward, [B=1, 9, 1])
  - ops launched per stage 1 decode step (talker wrapper preprocess +
    forward + compute_logits + sample + talker_mtp)
  - ops per stage 1 prefill (one-shot over bridge prompt span)
  - wall time per stage over a fixed-size request
  - per-stage total ops for typical request (N=16 AR steps + M=192 talker
    steps)

Static attribution (mimics tests/test_ncu_attribution.py) is in the report
comments.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Add tests/ to sys.path for test helper imports (_talker_fixtures, etc.)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from _talker_fixtures import make_fake_bundle, span_bridge  # noqa: E402
from test_batched_generation import FakeMiniMindOmni  # noqa: E402

from nanovllm_omni.models.minimind_omni.stage_processors import (  # noqa: E402
    TalkerInputPayload,
)
from nanovllm_omni.models.minimind_omni.talker import (  # noqa: E402
    _drive_talker_generation,
    wrap_talker,
)


def _build_setup():
    """Build a fake bundle + wrapped talker + a fake bridge payload."""
    bundle = make_fake_bundle(
        hidden_size=8,
        vocab_size=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        max_steps_after_last_thinker_token=8,
    )
    talker = wrap_talker(bundle)
    # Fake joint model for stage 0 profiling
    fake_model = FakeMiniMindOmni(
        num_thinker_layers=2, num_talker_layers=2, kv_heads=2, head_dim=4, vocab=4096
    )
    # Bridge payload: prompt_len=4, num_decode_steps=8 (within watchdog)
    prompt_len = 4
    num_decode_steps = 8
    bridge = span_bridge(hidden_size=8, sequence_len=prompt_len + num_decode_steps - 1)
    payload = TalkerInputPayload(
        input_ids=torch.full((prompt_len,), 9, dtype=torch.long),  # audio_pad
        bridge_states=bridge,
        text_token_ids=tuple(range(prompt_len + num_decode_steps - 1)),
        prompt_token_ids=tuple(range(prompt_len)),
        output_token_ids=tuple(range(prompt_len, prompt_len + num_decode_steps - 1)),
        request_id="profile-rid",
        metadata={},
    )
    return bundle, talker, fake_model, payload, prompt_len, num_decode_steps


def _profile(fn, *, n_warmup: int = 2, n_runs: int = 5, label: str) -> dict[str, object]:
    """Run fn() n times under torch.profiler, return op count + wall time stats."""
    for _ in range(n_warmup):
        fn()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(n_runs):
            with record_function(label):
                fn()
    # Aggregate key averages
    table = prof.key_averages()
    op_count = sum(getattr(item, "count", 0) for item in table)
    cpu_time_total_us = sum(getattr(item, "cpu_time_total", 0) for item in table)
    per_op_avg_us = cpu_time_total_us / max(op_count, 1)
    top = sorted(
        ((getattr(it, "key", ""), getattr(it, "count", 0)) for it in table),
        key=lambda x: -x[1],
    )[:10]
    return {
        "label": label,
        "n_runs": n_runs,
        "total_ops": op_count,
        "ops_per_call": op_count // n_runs,
        "cpu_time_total_ms": cpu_time_total_us / 1000,
        "ms_per_call": (cpu_time_total_us / 1000) / n_runs,
        "us_per_op": per_op_avg_us,
        "top_10_ops": top,
    }


def _wall_time(fn, *, n_runs: int = 5) -> tuple[float, float]:
    """(median, total) wall time over n runs in ms."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], sum(times)


def main() -> int:
    bundle, talker, fake_model, payload, prompt_len, num_decode_steps = _build_setup()

    # Use _drive_talker_generation for the full stage 1 run (prefill + decode loop)
    def stage1_full():
        # Recreate payload each time to avoid mutation between calls
        bridge = span_bridge(hidden_size=8, sequence_len=prompt_len + num_decode_steps - 1)
        p = TalkerInputPayload(
            input_ids=torch.full((prompt_len,), 9, dtype=torch.long),
            bridge_states=bridge,
            text_token_ids=tuple(range(prompt_len + num_decode_steps - 1)),
            prompt_token_ids=tuple(range(prompt_len)),
            output_token_ids=tuple(range(prompt_len, prompt_len + num_decode_steps - 1)),
            request_id=f"profile-{time.time_ns()}",
            metadata={},
        )
        _drive_talker_generation(talker, p, temperature=0.2, top_k=50, do_sample=True)

    # --- Stage 0 decode step (joint model forward, [1, 9, 1]) ---
    def stage0_decode_step():
        inp = torch.zeros(1, 9, 1, dtype=torch.long)  # 8 audio_pad + 1 text
        # Build a fake past_kv (12 layers for fake model)
        past = []
        for _ in range(fake_model.num_layers):
            past.append((torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4)))
        fake_model.forward(inp, past_key_values=tuple(past), use_cache=True, logits_to_keep=1)

    # --- Isolated single talker.talker_mtp call (the unit CUDA Graph captures) ---
    # Fixed shapes matching what the decode loop passes every step. do_sample=False
    # matches graph capture semantics (CUDA Graphs can't capture multinomial).
    hidden_size = 8
    num_code_layers = 8
    mtp_inputs = {
        "input_ids": torch.zeros(1, dtype=torch.long),
        "input_embeds": torch.zeros(1, 1, hidden_size),
        "last_talker_hidden": torch.zeros(1, hidden_size),
        "text_step": torch.zeros(1, hidden_size),
        "active_mask": torch.ones(1, num_code_layers, dtype=torch.bool),
    }

    def talker_mtp_single():
        talker.talker_mtp(
            **mtp_inputs,
            temperature=0.2,
            top_k=50,
            do_sample=False,
        )

    # Warm
    stage1_full()
    stage0_decode_step()
    for _ in range(3):
        talker_mtp_single()

    s1_full = _profile(stage1_full, n_runs=5, label="stage1_full")
    s0_step = _profile(stage0_decode_step, n_runs=20, label="stage0_decode_step")
    mtp_single = _profile(talker_mtp_single, n_runs=100, label="talker_mtp_single")

    # Wall time (independent, no profiler overhead)
    s1_med, s1_tot = _wall_time(stage1_full, n_runs=10)
    s0_med, s0_tot = _wall_time(stage0_decode_step, n_runs=50)
    mtp_med, mtp_tot = _wall_time(talker_mtp_single, n_runs=200)

    # --- Report ---
    print("=" * 70)
    print("STAGE 0 (joint model) vs STAGE 1 (talker wrapper) — CPU profile")
    print("=" * 70)
    print("Fake bundle: hidden=8, vocab=16, layers=2 (thinker/talker tiny)")
    print(f"Prompt: {prompt_len} tokens, talker decode steps: {num_decode_steps}")
    print()

    print("--- Stage 0: ONE joint model decode step ([B=1, 9, 1]) ---")
    print(f"  ops/call:          {s0_step['ops_per_call']}")
    print(f"  ms/call (profiler): {s0_step['ms_per_call']:.3f}")
    print(f"  us/op:             {s0_step['us_per_op']:.2f}")
    print(f"  ms/call (wall):    {s0_med:.3f}  (median over 50)")
    print("  top 5 ops:")
    for op, n in s0_step["top_10_ops"][:5]:
        print(f"    {n:>5}  {op[:80]}")
    print()

    print("--- Stage 1: ONE full talker run (prefill + decode loop) ---")
    print(
        f"  ops/call:          {s1_full['ops_per_call']}  (total for 1 prefill + {num_decode_steps} decode steps)"
    )
    print(f"  ms/call (profiler): {s1_full['ms_per_call']:.3f}")
    print(f"  us/op:             {s1_full['us_per_op']:.2f}")
    print(f"  ms/call (wall):    {s1_med:.3f}  (median over 10)")
    print("  top 5 ops:")
    for op, n in s1_full["top_10_ops"][:5]:
        print(f"    {n:>5}  {op[:80]}")
    print()

    print("--- Stage 1: ONE isolated talker.talker_mtp call (graph capture unit) ---")
    print(f"  ops/call:          {mtp_single['ops_per_call']}")
    print(f"  ms/call (profiler): {mtp_single['ms_per_call']:.3f}")
    print(f"  ms/call (wall):    {mtp_med:.4f}  (median over 200)")
    print()

    ops_per_step_s1 = s1_full["ops_per_call"] // num_decode_steps
    print(f"  per-decode-step ops (stage 1):  ~{ops_per_step_s1}")
    print(
        f"  ratio stage1-step / stage0-step: "
        f"{ops_per_step_s1 / max(s0_step['ops_per_call'], 1):.2f}x"
    )
    print()

    # --- Extrapolation to realistic request ---
    n_ar = 16  # max_tokens=16 -> 16 AR steps in stage 0
    m_talk = 192  # talker_max_steps_after_last_thinker_token
    s0_total_ops = s0_step["ops_per_call"] * n_ar
    s1_total_ops = ops_per_step_s1 * m_talk
    print("--- Per-request total ops (typical n_ar=16, m_talk=192) ---")
    print(
        f"  Stage 0 total:    {s0_total_ops}  ({s0_step['ops_per_call']} ops/step x {n_ar} steps)"
    )
    print(f"  Stage 1 total:    {s1_total_ops}  (~{ops_per_step_s1} ops/step x {m_talk} steps)")
    print(f"  Ratio stage1/stage0: {s1_total_ops / max(s0_total_ops, 1):.2f}x")
    print()

    # --- With CUDA Graph (assuming each replay = 1 launch, capture = 1 launch) ---
    print("--- With CUDA Graph (each step = 1 replay, capture once) ---")
    print(f"  Stage 0 graphed:  {n_ar + 1} ops ({n_ar} replays + 1 capture)")
    print(f"  Stage 1 graphed:  {m_talk + 1} ops ({m_talk} replays + 1 capture)")
    print(
        f"  Stage 1 wall-ops saved: {s1_total_ops - (m_talk + 1)} "
        f"({100 * (s1_total_ops - (m_talk + 1)) / s1_total_ops:.1f}% reduction)"
    )
    print()

    # --- Production reality vs measured eager baseline ---
    # In production, stage 0 runs graphed (~5 ops/step = 1 replay + capture overhead)
    # via the joint-model enable_cuda_graph module. We measured its eager cost above.
    s0_prod_ops_per_step = 5  # joint model: 1 replay per step + capture prologue
    s0_prod_total = s0_prod_ops_per_step * n_ar + 1
    print("--- Production picture (Stage 0 graphed, Stage 1 EAGER) ---")
    print(
        f"  Stage 0 graphed: {s0_prod_total} ops/request  ({s0_prod_ops_per_step} ops/step x {n_ar})"
    )
    print(f"  Stage 1 eager:   {s1_total_ops} ops/request")
    print(f"  Production ratio stage1/stage0: " f"{s1_total_ops / max(s0_prod_total, 1):.1f}x")
    print("  This is the apples-to-apples baseline CUDA Graph on Stage 1 would close.")
    print()

    print("--- With Stage 1 ALSO graphed (proposed) ---")
    s1_graphed = m_talk + 1
    print(f"  Stage 0 graphed: {s0_prod_total} ops/request")
    print(f"  Stage 1 graphed: {s1_graphed} ops/request")
    print(
        f"  Ratio: {s1_graphed / max(s0_prod_total, 1):.1f}x  (currently {s1_total_ops / max(s0_prod_total, 1):.1f}x)"
    )
    print(
        f"  Estimated Stage 1 dispatch saved: "
        f"{s1_total_ops - s1_graphed} ops ({100 * (1 - s1_graphed/s1_total_ops):.1f}%)"
    )
    print()

    # --- Caveats ---
    print("--- Caveats (why CPU numbers != GPU numbers) ---")
    print("  * us/op on CPU is dispatch+framework overhead; on GPU each op is a")
    print("    cudaLaunchKernel (~5-10us host-side + ~1-3us device). The 5-10us")
    print("    host overhead is the part CUDA Graph eliminates -- graph replay")
    print("    is ~3-5us per step total instead of N*5us.")
    print("  * Tiny fake fixtures (hidden=8, layers=2) undercount per-step ops")
    print("    vs real model (hidden=768, layers=16 thinker + 8 talker). Real")
    print("    numbers should be ~10x higher per step (KV cat, QKV reshape,")
    print("    RMSNorm all scale with hidden_size + layer_count).")
    print("  * torch.compile + fused projections in attention.py cut ops by")
    print("    ~3x on the joint model forward. The talker wrapper does NOT")
    print("    get these patches (only the joint model does).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
