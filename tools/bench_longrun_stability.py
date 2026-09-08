#!/usr/bin/env python3
"""Long-run serving stability of the CUDA-Graph path (RTX 3050).

Plan §4 risk table flagged two serving-time mitigations that were only
CPU-verified: (a) shape-drift / OOB GPU-memory writes, (b) long-running
buffer + decoder reuse. This probe exercises the REAL serving pattern in
one process:

  - load bundle + enable_cuda_graph ONCE (decoder cached on model._nanovllm_graph_decoder,
    capture happens once)
  - fire 20 heterogeneous prompts from BENCH_PROMPTS through
    run_generate(use_thinker_cuda_graph=True) — reusing the SAME model+buffers
  - interleave a CONTROL prompt (same text) at i=0,5,10,15,19 and require
    its frames to be IDENTICAL (no buffer-state leak / RNG drift over time)
  - track torch.cuda.memory_allocated growth (unbounded growth = leak)

Pass = control frames identical everywhere + no OOB crash + VRAM flat.
Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_longrun_stability.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/mcig/nanovllm-omni")
import torch
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle, run_generate
from nanovllm_omni.optim.bench.prompts import BENCH_PROMPTS

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
TOTAL = 20
CONTROL_EVERY = 5
CONTROL_TEXT = "你好。"


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    bundle = load_minimind_omni_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = bundle.model
    eos = tok.eos_token_id
    control_ids = tok(CONTROL_TEXT, return_tensors="pt").input_ids.cuda()

    control_baseline = None
    all_ok = True
    vram_pts = []
    torch.manual_seed(42)
    torch.cuda.synchronize()
    v0 = torch.cuda.memory_allocated()
    for i in range(TOTAL):
        if i % CONTROL_EVERY == 0 or i == TOTAL - 1:
            ids = control_ids
        else:
            p = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
            ids = tok(p.text, return_tensors="pt").input_ids.cuda()
        torch.manual_seed(42)
        frames = run_generate(
            model,
            ids,
            max_new_tokens=MAX_NEW,
            temperature=0.75,
            top_p=0.9,
            eos_token_id=eos,
            open_thinking=False,
            use_thinker_cuda_graph=True,
        )
        torch.cuda.synchronize()
        vram_pts.append(torch.cuda.memory_allocated())
        if CONTROL_EVERY and i % CONTROL_EVERY == 0 or i == TOTAL - 1:
            if control_baseline is None:
                control_baseline = frames
                print(f"call {i}: control baseline frames={len(frames)}", flush=True)
            else:
                same = frames == control_baseline
                if not same:
                    all_ok = False
                print(
                    f"call {i}: control {'SAME' if same else 'DIFF'} frames={len(frames)}",
                    flush=True,
                )
        else:
            print(f"call {i}: prompt frames={len(frames)}", flush=True)

    growth = (vram_pts[-1] - v0) / 1e6
    maxv = max(vram_pts) / 1e6
    # Leak = SUSTAINED growth, not the cold-start jump (torch/CUDA globals +
    # capture pool allocate once at session start; end==peak means plateau).
    # Check the tail 10 calls grow only slowly.
    tail_growth = (vram_pts[-1] - vram_pts[max(0, len(vram_pts) - 10)]) / 1e6
    print(
        f"VRAM: start {v0/1e6:.0f}MB peak {maxv:.0f}MB end {vram_pts[-1]/1e6:.0f}MB "
        f"growth {growth:.1f}MB tail10 {tail_growth:.1f}MB over {TOTAL} calls",
        flush=True,
    )
    # dump the per-call VRAM trace so a plateau vs linear leak is visible
    print("VRAM trace MB (per call):", flush=True)
    print("  " + " ".join(f"{v/1e6:.0f}" for v in vram_pts), flush=True)
    flat = tail_growth < 100  # tail-10 growth <100MB = plateau, not leak
    print(
        f"LONGRUN ({TOTAL} calls): controls_identical={all_ok} memory_stable={flat} "
        f"(tail-10 {tail_growth:.1f}MB)",
        flush=True,
    )
    print(f"RESULT: {'PASS' if (all_ok and flat) else 'FAIL'}", flush=True)
    return 0 if (all_ok and flat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
