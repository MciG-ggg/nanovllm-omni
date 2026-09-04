#!/usr/bin/env python3
"""defect #5 isolation probe: buffer-residue vs graph-slice-read (RTX 3050).

DECISION TOOL, not a fix. Runs the cross-prompt leak A/B that #49 found:

  A. same-prompt repeat — must be IDENTICAL (determinism control)
  B. cross-prompt (interleave a different prompt) then control — was DIFF
     (defect #5)

This probe adds a second arm: explicit zero of each attention buffer's
DECODE-KV region (i.e. past prefill_len) between prompts. If zeroing makes
B identical, the leak is buffer content residue (#50's leading hypothesis);
if B stays DIFF even with zeroing, the graph replay reads a stale slice
range regardless of content (mechanism, not residue).

Implements the isolation protocol from the plan doc (defect #5 open item).
GPU-only (CUDA Graph). Run on WSL when SSH is back:

  cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_longrun_residue_probe.py
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/home/mcig/nanovllm-omni")
import torch
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle, run_generate

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
CONTROL = "你好。"
INTERLEAVE = "请用两句话介绍MiniMind-O模型。"


def _decode_region(prefill_len: int, max_len: int) -> slice:
    """Buffer positions beyond the current prefill (the decode-KV region)."""
    return slice(prefill_len, max_len)


def _zero_decode_region(model: Any, prefill_len: int) -> int:
    """Zero every attention buffer beyond prefill_len (decode-KV region).
    Returns how many buffers were touched."""
    count = 0
    for m in model.modules():
        k = getattr(m, "_nanovllm_kv_buffer", None)
        if k is None:
            continue
        max_len = getattr(m, "_kv_past_key", None).shape[1]
        m._kv_past_key[:, _decode_region(prefill_len, max_len)] = 0.0
        m._kv_past_value[:, _decode_region(prefill_len, max_len)] = 0.0
        count += 1
    return count


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = bundle.model
    eos = tok.eos_token_id

    def run(txt: str):
        ids = tok(txt, return_tensors="pt").input_ids.cuda()
        return run_generate(
            model,
            ids,
            max_new_tokens=MAX_NEW,
            temperature=0.75,
            top_p=0.9,
            eos_token_id=eos,
            open_thinking=False,
            use_cuda_graph=True,
        )

    # --- without zeroing (baseline repro) --------------------------------
    a0 = run(CONTROL)
    run(INTERLEAVE)
    a1 = run(CONTROL)
    same_prompt = run(CONTROL)
    print(f"BASELINE: same-prompt repeat {same_prompt == run(CONTROL)}", flush=True)
    print(f"BASELINE: cross-prompt B {a0 == a1}  (defect #5 repro)", flush=True)

    # --- with zeroing of decode-KV region between prompts -----------------
    # zero BEFORE the interleave and before the return-to-control, so any
    # residue from the previous prompt's decode is cleared.
    ctrl0 = run(CONTROL)
    _ = run(INTERLEAVE)
    prefill_len = len(tok(INTERLEAVE).data["input_ids"])
    zeroed = _zero_decode_region(model, prefill_len)
    run(CONTROL)  # re-establish control with fresh KV
    pre = len(tok(CONTROL).data["input_ids"])
    zeroed += _zero_decode_region(model, pre)
    ctrl1 = run(CONTROL)
    ctrl_repeat = run(CONTROL)
    print(f"ZEROING: touched {zeroed} buffers", flush=True)
    print(f"ZEROING: cross-prompt B' {ctrl0 == ctrl1}  (repeat {ctrl1 == ctrl_repeat})", flush=True)

    # verdict
    base_b = a0 == a1
    zero_b = ctrl0 == ctrl1
    print(
        f"VERDICT: zeroing {'FIXES' if (not base_b and zero_b) else 'does NOT fix'} "
        f"cross-prompt leak (base B={base_b}, zero B={zero_b})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
