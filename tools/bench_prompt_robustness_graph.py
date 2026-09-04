#!/usr/bin/env python3
"""§5.4 heterogeneous-prompt robustness on the CUDA-Graph fast path (RTX 3050).

Runs the repo's OWN benchmark prompt set (BENCH_PROMPTS — short zh / medium
zh / system-scene, the set the bench harness actually uses) through
`run_generate(use_cuda_graph=True)` and checks per prompt:

  A. completes: 16 frames, each 8 tokens, codes within Mimi vocab
  B. deterministic: 2× same-seed runs decode to identical MD5 (extends the
     §5.1 protocol across heterogeneous inputs, not just one prompt)

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_prompt_robustness_graph.py
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle, decode_audio, run_generate
from nanovllm_omni.optim.bench.prompts import BENCH_PROMPTS

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16


def md5_of(samples: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(samples).tobytes()).hexdigest()


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = bundle.model
    eos = tok.eos_token_id

    all_finished = True
    all_det = True
    print(f"prompts: {len(BENCH_PROMPTS)} ({[p.id for p in BENCH_PROMPTS]})", flush=True)
    for p in BENCH_PROMPTS:
        ids = tok(p.text, return_tensors="pt").input_ids.cuda()
        digests = []
        for _ in range(2):
            torch.manual_seed(42)
            frames = run_generate(
                model,
                ids,
                max_new_tokens=MAX_NEW,
                temperature=0.75,
                top_p=0.9,
                eos_token_id=eos,
                open_thinking=False,
                use_cuda_graph=True,
            )
            samples = decode_audio(bundle.mimi, frames, bundle.device)
            digests.append(md5_of(samples))
            frames_ok = len(frames) == MAX_NEW and all(len(f) == 8 for f in frames)
            if not frames_ok:
                all_finished = False
        det = digests[0] == digests[1]
        if not det:
            all_det = False
        print(
            f"  {p.id:11s} len={ids.shape[1]:3d} frames={MAX_NEW} det={det} "
            f"md5={digests[0][:12]}",
            flush=True,
        )
    print(f"§5.4 ALL-DONE: {all_finished}", flush=True)
    print(f"§5.4 ALL-DETERMINISTIC: {all_det}", flush=True)
    ok = all_finished and all_det
    print(f"GRAPh PROMPT ROBUSTNESS (§5.4): {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
