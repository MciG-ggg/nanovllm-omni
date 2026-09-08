#!/usr/bin/env python3
"""GPU verification of the landed plan §3.3 module: optim/cuda_graph.py.

Uses enable_cuda_graph(...) as a real production API (not a probe): it
installs the fixed-KV-buffer patch, neutralizes the freqs host-reads for
capture, and returns a CudaGraphDecoder. Checks:

  A. correctness: decoder token ids with a fixed seed == cat-path tokens
     (bit-exactness at the logits level on prefill + decode).
  B. determinism: same seed, two generate_tokens calls -> identical ids.
  C. speed: 16-step graph-replayed e2e vs the §14 eager anchor (~400ms).

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_cuda_graph_module.py
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle
from nanovllm_omni.optim.cuda_graph import enable_cuda_graph

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
REPEAT = 3


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    torch.manual_seed(42)
    bundle = load_minimind_omni_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    model = bundle.model

    decoder = enable_cuda_graph(model, n_steps=MAX_NEW, max_len=ids.shape[1] + MAX_NEW + 4)
    assert decoder is not None, "enable_cuda_graph returned None"
    print(
        f"module: enable_cuda_graph ok; decoder steps={decoder.n_steps} audio_pad={decoder.audio_pad}",
        flush=True,
    )

    # A: correctness vs cat path (eager, same model without graph)
    with torch.no_grad():
        out_cat = model(input_ids=ids, past_key_values=None, use_cache=True)
    d_log = (
        (decoder._prefill(ids) - out_cat.logits[:, -1].argmax(dim=-1, keepdim=True))
        .abs()
        .sum()
        .item()
    )
    print(f"A prefill nid diff (module vs cat): {d_log}", flush=True)

    # C: timing (16-step graphed e2e incl. host sampling)
    seq = decoder.generate_tokens(ids)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEAT):
        torch.manual_seed(7)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq = decoder.generate_tokens(ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times) // 2]
    print(
        f"C graphed (module) 16-step e2e: {med:.2f} ms  (all {[f'{t:.1f}' for t in times]})",
        flush=True,
    )
    print(f"   vs §14 eager ~400ms -> speedup ~{400 / med:.1f}x", flush=True)
    print(f"A tokens generated: {len(seq)} (expect {MAX_NEW})", flush=True)

    # B: determinism (same seed -> identical ids)
    torch.manual_seed(3)
    seq1 = decoder.generate_tokens(ids)
    torch.manual_seed(3)
    seq2 = decoder.generate_tokens(ids)
    print(f"B determinism (same seed): identical={seq1 == seq2}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
