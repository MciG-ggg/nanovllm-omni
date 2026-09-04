#!/usr/bin/env python3
"""Text-token alignment v2 (RTX 3050) — decoder vs production.

#33 showed decoder start-from-argmax diverges from production's sample-
token0 at index 0. The decoder was fixed (plan-appendix fact #1): it now
samples token0 from prefill logits with the SAME `sample_text_token` used
by the production runner, and uses that sampler for every subsequent
replayed step.

This probe re-checks alignment under the SAME torch seed:
- reference: `stream_generate` text_tokens (production path)
- candidate: `CudaGraphDecoder.generate_tokens` (fixed module)

Matching means wiring the decoder into the real path cannot change the
model's text. Speed recorded for continuity with §14/§24.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_detext_align.py
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle
from nanovllm_omni.models.minimind_omni.generation import stream_generate
from nanovllm_omni.optim.cuda_graph import enable_cuda_graph

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
PROMPT = "Hello, how are you?"


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    torch.manual_seed(42)
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    model = bundle.model

    # reference: production path (same torch seed => RNG drives both)
    torch.manual_seed(7)
    ref_text: list[int] = []
    for text_chunk, _audio in stream_generate(
        model,
        ids,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=MAX_NEW,
        temperature=0.75,
        top_p=0.9,
        use_cache=True,
        return_audio_codes=True,
        open_thinking=False,
    ):
        if text_chunk is not None:
            ref_text.extend(int(t) for t in text_chunk.reshape(-1).tolist())
    ref_text = ref_text[:MAX_NEW]

    # candidate: fixed decoder (same torch seed; it seeds its own generator
    # from torch.initial_seed())
    decoder = enable_cuda_graph(model, n_steps=MAX_NEW, max_len=ids.shape[1] + MAX_NEW + 4)
    assert decoder is not None
    torch.manual_seed(7)
    t0 = time.perf_counter()
    dec_text = decoder.generate_tokens(ids)
    ms = (time.perf_counter() - t0) * 1000

    match = dec_text == ref_text
    print(f"production stream_generate: {ref_text}", flush=True)
    print(f"decoder (fixed sampler):   {dec_text}", flush=True)
    print(f"MATCH ({PROMPT!r}, seed 7): {match}", flush=True)
    if not match:
        d = next(
            (i for i, (a, b) in enumerate(zip(dec_text, ref_text, strict=False)) if a != b), None
        )
        print(f"first divergence at index {d}", flush=True)
    print(f"decoder 16-step graphed: {ms:.1f} ms", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
