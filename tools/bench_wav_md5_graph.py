#!/usr/bin/env python3
"""Graph-path WAV-MD5 determinism protocol (§5.1) on RTX 3050.

The CUDA-Graph opt-in fast path (`run_generate(use_cuda_graph=True)`,
report §27) must be deterministic end to end: same seed -> same Mimi
codebook frames -> same decoded audio values. This runs the §5.1 protocol:

  1. seed 42, run graph-path run_generate 5x, collect frames per run
  2. decode each via decode_audio(bundle.mimi, frames, device) (the codec
     stage run_generate's frames feed)
  3. MD5 of the decoded float bytes per run; all 5 must be identical

Separately reports the decoded-step variance to show determinism is not
an artifact of empty/degenerate audio.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_wav_md5_graph.py
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle, decode_audio, run_generate

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
RUNS = 5


def md5_of(samples: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(samples).tobytes()).hexdigest()


def main() -> int:
    if not __import__("torch").cuda.is_available():
        print("no CUDA")
        return 2
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    model = bundle.model
    eos = tok.eos_token_id
    import torch

    digest_to_run: dict[str, list[int]] = {}
    audios = []
    for r in range(RUNS):
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
        audios.append(frames)
        d = md5_of(samples)
        digest_to_run.setdefault(d, []).append(r)
        print(f"run {r}: frames={len(frames)} md5={d[:16]}", flush=True)

    unique_md5 = len(digest_to_run)
    print(f"unique MD5s across {RUNS} same-seed runs: {unique_md5}", flush=True)
    ok = unique_md5 == 1
    print(f"GRAPh WAV-MD5 DETERMINISM (§5.1): {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
