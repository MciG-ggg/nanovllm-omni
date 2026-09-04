#!/usr/bin/env python3
"""End-to-end eager MiniMind-O decode-loop probe (text channel, host
sampling) on RTX 3050 — measures the sampling/loop overhead that CUDA
Graph integration must sit on top of.

Report §14: full production Omni.generate = 660 ms median (16 steps).
Report §15: pure GPU forward loop = 68-78 ms/step eager, 3.5 ms/step
graphed. This probe faithfully replicates the model's own stream_generate
text branch (forward + .tolist()/multinomial host sampling + EOS-free 16
steps) to quantify how much wall the non-forward work costs. The result
bounds the end-to-end gain CUDA Graph can deliver on the REAL loop (not
just the forward part).

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_e2e_graphed_generate.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as functional

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
REPEAT = 5


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    torch.manual_seed(42)
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    audio_pad = int(bundle.model.config.audio_pad_token)

    start_pos = ids.shape[1]

    def run_one(cur, temperature=0.75, top_p=0.9):
        # Faithful text-decode subset of stream_generate: forward with
        # audio_buffer padding + past_kvs, then multinomial sampling.
        past_kvs = None
        while cur.shape[1] < start_pos + MAX_NEW:
            audio_buffer = torch.full(
                (1, 8, cur.shape[1] if past_kvs is None else 1),
                audio_pad,
                dtype=torch.long,
                device="cuda",
            )
            if past_kvs is None:
                inp = torch.cat((audio_buffer, cur.unsqueeze(1)), dim=1)
            else:
                inp = torch.cat((audio_buffer, cur[:, -1:].unsqueeze(1)), dim=1)
            with torch.no_grad():
                out = bundle.model(
                    input_ids=inp,
                    past_key_values=past_kvs,
                    use_cache=True,
                    audio_inputs=None,
                    audio_lens=None,
                )
            past_kvs = out.past_key_values
            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            logits[list(set(cur[0].tolist()))] /= 1.0
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(functional.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = float("-inf")
            next_tok = torch.multinomial(functional.softmax(logits, dim=-1), 1)
            cur = torch.cat((cur, next_tok.unsqueeze(1)), dim=1)
        return cur

    # warmup
    run_one(ids.clone())
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(ids.clone())
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    med = times[len(times) // 2]
    print(
        f"eager e2e text decode ({MAX_NEW} steps, incl. host sampling): {med:.2f} ms median",
        flush=True,
    )
    print(f"  all: {[f'{t:.1f}' for t in times]}", flush=True)
    print("  vs §14 production generate 660 ms; vs §15 forward-only 68-78 ms/step", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
