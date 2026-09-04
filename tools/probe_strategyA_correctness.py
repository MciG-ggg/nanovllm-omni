#!/usr/bin/env python3
"""Strategy-A fixed-KV-buffer correctness probe (RTX 3050).

Minimal first question of plan §3 strategy A: if attention's per-step
`torch.cat([past_key_value[i], cur])` is replaced by a FIXED preallocated
buffer whose per-step KV lives in a stable slice (read via narrow of
[0:pos]), does the model produce bit-identical logits to the cat-based
eager path?

KV layout (verified on-device): (B, n_kv_heads, seq, head_dim) — batch,
kv-heads, sequence, head — so the fixed buffer is (B, n_kv_heads,
max_len, head_dim) and the seq axis is dim 2.

Correct only; no capture yet. If buffer-slice reads are bit-exact vs cat,
attention simply needs cat -> buffer slice-write+read (no arithmetic
change) and a single graph CAN replay across decode steps (stable buffer
address).

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/probe_strategyA_correctness.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16


class KVBuffers:
    """Fixed per-layer (K,V) buffers of max_len on the seq axis.

    as_past returns (B, pos, n_kv, d) narrows; advance writes a token.
    K/V (B, n_kv, 1, d) at index pos. ``reset`` rewrites the same storage
    with a fresh prefill so a CUDA-Graph replay keeps reading live content.
    """

    def __init__(self, init_past, max_len: int) -> None:
        self.max_len = max_len
        self.n_layer = len(init_past)
        self.pos = 0
        b, _, n_kv, d = init_past[0][0].shape  # [B, seq, n_kv, head_dim]
        shape = (b, max_len, n_kv, d)
        self.k = [torch.empty(shape, dtype=pv[0].dtype, device=pv[0].device) for pv in init_past]
        self.v = [torch.empty(shape, dtype=pv[1].dtype, device=pv[1].device) for pv in init_past]
        self.advance(init_past)  # fills prefill KV

    def reset(self, init_past) -> None:
        """REUSE the same storage (address-stable for CUDA-Graph capture);
        only rewrite contents back to the given prefill KV at pos=0."""
        self.pos = 0
        self.advance(init_past)

    def advance(self, kv_list) -> None:
        pos = self.pos
        end = pos + kv_list[0][0].shape[1]  # seq axis is dim 1
        for i, (k, v) in enumerate(kv_list):
            self.k[i][:, pos:end].copy_(k)
            self.v[i][:, pos:end].copy_(v)
        self.pos = end

    def advance_one(self, kv_list) -> None:
        """Write only the new token column(s) beyond self.pos."""
        old = self.pos
        end = kv_list[0][0].shape[1]
        if end > old:
            for i, (k, v) in enumerate(kv_list):
                self.k[i][:, old:end].copy_(k[:, old:end])
                self.v[i][:, old:end].copy_(v[:, old:end])
            self.pos = end

    def as_past(self):
        pos = self.pos
        return [
            (self.k[i].narrow(1, 0, pos), self.v[i].narrow(1, 0, pos)) for i in range(self.n_layer)
        ]


def _decode_input(nid: torch.Tensor, audio_pad: int) -> torch.Tensor:
    """[1, 9, 1]: 8 audio rows (pad) + 1 text token row."""
    buf = torch.full((1, 8, 1), audio_pad, dtype=torch.long, device=nid.device)
    return torch.cat((buf, nid.unsqueeze(-1)), dim=1)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    torch.manual_seed(42)
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    model = bundle.model
    audio_pad = int(model.config.audio_pad_token)

    # Prefill (cat path).
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=None, use_cache=True)
    past_cat = out.past_key_values

    nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    inp = _decode_input(nid, audio_pad)

    # Reference decode step via cat path.
    with torch.no_grad():
        ref = model(input_ids=inp, past_key_values=past_cat, use_cache=True)

    # Buffer path.
    buf = KVBuffers(past_cat, max_len=ids.shape[1] + MAX_NEW)
    with torch.no_grad():
        got = model(input_ids=inp, past_key_values=buf.as_past(), use_cache=True)

    diff = (ref.logits - got.logits).abs().max().item()
    same = diff <= 0.0
    print(f"step1 max|logits_cat - logits_buffer| = {diff:.3e}  bit-identical={same}", flush=True)

    # Feed-through: write the newly-generated token KV into the buffer, then
    # compare a 2nd decode step... against the cat path advanced with its KV.
    ref_present = ref.past_key_values
    got_present = got.past_key_values

    buf.advance_one(got_present)
    nid2 = ref.logits[:, -1].argmax(dim=-1, keepdim=True)
    inp2 = _decode_input(nid2, audio_pad)
    with torch.no_grad():
        ref2 = model(input_ids=inp2, past_key_values=ref_present, use_cache=True)
        got2 = model(input_ids=inp2, past_key_values=buf.as_past(), use_cache=True)
    diff2 = (ref2.logits - got2.logits).abs().max().item()
    print(f"step2 max|logits_cat - logits_buffer| = {diff2:.3e}", flush=True)

    ok = same and diff2 <= 0.0
    print(f"STRATEGY-A BUFFER CORRECTNESS: {'PASS' if ok else 'FAIL'}", flush=True)
    print("done", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
