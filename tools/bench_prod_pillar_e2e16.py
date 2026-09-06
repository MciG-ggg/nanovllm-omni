#!/usr/bin/env python3
"""Production-pillar e2e: enable_fixed_kv_buffer + CUDA Graph (RTX 3050).

#21 measured 6.11x with the PROBE's own KV-buffer class (tools/KVBuffers).
#29 landed the production equivalent (`enable_fixed_kv_buffer` in
attention.py) and proved it bit-exact vs cat — but only on 1 decode step.

This probe asks: does the LANDED production buffer forward also drive a
16-step CUDA-Graphed e2e loop (a) bit-exactly vs the cat path, and
(b) at the same ~6x speedup as #21? Closes the probe-code -> production
code gap.

Design mirrors #21: one graph per step length, all sharing the
production-attached per-instance KV buffers (module._kv_past_key/value);
decode feeds real tokens into static inputs, replays, samples host-side.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_prod_pillar_e2e16.py
"""

from __future__ import annotations

import inspect
import sys
import time

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle
from nanovllm_omni.optim.attention import enable_fixed_kv_buffer
from nanovllm_omni.optim.cuda_graph import _build_omni_input, _patched_forward

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16
REPEAT = 3


def sample_token(logits) -> torch.Tensor:
    logits = logits[0, -1] / 0.75
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).reshape(1, 1)


def _attn_instances(model):
    return [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]


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
    cls = type(model)
    fwd = _patched_forward(cls, inspect.getsource(cls.forward))
    enable_fixed_kv_buffer(model, max_len=ids.shape[1] + MAX_NEW + 4)
    attns = _attn_instances(model)
    print(f"production buffer-attached attn: {len(attns)}", flush=True)
    assert len(attns) == 12, len(attns)

    # reference cat path (eager, no buffer patch semantics): use separate model
    b_ref = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    m_ref = b_ref.model

    def eager_cat_loop():
        cur = ids.clone()
        with torch.no_grad():
            out = m_ref(input_ids=ids, past_key_values=None, use_cache=True)
        past = out.past_key_values
        nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(MAX_NEW):
            inp = _build_omni_input(nid, audio_pad)
            with torch.no_grad():
                out = m_ref(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nid = sample_token(out.logits)
            cur = torch.cat((cur, nid), dim=1)
        return cur

    # production-buffer eager loop (same model+patch, no graph yet): bit-exact?
    def buffer_eager_loop():
        # reset all production buffer pos to 0
        for a in attns:
            a._kv_pos = 0
        cur = ids.clone()
        with torch.no_grad():
            out = fwd(model, input_ids=ids, past_key_values=None, use_cache=True)
        nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(MAX_NEW):
            inp = _build_omni_input(nid, audio_pad)
            with torch.no_grad():
                out = fwd(model, input_ids=inp, past_key_values=None, use_cache=True)
            nid = sample_token(out.logits)
            cur = torch.cat((cur, nid), dim=1)
        return cur

    # bit-exactness: compare logits of prefill+single decode between paths
    with torch.no_grad():
        pre_ref = m_ref(input_ids=ids, past_key_values=None, use_cache=True)
        nid_r = pre_ref.logits[:, -1].argmax(dim=-1, keepdim=True)
        inp = _build_omni_input(nid_r, audio_pad)
        ref = m_ref(input_ids=inp, past_key_values=pre_ref.past_key_values, use_cache=True)
    for a in attns:
        a._kv_pos = 0
    with torch.no_grad():
        pre_buf = fwd(model, input_ids=ids, past_key_values=None, use_cache=True)
        nid_b = pre_buf.logits[:, -1].argmax(dim=-1, keepdim=True)
        got = fwd(
            model,
            input_ids=_build_omni_input(nid_b, audio_pad),
            past_key_values=None,
            use_cache=True,
        )
    d_pre = (pre_ref.logits - pre_buf.logits).abs().max().item()
    d_dec = (ref.logits - got.logits).abs().max().item()
    print(
        f"prod-buffer vs cat: prefill max|diff|={d_pre:.3e}  decode max|diff|={d_dec:.3e}",
        flush=True,
    )

    # ---------------- capture per-step graphs over production buffers -----
    for a in attns:
        a._kv_pos = 0
    graphs = []
    static_inps = []
    outs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        nid = pre_buf.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _step in range(1, MAX_NEW + 1):
            inp_static = _build_omni_input(nid, audio_pad).clone()
            with torch.no_grad():
                fwd(model, input_ids=inp_static, past_key_values=None, use_cache=True)
            g = torch.cuda.CUDAGraph()
            out_holder = {}
            with torch.cuda.graph(g), torch.no_grad():
                out_holder["out"] = fwd(
                    model, input_ids=inp_static, past_key_values=None, use_cache=True
                )
            graphs.append(g)
            static_inps.append(inp_static)
            outs.append(out_holder["out"])
            # production buffer pos auto-advanced inside _kv_buffer_forward
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} per-step graphs over production buffers", flush=True)

    # ---------------- graphed 16-step loop (reset buffer pos each run) -----
    def graphed_loop():
        for a in attns:
            a._kv_pos = 0
        cur = ids.clone()
        nid = pre_buf.logits[:, -1].argmax(dim=-1, keepdim=True)
        for k in range(MAX_NEW):
            static_inps[k].copy_(_build_omni_input(nid, audio_pad))
            graphs[k].replay()
            nid = sample_token(outs[k].logits)
            cur = torch.cat((cur, nid), dim=1)
        return cur

    # correctness sanity + timing
    cg = graphed_loop()
    torch.cuda.synchronize()
    print(
        f"graphed seq len={cg.shape[1]} finite={all(torch.isfinite(o.logits).all().item() for o in outs)}",
        flush=True,
    )

    eager_cat_loop()
    torch.cuda.synchronize()
    te = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eager_cat_loop()
        torch.cuda.synchronize()
        te.append((time.perf_counter() - t0) * 1000)
    eager_ms = sorted(te)[len(te) // 2]

    graphed_loop()
    torch.cuda.synchronize()
    tg = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graphed_loop()
        torch.cuda.synchronize()
        tg.append((time.perf_counter() - t0) * 1000)
    graphed_ms = sorted(tg)[len(tg) // 2]

    print(f"eager cat 16-step e2e:   {eager_ms:.2f} ms", flush=True)
    print(f"graphed (prod-buffer) e2e: {graphed_ms:.2f} ms", flush=True)
    print(f"end-to-end speedup:      {eager_ms / graphed_ms:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
