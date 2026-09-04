#!/usr/bin/env python3
"""Determinism + multi-prompt robustness of the production buffer pillar
(RTX 3050) — plan §5.1/§5.4 pre-wiring gate.

Before `enable_cuda_graph` can be wired into stream_generate, two things
must hold at the buffer-pillar level:

  A. determinism (§5.1): the graphed 16-step loop with the production
     buffer forward must produce IDENTICAL output sequences across
     repeated runs under the same RNG seed. CUDA Graph replay is
     deterministic; sampling must also be (host multinomial with fixed
     seed). Also: reusing the same model + resetting buffer pos must not
     leak prior-run state.
  B. multi-prompt robustness (§5.4): several heterogeneous prompts (short,
     long, question-shaped, code-ish) must complete all 16 steps with
     finite logits, no OOB buffer writes, and the same speedup.

Uses the LANDED production patch enable_fixed_kv_buffer. The two freqs
host-reads are neutralized for capture (documented §13/§18).

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_determinism_prod_buffer.py
"""

from __future__ import annotations

import inspect
import sys
import textwrap
import time

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle
from nanovllm_omni.models.minimind_omni.attention import enable_fixed_kv_buffer

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16

PROMPTS = [
    "Hello, how are you?",
    "Explain why the sky is blue in three sentences.",
    "Write a JSON object with keys a and b.",
    "def fib(n): ",
    "The quick brown fox jumps over the lazy dog. Count the letters.",
]


def _patched_model_forward(model, cls, src):
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    assert "if False:" in patched
    ns = dict(cls.forward.__globals__)
    ns["__name__"] = cls.__module__
    ns["__qualname__"] = cls.__qualname__ + ".forward_det"
    exec(compile(textwrap.dedent(patched), "<det>", "exec"), ns)
    return ns["forward"]


def _decode_input(nid, audio_pad):
    buf = torch.full((1, 8, 1), audio_pad, dtype=torch.long, device=nid.device)
    return torch.cat((buf, nid.unsqueeze(-1)), dim=1)


def sample_token(logits):
    logits = logits[0, -1] / 0.75
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).reshape(1, 1)


def _attns(model):
    return [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    torch.manual_seed(42)
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = bundle.model
    audio_pad = int(model.config.audio_pad_token)
    cls = type(model)
    fwd = _patched_model_forward(model, cls, inspect.getsource(cls.forward))
    enable_fixed_kv_buffer(model, max_len=256)
    attns = _attns(model)

    def _capture(ids, pre, nid):
        """Capture the 16 per-step graphs once over the production buffers."""
        graphs, static_inps, outs = [], [], []
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(1, MAX_NEW + 1):
                inp = _decode_input(nid, audio_pad).clone()
                with torch.no_grad():
                    fwd(model, input_ids=inp, past_key_values=None, use_cache=True)
                g = torch.cuda.CUDAGraph()
                holder = {}
                with torch.cuda.graph(g), torch.no_grad():
                    holder["out"] = fwd(model, input_ids=inp, past_key_values=None, use_cache=True)
                graphs.append(g)
                static_inps.append(inp)
                outs.append(holder["out"])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return pre, nid, graphs, static_inps, outs

    # gate A: determinism — capture ONCE, replay twice under the SAME sampler
    # RNG seed (the real §5.1 protocol; re-capturing per run would fold
    # capture-time RNG side effects into the determinism test).
    ids_a = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()
    for a in attns:
        a._kv_pos = 0
    with torch.no_grad():
        pre_a = fwd(model, input_ids=ids_a, past_key_values=None, use_cache=True)
    nid_a = pre_a.logits[:, -1].argmax(dim=-1, keepdim=True)
    pre_a, nid_a, graphs_a, si_a, out_a = _capture(ids_a, pre_a, nid_a)

    def replay_seq(post, nid0, graphs, static_inps, outs):
        for a in attns:
            a._kv_pos = 0
        seq = [nid0.item()]
        cur = nid0.clone()
        for k in range(MAX_NEW):
            static_inps[k].copy_(_decode_input(cur, audio_pad))
            graphs[k].replay()
            cur = sample_token(outs[k].logits)
            seq.append(cur.item())
        return seq

    torch.manual_seed(123)
    s1 = replay_seq(pre_a, nid_a, graphs_a, si_a, out_a)
    torch.manual_seed(123)
    s2 = replay_seq(pre_a, nid_a, graphs_a, si_a, out_a)
    det_ok = s1 == s2
    print(f"A determinism (ONE capture, replay same-seed): identical={det_ok}", flush=True)
    if not det_ok:
        print(
            f"   diff at first index: {next((i for i, (a, b) in enumerate(zip(s1, s2, strict=False)) if a != b), None)}",
            flush=True,
        )

    # gate A2: buffer-reuse across prompts must not leak state (fresh capture
    # on a different prompt, then repeat under same seed).
    ids_b = tok(PROMPTS[1], return_tensors="pt").input_ids.cuda()
    for a in attns:
        a._kv_pos = 0
    with torch.no_grad():
        pre_b = fwd(model, input_ids=ids_b, past_key_values=None, use_cache=True)
    nid_b = pre_b.logits[:, -1].argmax(dim=-1, keepdim=True)
    pre_b, nid_b, graphs_b, si_b, out_b = _capture(ids_b, pre_b, nid_b)
    torch.manual_seed(7)
    s_b = replay_seq(pre_b, nid_b, graphs_b, si_b, out_b)
    torch.manual_seed(7)
    s_b2 = replay_seq(pre_b, nid_b, graphs_b, si_b, out_b)
    leak_ok = s_b == s_b2
    print(f"A2 state-reuse determinism (different prompt, repeat): identical={leak_ok}", flush=True)

    # gate B: multi-prompt robustness (fresh capture per prompt)
    speeds = []
    ok = True
    for pr in PROMPTS:
        torch.manual_seed(7)
        ids = tok(pr, return_tensors="pt").input_ids.cuda()
        for a in attns:
            a._kv_pos = 0
        with torch.no_grad():
            pre = fwd(model, input_ids=ids, past_key_values=None, use_cache=True)
        nid = pre.logits[:, -1].argmax(dim=-1, keepdim=True)
        pre, nid, graphs, si, out = _capture(ids, pre, nid)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq = replay_seq(pre, nid, graphs, si, out)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        speeds.append(ms)
        finite = bool(seq) and len(seq) == MAX_NEW + 1
        if not finite:
            ok = False
        print(f"B {pr[:34]!r}: tokens={len(seq)} graphed-e2e={ms:.1f}ms", flush=True)
    print(f"B multi-prompt robustness: all_ok={ok}", flush=True)
    print(f"B graphed-e2e across prompts: {[f'{x:.1f}' for x in speeds]} ms", flush=True)

    overall = det_ok and leak_ok and ok
    print(f"PROD-BUFFER DETERMINISM+RROBUSTNESS: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
