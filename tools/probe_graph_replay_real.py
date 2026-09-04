#!/usr/bin/env python3
"""Single CUDA Graph over a real buffer-driven decode step (RTX 3050).

#25 proved fixed-KV-buffer decode is bit-exact vs cat (0.0 diff). #18 proved
a decode-step graph captures once the two freqs[0,0] host-reads are
neutralized. This probe ties them together on REAL state:

  A. bit-exact replay: logits from a captured graph == eager forward on the
     same (static input buffer, static KV slice).
  B. input-sensitivity: after copy_()ing a DIFFERENT token into the static
     input buffer and replaying, logits == eager forward with that new token
     and the SAME KV slice. Proves replay reads live buffer content, not a
     stale capture (the §23 correctness trap).
  C. replay speed per decode step (N replays).

If A+B pass and C is <#25 eager anchor/share, the remaining 16-step work is
pure pad+mask integration (pillars already CPU-validated) — all numeric
prerequisites now GPU-proven on real KV.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/probe_graph_replay_real.py
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
from nanovllm_omni.models.minimind_omni.attention import (
    _attention_forward,  # noqa: F401  (ensures patch targets exist)
)
from tools.probe_strategyA_correctness import KVBuffers, _decode_input

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
REPLAY_N = 200


def _patched_model_forward(model, cls, src):
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    assert "if False:" in patched
    ns = dict(cls.forward.__globals__)
    ns["__name__"] = cls.__module__
    ns["__qualname__"] = cls.__qualname__ + ".forward_graphreal"
    exec(compile(textwrap.dedent(patched), "<graph-real>", "exec"), ns)
    return ns["forward"]


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
    fwd = _patched_model_forward(model, cls, inspect.getsource(cls.forward))

    # Prefill (cat path), then one real eager decode step to get step-1 KV.
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=None, use_cache=True)
    nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    inp1 = _decode_input(nid, audio_pad)
    with torch.no_grad():
        ref1 = model(input_ids=inp1, past_key_values=out.past_key_values, use_cache=True)

    # Buffer holding prefill + step-1 KV (real).
    buf = KVBuffers(out.past_key_values, max_len=ids.shape[1] + 16)
    buf.advance_one(ref1.past_key_values)
    static_past = buf.as_past()  # narrow views at fixed pos, stable storage
    print(f"buffer pos={buf.pos} (prefill+1), shape={static_past[0][0].shape}", flush=True)

    # Static input buffer (stable address for capture/replay).
    static_inp = inp1.clone()

    # Warmup on a side stream (capture requires prior warmup in graph mode).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        fwd(model, input_ids=static_inp, past_key_values=static_past, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # --- capture ---
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        captured = fwd(model, input_ids=static_inp, past_key_values=static_past, use_cache=True)
    torch.cuda.synchronize()

    # --- A: bit-exact replay vs eager on identical inputs ---
    with torch.no_grad():
        eager_a = fwd(model, input_ids=static_inp, past_key_values=static_past, use_cache=True)
    diff_a = (captured.logits - eager_a.logits).abs().max().item()
    print(
        f"A bit-exact replay: max|graph - eager| = {diff_a:.3e}  pass={diff_a == 0.0}", flush=True
    )

    # --- B: input-sensitivity — copy a DIFFERENT token, replay, compare ---
    other_tok = ref1.logits[:, -1].argmax(dim=-1, keepdim=True)
    inp2 = _decode_input(other_tok, audio_pad)
    static_inp.copy_(inp2)
    g.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        eager_b = fwd(model, input_ids=static_inp, past_key_values=static_past, use_cache=True)
    diff_b = (captured.logits - eager_b.logits).abs().max().item()
    diff_inp = (inp1 - inp2).abs().sum().item()
    print(
        f"B input-sensitivity: diff on inputs = {diff_inp} (nonzero expected), max|graph(new)-eager(new)| = {diff_b:.3e}  pass={diff_b == 0.0}",
        flush=True,
    )

    # --- C: replay speed ---
    g.replay()  # restore consistency
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPLAY_N):
        g.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / REPLAY_N * 1000
    print(f"C replay x{REPLAY_N}: {ms:.3f} ms/step", flush=True)

    ok = diff_a == 0.0 and diff_b == 0.0
    print(f"GRAPh REPLAY on REAL buffer state: {'PASS' if ok else 'FAIL'}", flush=True)
    print("done", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
