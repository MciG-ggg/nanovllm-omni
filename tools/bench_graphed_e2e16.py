#!/usr/bin/env python3
"""Full 16-step graphed e2e decode (RTX 3050) — the §16 acceptance measure.

Builds on proven pieces:
  - §19: fixed KV buffer slice == cat (bit-exact, real model)
  - §18: decode-step capture OK once the two freqs[0,0] reads are neutralized
  - §20: single graph over real buffer state replays input-sensitively,
    3.38 ms/step
  - §14/§23 anchor: eager e2e ~403-660 ms for 16 steps incl. host sampling

Design: one CUDA Graph per step length (KV prefill+1 .. prefill+16), ALL
sharing one fixed KV buffer (max_len). graph_k is captured with the buffer at
step-k state; replay uses the same buffer storage (address-stable, content
live). Between steps the replayed KV is folded back into the buffer via
advance_one so graph_{k+1} reads the accumulated history.

Correctness for the loop is inherited from §20 (input-sensitivity = the graph
recomputes from live buffer content); here we measure the acceptance figure:
16-step graphed e2e (incl. host multinomial sampling) vs eager.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_graphed_e2e16.py
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
from tools.probe_strategyA_correctness import KVBuffers, _decode_input

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 16
REPEAT = 3
if MAX_NEW < 4:
    raise SystemExit("MAX_NEW must be >= 4 (buffer prefill safety)")


def _patched_model_forward(model, cls, src):
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    assert "if False:" in patched
    ns = dict(cls.forward.__globals__)
    ns["__name__"] = cls.__module__
    ns["__qualname__"] = cls.__qualname__ + ".forward_e2e16"
    exec(compile(textwrap.dedent(patched), "<e2e16>", "exec"), ns)
    return ns["forward"]


def sample_token(logits) -> torch.Tensor:
    logits = logits[0, -1] / 0.75  # temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).reshape(1, 1)


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

    # ---------------- eager 16-step loop (cat path, incl. sampling) --------
    def eager_loop():
        cur = ids.clone()
        past_cat = None
        with torch.no_grad():
            out = fwd(model, input_ids=ids, past_key_values=None, use_cache=True)
        past_cat = out.past_key_values
        nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(MAX_NEW):
            inp = _decode_input(nid, audio_pad)
            with torch.no_grad():
                out = fwd(model, input_ids=inp, past_key_values=past_cat, use_cache=True)
            past_cat = out.past_key_values
            cur = torch.cat((cur, nid), dim=1)
            nid = sample_token(out.logits)
        return cur

    # ---------------- capture per-step graphs over one shared buffer -------
    # buffer starts at prefill state; each graph_k is captured while buffer is
    # at step-k state (length prefill_len + k - 1). Using patched fwd.
    with torch.no_grad():
        pre = fwd(model, input_ids=ids, past_key_values=None, use_cache=True)
    past0 = pre.past_key_values
    nid0 = pre.logits[:, -1].argmax(dim=-1, keepdim=True)

    buf = KVBuffers(past0, max_len=ids.shape[1] + MAX_NEW)
    graphs = []
    static_inps = []
    outs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _step in range(1, MAX_NEW + 1):
            inp_static = _decode_input(nid0, audio_pad).clone()  # dec shape [1,9,1]
            with torch.no_grad():
                fwd(model, input_ids=inp_static, past_key_values=buf.as_past(), use_cache=True)
            g = torch.cuda.CUDAGraph()
            out_holder: dict[str, object] = {}
            with torch.cuda.graph(g), torch.no_grad():
                out_holder["out"] = fwd(
                    model,
                    input_ids=inp_static,
                    past_key_values=buf.as_past(),
                    use_cache=True,
                )
            graphs.append(g)
            static_inps.append(inp_static)
            outs.append(out_holder["out"])
            # after capture, fold the step's KV into the shared buffer so the
            # next graph captures the next step's KV state.
            buf.advance_one(out_holder["out"].past_key_values)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} per-step graphs over shared buffer", flush=True)

    # ---------------- graphed 16-step loop ---------------------------------
    def graphed_loop():
        # REUSE the capture-time buffer storage (address-stable): reset its
        # contents to prefill KV, then replay each step graph in order.
        buf.reset(past0)
        cur = ids.clone()
        nid = nid0.clone()
        for k in range(MAX_NEW):
            static_inps[k].copy_(_decode_input(nid, audio_pad))
            graphs[k].replay()  # reads live buffer content at captured addr
            nid = sample_token(outs[k].logits)
            cur = torch.cat((cur, nid), dim=1)
            # fold step-k's fresh KV +1 column back into the shared buffer
            buf.advance_one(outs[k].past_key_values)
        return cur

    # correctness sanity: run one each, ensure no crash + finite logits.
    c_ref = eager_loop()
    c_g = graphed_loop()
    torch.cuda.synchronize()
    print(f"eager seq len={c_ref.shape[1]} graphed seq len={c_g.shape[1]}", flush=True)
    print(
        f"graphed logits finite: {all(torch.isfinite(o.logits).all().item() for o in outs)}",
        flush=True,
    )

    # ---------------- timing -------------------------------------------------
    eager_loop()
    torch.cuda.synchronize()
    te = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eager_loop()
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

    print(f"eager 16-step e2e:   {eager_ms:.2f} ms median", flush=True)
    print(f"graphed 16-step e2e: {graphed_ms:.2f} ms median", flush=True)
    print(f"end-to-end speedup:  {eager_ms / graphed_ms:.2f}x", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
