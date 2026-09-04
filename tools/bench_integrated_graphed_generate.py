#!/usr/bin/env python3
"""Integrated CUDA Graph e2e decode probe (RTX 3050) — the §16 acceptance
experiment.

Real text-decode loop (forward + audio_buffer padding + host multinomial
sampling, KV grows 1 token/step) with the forward replaced by a per-step
CUDA Graph. Two timings:

  1. eager e2e  (592.83 ms anchor from §16 / bench_e2e_graphed_generate)
  2. graphed e2e — same loop, but step k replays graph_k captured at KV
     length k. Host sampling stays on the CPU as in the real generate.

Because KV grows each step, one graph per step length (1..16) is captured
against real step buffers, then the decode loop replays the matching graph.

Measures the actual end-to-end wall (incl. host sampling overlap) — the
question §16 left open: does 6-10x hold once sampling is in the loop?

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/bench_integrated_graphed_generate.py
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
REPEAT = 3


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

    # --- neutralize host-reads (proven capture-unblock, report §13) ---
    import inspect

    import nanovllm_omni.models.minimind_omni.attention as _attn_mod  # noqa: F401

    cls = type(model)
    src = inspect.getsource(cls.forward)
    src = src.replace("if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph").replace(
        "if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph"
    )
    ns = dict(cls.forward.__globals__)
    ns["__name__"] = cls.__module__
    ns["__qualname__"] = cls.__qualname__ + ".forward_graphed"
    import textwrap

    exec(compile(textwrap.dedent(src), "<probe>", "exec"), ns)
    fwd = ns["forward"]

    def make_input(cur, past_kvs):
        """Build the exact forward input stream_generate would pass."""
        if past_kvs is None:
            buf = torch.full((1, 8, cur.shape[1]), audio_pad, dtype=torch.long, device="cuda")
            return torch.cat((buf, cur.unsqueeze(1)), dim=1)
        buf = torch.full((1, 8, 1), audio_pad, dtype=torch.long, device="cuda")
        return torch.cat((buf, cur[:, -1:].unsqueeze(1)), dim=1)

    def sample(out, cur, temperature=0.75, top_p=0.9):
        logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
        logits[list(set(cur[0].tolist()))] /= 1.0
        if top_p and top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            mask = torch.cumsum(functional.softmax(sorted_l, dim=-1), dim=-1) > top_p
            mask[1:], mask[0] = mask[:-1].clone(), False
            logits[sorted_i[mask]] = float("-inf")
        return torch.multinomial(functional.softmax(logits, dim=-1), 1)

    # --- run one full decode pass (per-stp graph optional) ---
    def decode_pass(use_graph, graphs=None, step_bufs=None):
        cur = ids.clone()
        past_kvs = None
        for k in range(MAX_NEW):
            inp = make_input(cur, past_kvs)
            if use_graph:
                # Feed the real step input into the graph's static input
                # buffer, replay, then read the static output. past_kvs must
                # stay the SAME tensor object each step so graph_k's captured
                # KV buffers hold across replays.
                step_bufs[k]["inp"].copy_(inp)
                graphs[k].replay()
                out = step_bufs[k]["out"]
            else:
                with torch.no_grad():
                    out = fwd(
                        model,
                        input_ids=inp,
                        past_key_values=past_kvs,
                        use_cache=True,
                        audio_inputs=None,
                        audio_lens=None,
                    )
            past_kvs = out.past_key_values
            tok = sample(out, cur)
            cur = torch.cat((cur, tok.unsqueeze(1)), dim=1)
        return cur

    # --- eager timing ---
    decode_pass(use_graph=False)
    torch.cuda.synchronize()
    t_e = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_pass(use_graph=False)
        torch.cuda.synchronize()
        t_e.append((time.perf_counter() - t0) * 1000)
    eager_ms = sorted(t_e)[len(t_e) // 2]
    print(f"eager e2e (16 steps + sampling): {eager_ms:.2f} ms", flush=True)

    # --- capture per-step graphs against real step shapes ---
    # First pass over real shapes to know each step's buffers, capturing a
    # graph per step on a side stream.
    cur = ids.clone()
    past_kvs = None
    graphs = []
    step_bufs = []
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _k in range(MAX_NEW):
            inp = make_input(cur, past_kvs)
            # warm this shape's forward first (capture needs prior warmup in graph mode)
            with torch.no_grad():
                fwd(
                    model,
                    input_ids=inp,
                    past_key_values=past_kvs,
                    use_cache=True,
                    audio_inputs=None,
                    audio_lens=None,
                )
            # capture graph_k against a STATIC input buffer + output holder.
            # The input buffer is persisted so decode can copy_() the real
            # step input into it before each replay (the fix for the
            # device-side assert: static-graph replay without input feeding
            # indexes logits against stale token ids).
            inp_static = inp.clone()
            g = torch.cuda.CUDAGraph()
            out_holder: dict[str, object] = {}
            with torch.cuda.graph(g), torch.no_grad():
                out_holder["out"] = fwd(
                    model,
                    input_ids=inp_static,
                    past_key_values=past_kvs,
                    use_cache=True,
                    audio_inputs=None,
                    audio_lens=None,
                )
            graphs.append(g)
            step_bufs.append({"inp": inp_static, "out": out_holder["out"]})
            # advance past_kvs to next step's shape using eager forward on the
            # captured-out (mirrors real progression)
            past_kvs = out_holder["out"].past_key_values
            tok = sample(out_holder["out"], cur)
            cur = torch.cat((cur, tok.unsqueeze(1)), dim=1)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} per-step graphs", flush=True)

    # --- graphed e2e timing (uses fresh buffers captured in graph) ---
    torch.cuda.synchronize()
    t_g = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_pass(use_graph=True, graphs=graphs, step_bufs=step_bufs)
        torch.cuda.synchronize()
        t_g.append((time.perf_counter() - t0) * 1000)
    graphed_ms = sorted(t_g)[len(t_g) // 2]
    print(f"graphed e2e (16 steps + sampling): {graphed_ms:.2f} ms", flush=True)
    print(
        f"end-to-end speedup: {eager_ms / graphed_ms:.2f}x  (delta {(eager_ms - graphed_ms) / eager_ms * 100:+.1f}%)",
        flush=True,
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
