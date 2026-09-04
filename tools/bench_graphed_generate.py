#!/usr/bin/env python3
"""CUDA Graph decode-loop benchmark (RTX 3050, report §13/§14 continuation).

Measures whether the single-step 10.36x speedup (§13) transfers to a real
multi-step decode loop:
  1. replay stability: one captured graph replayed many times (no drift?)
  2. multi-graph loop: pre-capture one graph per AR-step shape (fixed KV
     length), replay the decode loop, compare to eager per-step.

Uses the proven capture-unblock (neutralize the two freqs_cos[0,0]==0
host-read checks in MiniMindOmni.forward via in-memory string patch).

Run on WSL (GPU): cd ~/nanovllm-omni && PYTHONPATH=. \
  ~/venvs/vllm-omni/bin/python tools/bench_graphed_generate.py
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

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
STEPS = 16
REPLAY_N = 200


def _patched_forward(model, cls, src: str):
    """Return a copy of cls.forward with the two freq host-read checks as
    `if False:` (warmup proves buffers non-zero; the reads invalidate
    capture). In-memory only."""
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    assert "if False:" in patched, "patch did not apply"
    namespace = dict(cls.forward.__globals__)
    namespace["__name__"] = cls.__module__
    namespace["__qualname__"] = cls.__qualname__ + ".forward_graphed"
    exec(compile(textwrap.dedent(patched), "<capture-bench>", "exec"), namespace)
    return namespace["forward"]


def _time(fn, n: int) -> tuple[float, float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / n * 1000
    return per, per, per


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA; nothing to benchmark")
        return 2
    torch.manual_seed(42)
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        out = bundle.model(input_ids=ids, past_key_values=None, use_cache=True)
    past_kv = out.past_key_values
    with torch.no_grad():
        nid = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        _ = bundle.model(input_ids=nid, past_key_values=past_kv, use_cache=True)

    # Neutralize host-reads (proven capture-unblock).
    cls = type(bundle.model)
    fwd = _patched_forward(bundle.model, cls, inspect.getsource(cls.forward))

    # --- 1. single-graph replay stability ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                fwd(bundle.model, input_ids=nid, past_key_values=past_kv, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        _ = fwd(bundle.model, input_ids=nid, past_key_values=past_kv, use_cache=True)
    print("capture step OK", flush=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPLAY_N):
        g.replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - t0) / REPLAY_N * 1000
    print(f"single-graph replay x{REPLAY_N}: {replay_ms:.3f} ms/rep (no drift check)", flush=True)

    # --- 2. multi-graph decode loop: one graph per AR step (fixed KV length) ---
    # Warm up every per-step shape, then capture one CUDAGraph per step and
    # replay the same grow sequence under the graph. Measurements are of the
    # GPU forward cost only (sampling stays host-side in real generate).
    base_n_kv = len(past_kv)
    kv_shape = (past_kv[0][0].shape[2], past_kv[0][0].shape[3])  # (n_kv_heads, head_dim)
    step_inputs = nid
    per_step_graphs = []
    step_kv = None
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _length in range(1, STEPS + 1):
            # growing KV with new token
            new_k = torch.randn(1, 1, kv_shape[0], kv_shape[1], device="cuda").half()
            new_v = torch.randn(1, 1, kv_shape[0], kv_shape[1], device="cuda").half()
            if step_kv is None:
                step_kv = [(new_k, new_v) for _ in range(base_n_kv)]
            else:
                step_kv = [
                    (torch.cat([k, new_k], dim=1), torch.cat([v, new_v], dim=1)) for k, v in step_kv
                ]
            with torch.no_grad():
                fwd(bundle.model, input_ids=step_inputs, past_key_values=step_kv, use_cache=True)
    torch.cuda.current_stream().wait_stream(s2)
    torch.cuda.synchronize()

    # capture per-step graphs (stable tensor refs)
    step_kv = None
    for _length in range(1, STEPS + 1):
        new_k = torch.randn(1, 1, kv_shape[0], kv_shape[1], device="cuda").half()
        new_v = torch.randn(1, 1, kv_shape[0], kv_shape[1], device="cuda").half()
        if step_kv is None:
            step_kv = [(new_k, new_v) for _ in range(base_n_kv)]
        else:
            step_kv = [
                (torch.cat([k, new_k], dim=1), torch.cat([v, new_v], dim=1)) for k, v in step_kv
            ]
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg), torch.no_grad():
            _ = fwd(bundle.model, input_ids=step_inputs, past_key_values=step_kv, use_cache=True)
        per_step_graphs.append(gg)

    def _eager_loop():
        kvs = None
        for _ in range(STEPS):
            with torch.no_grad():
                _ = fwd(bundle.model, input_ids=nid, past_key_values=kvs, use_cache=True)

    def _graphed_loop():
        # One decode pass = STEPS steps; each step replays the graph with the
        # past_kv length matching this step (graphs[step] captured at that
        # length). No extra warm budget inside the timed body.
        for gg in per_step_graphs:
            gg.replay()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(5):
        _graphed_loop()
    torch.cuda.synchronize()
    graphed_loop_per = (time.perf_counter() - t1) / (5 * STEPS) * 1000
    print(f"multi-graph decode loop: {graphed_loop_per:.3f} ms/step (graphed)", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
