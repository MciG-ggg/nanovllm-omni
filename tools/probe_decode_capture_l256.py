#!/usr/bin/env python3
"""Decode-step CUDA Graph capture probe with L256 host-read INTACT.

#13/chapter §13 neutralized two freqs_cos[0,0]==0 host-reads to unblock
capture. But MiniMindOmni.forward line 256 has a SECOND host read that
reads REAL data during decode:
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
During a decode step past_kvs is not None, so start_pos is a host read of
the KV tensor's .shape — which CUDA Graph forbids during capture.

This probe answers: does capturing a DECODE step (past_kv non-None) fail
on L256 even after the freqs host-reads are neutralized? If yes, the
production integration (plan §3 strategy A) must ALSO externalize L256's
start_pos host-read — a concrete addition to the capture-blocker list in
§12/§17 that was only flagged speculatively.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=. \\
  ~/venvs/vllm-omni/bin/python tools/probe_decode_capture_l256.py
"""

from __future__ import annotations

import inspect
import sys
import textwrap

import torch

sys.path.insert(0, "/home/mcig/nanovllm-omni")
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"


def _patched_forward(model, cls, src: str):
    """Neutralize ONLY the two freqs host-reads; leave L256 start_pos read."""
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed"
    ).replace("if self.talker.freqs_cos[0, 0] == 0:", "if False:  # CUDA-Graph: warmup precomputed")
    assert "if False:" in patched, "patch did not apply"
    assert (
        "start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0"
        in patched
    )
    ns = dict(cls.forward.__globals__)
    ns["__name__"] = cls.__module__
    ns["__qualname__"] = cls.__qualname__ + ".forward_l256"
    exec(compile(textwrap.dedent(patched), "<capture-l256>", "exec"), ns)
    return ns["forward"]


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
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
    torch.cuda.synchronize()

    cls = type(bundle.model)
    fwd = _patched_forward(bundle.model, cls, inspect.getsource(cls.forward))

    # Warmup decode path (side stream, L256 reads real past_kv shape).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        fwd(bundle.model, input_ids=nid, past_key_values=past_kv, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            _ = fwd(bundle.model, input_ids=nid, past_key_values=past_kv, use_cache=True)
        print("DECODE-STEP CAPTURE OK with L256 host-read intact", flush=True)
        print("result: L256 start_pos .shape read does NOT block capture", flush=True)
        return 0
    except torch.cuda.CUDAGraphError as exc:  # noqa: PERF203
        print("DECODE-STEP CAPTURE FAILED with L256 host-read intact", flush=True)
        print("failure:", type(exc).__name__, str(exc)[:400], flush=True)
        print("result: L256 start_pos .shape read BLOCKS capture — production", flush=True)
        print("integration must externalize it (add to capture-blocker list §17)", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
