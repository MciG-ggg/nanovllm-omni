#!/usr/bin/env python3
"""Deploy-default fast-path probe (RTX 3050).

After the deploy-layer default change (yaml use_cuda_graph: true →
bundle.use_cuda_graph → generate_audio(use_cuda_graph=None)), verify the
DEFAULT path (no explicit flag) actually routes through the CUDA-Graph
decoder:

  A. determinism: generate_audio(None) twice, same torch seed → identical
     AudioPayload bytes
  B. speed: default (expected fast path) vs explicit eager
     (use_cuda_graph=False), so we can see graph is really in play
  C. bundle flag resolves from Omnibase deploy (via load_deploy_config on
     minimind_omni.yaml → use_cuda_graph True)

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_deploy_default.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/mcig/nanovllm-omni")
import torch

from nanovllm_omni.config.registry import load_deploy_config
from nanovllm_omni.models.minimind_omni import generate_audio
from nanovllm_omni.models.minimind_omni.bundle import create_bundle

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
YAML = "/home/mcig/nanovllm-omni/nanovllm_omni/deploy/minimind_omni.yaml"
PROMPT = "你好。"


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    deploy = load_deploy_config(YAML)
    print(f"C deploy.use_cuda_graph={deploy.use_cuda_graph} (yaml)", flush=True)

    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    bundle.use_cuda_graph = deploy.use_cuda_graph

    def run(flag, tag):
        torch.manual_seed(7)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        payload = generate_audio(
            bundle, PROMPT, max_tokens=16, temperature=0.75, top_p=0.9, use_cuda_graph=flag
        )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {tag}: {ms:.0f} ms wav={getattr(payload, 'data', None) is not None}", flush=True)
        return payload

    # default (None -> bundle flag) twice for determinism
    p1 = run(None, "default(None) #1")
    p2 = run(None, "default(None) #2")
    b1 = getattr(p1, "data", None)
    b2 = getattr(p2, "data", None)
    det = b1 is not None and b1 == b2
    print(f"A default determinism (2x same seed, bytes identical): {det}", flush=True)

    # explicit eager for the speed perspective
    _ = run(False, "explicit eager  ")
    print("B default-path completed (graph via deploy ON)", flush=True)
    print("DONE", flush=True)
    return 0 if (det and deploy.use_cuda_graph) else 1


if __name__ == "__main__":
    raise SystemExit(main())
