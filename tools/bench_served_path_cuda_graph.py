"""GPU seam verification: served path (runner->stage->generate_audio) actually
honors deploy.use_thinker_cuda_graph after the _thinker_stage fix (#66).

Before the fix, the stage constructed its own bundle without use_thinker_cuda_graph,
so the deploy yaml default (true) never reached served requests even though
the bench harness (which sets bundle.use_thinker_cuda_graph directly via OmniBase)
was fast. This script loads MinimindBundle via _thinker_stage on the 3050 and
asserts bundle.use_thinker_cuda_graph tracks deploy.use_thinker_cuda_graph for both
yaml-default-on and explicit-off cases.

Usage (WSL):
    PYTHONPATH=$PWD $VENV/bin/python tools/bench_served_path_cuda_graph.py
"""

import os
import sys
import time
import types

import yaml

from nanovllm_omni.config.registry import DeployConfig
from nanovllm_omni.models.minimind_omni import thinker as th

MODEL = os.environ.get("MINIMIND_MODEL", "/home/mcig/minimind-3o")
DEPLOY_YAML = os.environ.get(
    "DEPLOY_YAML", "/home/mcig/nanovllm-omni/nanovllm_omni/deploy/minimind_omni.yaml"
)

with open(DEPLOY_YAML) as f:
    cfg = yaml.safe_load(f)
deploy = DeployConfig(**{k: v for k, v in cfg.items() if k in DeployConfig.__dataclass_fields__})
print(f"deploy.use_thinker_cuda_graph = {deploy.use_thinker_cuda_graph}")

args = types.SimpleNamespace(
    model=MODEL, device="cuda", trust_remote_code=True, dtype=None, extra={}
)
captured: list = []
orig_create = th.load_minimind_omni_bundle


def fake_create_record(model_id, **kw):
    b = orig_create(model_id, **kw)
    captured.append(b)
    return b


try:
    t0 = time.time()
    th.load_minimind_omni_bundle = fake_create_record  # type: ignore[attr-defined]
    stage = th._thinker_stage(deploy, args)
    print(f"stage built in {time.time()-t0:.1f}s")
finally:
    th.load_minimind_omni_bundle = orig_create  # type: ignore[attr-defined]

if not captured:
    print("ERROR: no bundle captured")
    sys.exit(1)
b = captured[0]
print(f"bundle.use_thinker_cuda_graph = {b.use_thinker_cuda_graph}")
if b.use_thinker_cuda_graph is not True:
    print(f"FAIL: stage bundle use_thinker_cuda_graph={b.use_thinker_cuda_graph} (expected True)")
    sys.exit(1)
print("PASS: stage bundle honors deploy.use_thinker_cuda_graph=True")

# Explicit-off must propagate
deploy_off = DeployConfig(use_thinker_cuda_graph=False)
captured.clear()
try:
    th.load_minimind_omni_bundle = fake_create_record  # type: ignore[attr-defined]
    th._thinker_stage(deploy_off, args)
finally:
    th.load_minimind_omni_bundle = orig_create  # type: ignore[attr-defined]
b2 = captured[0]
print(
    f"deploy.use_thinker_cuda_graph=False -> bundle.use_thinker_cuda_graph = {b2.use_thinker_cuda_graph}"
)
if b2.use_thinker_cuda_graph is not False:
    print(f"FAIL: explicit False not honored (got {b2.use_thinker_cuda_graph})")
    sys.exit(1)
print("PASS: explicit False in deploy propagates to bundle")
