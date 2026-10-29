"""Profile a model family on the Colab GPU session with nsys or ncu.

The script writes the target to ``/content/profile_target.py`` and runs
it under the profiler. The knobs come from the environment, because
``colab exec`` passes them with ``--env``:

- ``PROFILE_TOOL``: ``nsys`` (default) or ``ncu``
- ``PROFILE_TARGET``: ``minimind_o`` (default), ``smolvla``, ``smolvla_flow``
- ``NCU_LAUNCH_COUNT`` / ``NCU_LAUNCH_SKIP``: the ncu launch window

The capture filters on the kernel name, not on an NVTX range. The repo
emits its stage markers as NVTX push/pop ranges, and the ncu NVTX
filter only matches start/end ranges (see ADR-0003). The eager path is
the default target because a CUDA-graph replay hides the per-kernel
attribution.

Usage:

    colab exec -s omni-t4 --timeout 3600 \
        --env PROFILE_TOOL=nsys --env PROFILE_TARGET=minimind_o \
        -f tools/profile_session.py
"""

import glob
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

TOOL = os.environ.get("PROFILE_TOOL", "nsys")
TARGET = os.environ.get("PROFILE_TARGET", "minimind_o")
LAUNCH_COUNT = os.environ.get("NCU_LAUNCH_COUNT", "5")
LAUNCH_SKIP = os.environ.get("NCU_LAUNCH_SKIP", "0")
REPORT = f"/content/{TOOL}_{TARGET}"
TARGET_PATH = "/content/profile_target.py"

KERNEL_REGEX = {
    "minimind_o": "regex:sgemm|gemm|hmma|volta|gemv",
    "smolvla": "regex:sgemm|gemm|hmma|volta|gemv",
    "smolvla_flow": "regex:sgemm|gemm|hmma|volta|gemv",
}.get(TARGET, "regex:.*")

OMNI_TARGET = """
import sys
import time

sys.path.insert(0, "/content/nanovllm-omni")
sys.path.insert(0, "/content/nanovllm-omni/third_party/nano-vllm")

import torch

from nanovllm_omni import Omni, SamplingParams

HUB = "/root/.cache/huggingface/hub"


def local_dir(repo_id):
    import os

    folder = os.path.join(HUB, "models--" + repo_id.replace("/", "--"), "snapshots")
    return os.path.join(folder, sorted(os.listdir(folder))[-1])


WHICH = "{target}"

if WHICH == "smolvla":
    from nanovllm_omni.config.params import OmniEngineArgs
    from nanovllm_omni.config.registry import (
        DeployConfig,
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.engine.runner import PipelineRunner

    FAMILY = "nanovllm_omni.models.smolvla"
    pipeline = PipelineConfig(
        name="smolvla_test",
        stages=(
            StageConfig(
                stage_id=0,
                name="vlm",
                kind=StageExecutionType.LLM_AR,
                stage_factory=(FAMILY + ".vlm_stage", "_vlm_stage"),
                process_input=None,
                input_sources=(),
                is_terminal=False,
            ),
            StageConfig(
                stage_id=1,
                name="action",
                kind=StageExecutionType.DIFFUSION,
                stage_factory=(FAMILY + ".action_stage", "_action_stage"),
                process_input=(FAMILY + ".stage_processors", "vlm2action"),
                input_sources=(0,),
                is_terminal=True,
                final_output_type="actions",
            ),
        ),
        default_deploy_config_name="smolvla.yaml",
    )
    args = OmniEngineArgs(
        model=local_dir("HuggingFaceTB/SmolVLM-500M-Instruct"),
        dtype="bfloat16",
        device="cuda",
        extra={{"allow_hf_download": False}},
    )
    runner = PipelineRunner(pipeline, DeployConfig(), args)
    sampling = SamplingParams(
        extra={{
            "image": torch.randn(3, 224, 224).byte(),
            "state": __import__("numpy").random.randn(7).astype("float32"),
            "chunk_len": 10,
            "action_dim": 7,
            "num_inference_steps": 10,
        }},
    )
    t0 = time.perf_counter()
    runner.run_payload("pick up the red block", sampling)
    torch.cuda.synchronize()
    print("PROFILE_TARGET wall_ms=%.1f" % ((time.perf_counter() - t0) * 1000.0))
    print("PROFILE_TARGET_DONE")
    raise SystemExit(0)

if WHICH == "minimind_o":
    omni = Omni(
        model="jingyaogong/minimind-3o",
        device="cuda",
        extra={{"use_thinker_cuda_graph": False}},
    )
    sampling = SamplingParams(max_tokens=120, temperature=0.7, top_p=0.9)
    prompt = "Hello, how are you?"
else:
    omni = Omni(
        model=local_dir("HuggingFaceTB/SmolVLM-500M-Instruct"),
        device="cuda",
        dtype="bfloat16",
        extra={{"use_cuda_graph": False}},
    )
    sampling = SamplingParams(max_tokens=64, temperature=0.0)
    prompt = "Hello, how are you?"

t0 = time.perf_counter()
omni.generate(prompt, sampling)
torch.cuda.synchronize()
print("PROFILE_TARGET wall_ms=%.1f" % ((time.perf_counter() - t0) * 1000.0))
print("PROFILE_TARGET_DONE")
"""

FLOW_TARGET = """
import runpy
import sys

sys.path.insert(0, "/content/nanovllm-omni")
sys.path.insert(0, "/content/nanovllm-omni/third_party/nano-vllm")

sys.argv = ["bench_smolvla_flow", "--repeats", "20", "--steps", "10"]
runpy.run_module("nanovllm_omni.bench.bench_smolvla_flow", run_name="__main__")
"""


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        hits = [hit for hit in hits if os.access(hit, os.X_OK)]
        if hits:
            return hits[0]
    return None


def write_target() -> None:
    source = FLOW_TARGET if TARGET == "smolvla_flow" else OMNI_TARGET.format(target=TARGET)
    Path(TARGET_PATH).write_text(source, encoding="utf-8")
    print("target:", TARGET_PATH, flush=True)


def run_nsys() -> int:
    nsys = find_tool("nsys")
    if nsys is None:
        print("NSYS_UNAVAILABLE")
        return 1
    command = [
        nsys,
        "profile",
        "-o",
        REPORT,
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        sys.executable,
        TARGET_PATH,
    ]
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout[-4000:], flush=True)
    print(result.stderr[-3000:], flush=True)

    for name in ("cuda_gpu_kern_sum", "nvtx_sum"):
        stats = subprocess.run(
            [nsys, "stats", "--report", name, REPORT + ".nsys-rep"],
            capture_output=True,
            text=True,
        )
        print(f"=== {name} ===", flush=True)
        print(stats.stdout[-6000:], flush=True)
    return 0


def run_ncu() -> int:
    ncu = find_tool("ncu")
    if ncu is None:
        print("NCU_UNAVAILABLE")
        return 1
    command = [
        ncu,
        "--set",
        "full",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        KERNEL_REGEX,
        "--launch-skip",
        LAUNCH_SKIP,
        "--launch-count",
        LAUNCH_COUNT,
        "-o",
        REPORT,
        "--force-overwrite",
        sys.executable,
        TARGET_PATH,
    ]
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout[-6000:], flush=True)
    print(result.stderr[-3000:], flush=True)

    report = REPORT + ".ncu-rep"
    if not os.path.exists(report):
        print("no ncu report", flush=True)
        return 0
    print(f"report: {report} ({os.path.getsize(report) / 1e6:.1f} MB)", flush=True)
    details = subprocess.run(
        [ncu, "--import", report, "--page", "details", "--csv"],
        capture_output=True,
        text=True,
    )
    print("=== ncu details (csv) ===", flush=True)
    print(details.stdout[-20000:], flush=True)
    return 0


write_target()
if TOOL == "ncu":
    raise SystemExit(run_ncu())
raise SystemExit(run_nsys())
