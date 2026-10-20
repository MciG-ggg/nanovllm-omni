"""Prepare a Colab GPU session for the four model-family measurements.

One command does everything:

1. clone the repository and the fork
2. build one Python environment for all four families
3. install the profiler CLIs (nsys, ncu)
4. pre-download the checkpoints
5. report what is ready

The environment pins transformers to 5.5.4, the version that
``lerobot[smolvla]`` requires. The Mimi codec that MiniMind-O needs
exists in that version, so one environment serves all four families.
The torch stack keeps the version that the Colab image ships, so the
CUDA build stays intact.

Usage:

    colab exec -s omni-t4 --timeout 3600 -f tools/colab_setup.py
"""

import importlib.metadata as metadata
import shutil
import site
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/MciG-ggg/nanovllm-omni.git"
FORK = "https://github.com/MciG-ggg/nano-vllm.git"
BRANCH = "perf/colab-2x"
DEST = Path("/content/nanovllm-omni")

PACKAGES = [
    "lerobot[smolvla]==0.6.1",
    "av>=12",
    "diffusers>=0.31",
    "accelerate>=0.34",
    "safetensors",
    "huggingface_hub>=0.20",
    "pyyaml",
    "sentencepiece",
    "protobuf",
    "einops",
    "timm",
    "Pillow",
    "pytest",
    "nvidia-nsight-systems-cu12",
    "nvidia-nsight-compute-cu12",
]

CHECKPOINTS = [
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "stabilityai/sd-turbo",
    "jingyaogong/minimind-3o",
    "kyutai/mimi",
    "HuggingFaceVLA/smolvla_libero",
]

VERIFY = """
import importlib
import sys

sys.path.insert(0, "/content/nanovllm-omni")
sys.path.insert(0, "/content/nanovllm-omni/third_party/nano-vllm")

import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))

import transformers

print("transformers", transformers.__version__)

for label, module in [
    ("mimi", "transformers.models.mimi.modeling_mimi"),
    ("smolvla_policy", "lerobot.policies.smolvla.modeling_smolvla"),
    ("diffusers", "diffusers"),
    ("av", "av"),
]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"{label} FAILED {type(exc).__name__}: {exc}")
    else:
        print(f"{label} ok")

from nanovllm_omni.models.minimind_omni.hf_config import register_minimind_config

register_minimind_config()
print("minimind_config ok")
"""


def run(args: list[str], **kwargs) -> int:
    print("$", " ".join(args), flush=True)
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    print(result.stdout[-2000:], flush=True)
    print(result.stderr[-2000:], flush=True)
    return result.returncode


def clone() -> None:
    if (DEST / ".git").is_dir():
        print("repo already cloned", flush=True)
        return
    run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO, str(DEST)])
    run(
        [
            "git",
            "clone",
            "--branch",
            BRANCH,
            "--depth",
            "1",
            FORK,
            str(DEST / "third_party" / "nano-vllm"),
        ]
    )
    run(["git", "-C", str(DEST), "rev-parse", "HEAD"])
    run(["git", "-C", str(DEST / "third_party" / "nano-vllm"), "rev-parse", "HEAD"])


def constraint_file() -> Path:
    """Pin the installed torch stack so pip keeps the CUDA build."""
    lines = []
    for name in ("torch", "torchvision", "torchaudio"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        lines.append(f"{name}=={version.split('+')[0]}")
    path = Path("/content/constraint.txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("constraint:", lines, flush=True)
    return path


def clean_transformers() -> None:
    """Remove the image transformers before the pinned install.

    The image ships transformers 5.15.0 and the install replaces it with
    5.5.4. pip uninstalls the old version first, and the old RECORD
    then deletes files that the new version already wrote. The result is
    a package directory that mixes both versions.
    """
    for root in site.getsitepackages():
        path = Path(root) / "transformers"
        if path.is_dir():
            print("removing:", path, flush=True)
            shutil.rmtree(path)
    run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"])


def install(constraint: Path) -> None:
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--constraint",
        str(constraint),
        *PACKAGES,
    ]
    if run(args) != 0:
        print("constrained install failed; retrying without the constraint", flush=True)
        run([sys.executable, "-m", "pip", "install", "--quiet", *PACKAGES])


def editable_installs() -> None:
    """Install the fork and the repository without touching dependencies."""
    fork = str(DEST / "third_party" / "nano-vllm")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", fork])
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(DEST)])


def drop_torchao() -> None:
    """Remove the preinstalled torchao.

    diffusers imports ``FqnToConfig`` from torchao whenever torchao is
    importable, and the image's torchao does not have that name. The
    SD-Turbo stage then fails to import.
    """
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])


def fetch_checkpoints() -> None:
    from huggingface_hub import snapshot_download

    for repo in CHECKPOINTS:
        print(f"=== {repo} ===", flush=True)
        try:
            path = snapshot_download(repo, ignore_patterns=["*.pth", "*.onnx", "*.msgpack", "*.h5"])
            print(f"OK {repo} -> {path}", flush=True)
        except Exception as exc:
            print(f"FAIL {repo}: {exc}", flush=True)


def verify() -> None:
    run([sys.executable, "-c", VERIFY])


clone()
constraint = constraint_file()
clean_transformers()
install(constraint)
drop_torchao()
editable_installs()
fetch_checkpoints()
verify()
print("GPU_SETUP_DONE", flush=True)
