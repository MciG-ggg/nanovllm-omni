"""The base install must not pull an array framework.

The three tracks (base / cuda / mlx) are separate extras because their
dependencies cannot share one environment: ``mlx-vlm`` needs
``datasets>=4`` while ``lerobot`` pins ``datasets<=3.6``. That split only
holds if the base package really is framework-free — one stray
``import torch`` at module scope and ``pip install -e .`` starts needing a
CUDA-toolkit-sized download just to read the config.

``tests/test_backend_wiring.py`` pins this for ``nanovllm_omni.backend``;
this file pins it for the package root and the pieces the base install
advertises (scheduler, config registry, deploy parsing).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each entry is imported in its own subprocess so one bad module cannot hide
# behind another's already-imported framework.
BASE_MODULES = (
    "nanovllm_omni",
    "nanovllm_omni.backend",
    "nanovllm_omni.config",
    "nanovllm_omni.config.params",
    "nanovllm_omni.config.registry",
)


def _frameworks_loaded(module: str) -> tuple[bool, bool]:
    code = (
        "import sys\n" f"import {module}\n" "print('torch' in sys.modules, 'mlx' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    torch_loaded, mlx_loaded = result.stdout.split()
    return torch_loaded == "True", mlx_loaded == "True"


def test_importing_the_package_root_loads_no_framework() -> None:
    assert _frameworks_loaded("nanovllm_omni") == (False, False)


def test_the_config_layer_loads_no_framework() -> None:
    """Deploy parsing and pipeline registry are read on every host."""
    assert _frameworks_loaded("nanovllm_omni.config.registry") == (False, False)


def test_the_base_declares_no_framework() -> None:
    """``dependencies`` must stay framework-free; the extras carry them."""
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base = " ".join(project["dependencies"]).lower()

    for framework in ("torch", "triton", "flash-attn", "mlx"):
        assert framework not in base, f"{framework!r} leaked into the base dependencies"

    extras = project["optional-dependencies"]
    assert "cuda" in extras and "mlx" in extras, "the tracks must be separate extras"
    assert any("triton" in dep for dep in extras["cuda"])


def test_the_mlx_extra_carries_every_mlx_library() -> None:
    """One extra for MLX: the three libraries coexist, so splitting is a trap."""
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    mlx = " ".join(project["optional-dependencies"]["mlx"])

    for package in ("mlx>=", "mlx-lm>=", "mlx-vlm>=", "torchvision>="):
        assert package in mlx, f"{package!r} is missing from the mlx extra"

    assert "mlx-vlm" not in project["optional-dependencies"], (
        "mlx-vlm must not be its own extra: installing `mlx` alone would then "
        "leave SmolVLM failing with ModuleNotFoundError"
    )


def test_the_cuda_track_is_marked_off_macos() -> None:
    """triton has no macOS wheel; the marker keeps a Mac install working."""
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    cuda = project["optional-dependencies"]["cuda"]

    for package in ("triton", "flash-attn"):
        entries = [dep for dep in cuda if dep.split(";")[0].strip().startswith(package)]
        assert entries, f"{package} is missing from the cuda extra"
        assert all(
            "sys_platform != 'darwin'" in dep for dep in entries
        ), f"{package} needs the darwin marker or `pip install .[cuda]` fails on a Mac: {entries}"
