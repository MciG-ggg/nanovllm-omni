"""Shared env helpers for the headline bench scripts.

``git_commit`` and ``gpu_label`` tag each CSV row with hardware + source
identity so the README headline tables can be regenerated across hosts
and re-traced to a specific commit. Both are best-effort: a CPU-only
host or a non-git checkout returns a sentinel rather than raising.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit() -> str:
    """Best-effort short SHA of the project root, or ``"unknown"`` if git is absent."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def gpu_label() -> str:
    """``"<name> (<mem> GB, driver <ver>)"`` or ``"cpu"`` when no accelerator is present."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        try:
            driver = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
                .splitlines()[0]
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError, IndexError):
            driver = "?"
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{name} ({mem_gb:.0f} GB, driver {driver})"
    try:
        # ``torch.npu`` exists only after torch_npu registers it.
        import torch_npu  # noqa: F401
    except ImportError:
        return "cpu"
    if not torch.npu.is_available():
        return "cpu"
    name = torch.npu.get_device_name(0)
    mem_gb = torch.npu.get_device_properties(0).total_memory / (1024**3)
    return f"npu: {name} ({mem_gb:.0f} GB, driver {_npu_driver()})"


def _npu_driver() -> str:
    """Best-effort version line from ``npu-smi info``, or ``"?"``."""
    try:
        out = subprocess.check_output(["npu-smi", "info"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "?"
    for line in out.decode("utf-8", "replace").splitlines():
        if "Version" in line:
            return line.split(":", 1)[-1].strip()
    return "?"
