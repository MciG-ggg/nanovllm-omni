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
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def gpu_label() -> str:
    """``"<name> (<mem> GB, driver <ver>)"`` or ``"cpu"`` when CUDA isn't present."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
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
