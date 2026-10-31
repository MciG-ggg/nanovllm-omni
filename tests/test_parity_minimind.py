"""MiniMind-O parity vs vendor stream_generate (docs/adr/0011-parity-standard.md P1)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("MINIMIND_PARITY_MODEL_DIR", "/home/mcig/minimind-3o")
MIMI = os.environ.get("MINIMIND_PARITY_MIMI_DIR", "/home/mcig/mimi")


@pytest.mark.smoke
def test_minimind_matches_vendor() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not os.path.isdir(MODEL) or not os.path.isdir(MIMI):
        pytest.skip(f"MiniMind/Mimi weights not found: {MODEL}, {MIMI}")

    env = dict(
        os.environ,
        MINIMIND_PARITY_MODEL_DIR=MODEL,
        MINIMIND_PARITY_MIMI_DIR=MIMI,
        HF_HUB_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, "tests/test_minimind_hf_parity.py"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "PASS: parity ratio" in result.stdout, result.stdout[-4000:]
