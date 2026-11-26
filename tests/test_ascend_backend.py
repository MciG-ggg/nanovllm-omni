"""Ascend backend contracts (no NPU / torch_npu needed).

The registry must stay importable without torch, and the Ascend module
must not load ``torch_npu`` at import time: this machine has neither.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from nanovllm_omni.backend import available_backends, get_backend

REPO = Path(__file__).resolve().parents[1]


def test_registry_lists_ascend() -> None:
    assert "ascend" in available_backends()


def test_ascend_capabilities() -> None:
    backend = get_backend("ascend")

    assert backend.name == "ascend"
    assert backend.device == "npu"
    assert backend.dtype == "bfloat16"
    assert backend.capabilities.varlen is True
    assert backend.capabilities.graph is False
    assert backend.capabilities.max_batch is None


def test_default_backend_is_torch_without_npu() -> None:
    assert get_backend().name == "torch"


def test_registry_import_does_not_load_torch_or_torch_npu() -> None:
    code = (
        "import sys\n"
        "import nanovllm_omni.backend\n"
        "print('torch' in sys.modules, 'torch_npu' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )

    assert result.stdout.split() == ["False", "False"]


def test_ascend_attention_matches_the_fork_interface() -> None:
    from nanovllm_omni.backend.ascend_attention import AscendAttention

    params = list(inspect.signature(AscendAttention.__init__).parameters)
    assert params[1:] == ["num_heads", "head_dim", "scale", "num_kv_heads"]

    attention = AscendAttention(4, 8, 0.125, 2)
    assert attention.num_heads == 4
    assert attention.num_kv_heads == 2
    assert attention.k_cache.numel() == 0
