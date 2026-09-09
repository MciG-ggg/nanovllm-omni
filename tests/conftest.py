# Global test fixtures for nanovllm-omni tests.
#
# 1. Mock torch.distributed before any fork layer import — fork layers
#    (LinearBase, VocabParallelEmbedding, Attention) call
#    dist.get_world_size() / dist.get_rank() at __init__ time.
# 2. Mock triton and flash_attn — fork's attention.py imports them at
#    module level; on macOS they aren't available.  We only need the
#    Attention __init__ signature (num_heads, head_dim, scale, num_kv_heads)
#    and a callable forward; the actual SDPA kernels are never invoked in
#    unit tests.
#
# nano-vllm is installed as an editable package via uv, so no sys.path
# manipulation is needed.
from __future__ import annotations

import os
import sys
import types

import pytest

_STUB_DIR = os.path.join(os.path.dirname(__file__), "_stub_modules")


def _ensure_stub(name: str) -> types.ModuleType:
    """Insert a stub package into sys.modules so fork + torch imports succeed.

    Uses a real directory as ``__path__`` so Python treats it as a package
    and can resolve sub-imports like ``triton.backends.compiler``.
    """
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [_STUB_DIR]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


# Stub triton + flash_attn + flash_attn CUDA kernels.
# On macOS these aren't installed; stubs let the fork's layer modules
# import without error.  torch._dynamo/inductor also access
# triton.language.dtype and triton.backends.compiler at import time.
for _mod in (
    "triton",
    "triton.language",
    "triton.backends",
    "triton.backends.compiler",
    "flash_attn",
    "flash_attn.flash_attn_varlen_func",
    "flash_attn.flash_attn_with_kvcache",
):
    _ensure_stub(_mod)

# Wire sub-module attributes so attribute access resolves.
_triton = sys.modules["triton"]
_triton.language = sys.modules["triton.language"]
_triton.language.dtype = type("dtype", (), {})
_triton.backends = sys.modules["triton.backends"]
_triton.backends.compiler = sys.modules["triton.backends.compiler"]


@pytest.fixture(autouse=True)
def _mock_dist(monkeypatch):
    """Make fork layers importable and constructible without a real process group."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
