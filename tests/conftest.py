# Global test fixtures for nanovllm-omni tests.
#
# 1. Add the fork (third_party/nano-vllm) to sys.path so `import nanovllm`
#    resolves to the vendored fork rather than an installed package.
# 2. Mock torch.distributed before any fork layer import — fork layers
#    (LinearBase, VocabParallelEmbedding, Attention) call
#    dist.get_world_size() / dist.get_rank() at __init__ time.
# 3. Mock triton and flash_attn — fork's attention.py imports them at
#    module level; on macOS they aren't available.  We only need the
#    Attention __init__ signature (num_heads, head_dim, scale, num_kv_heads)
#    and a callable forward; the actual SDPA kernels are never invoked in
#    unit tests.
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# --- Fork path setup ---
_FORK_ROOT = str(Path(__file__).resolve().parent.parent / "third_party" / "nano-vllm")
if _FORK_ROOT not in sys.path:
    sys.path.insert(0, _FORK_ROOT)


def _ensure_stub(name: str) -> None:
    """Insert a stub module into sys.modules so fork imports succeed."""
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


# Stub triton + flash_attn + flash_attn CUDA kernels.
# Only stub modules that aren't already installed (triton is real on WSL).
for _mod in ("flash_attn", "flash_attn.flash_attn_varlen_func", "flash_attn.flash_attn_with_kvcache"):
    _ensure_stub(_mod)


@pytest.fixture(autouse=True)
def _mock_dist(monkeypatch):
    """Make fork layers importable and constructible without a real process group."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
