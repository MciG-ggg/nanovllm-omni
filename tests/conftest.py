# Global test fixtures for nanovllm-omni tests.
#
# Load order matters: stubs must be in sys.modules BEFORE any fork
# module is imported, because fork modules do top-level `import triton`
# and `@torch.compile` at class-definition time.
#
# nano-vllm is installed as an editable package via uv, so no sys.path
# manipulation is needed.
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# 1. Stub triton (CUDA-only, not installed on macOS).
#    Must happen before ANY fork or torch._inductor import.
# ---------------------------------------------------------------------------
_triton = types.ModuleType("triton")
_triton.__path__ = []  # mark as package so sub-imports resolve
_triton.language = types.ModuleType("triton.language")
_triton.language.dtype = type("dtype", (), {})
_triton.compiler = types.ModuleType("triton.compiler")
_triton.compiler.compiler = types.ModuleType("triton.compiler.compiler")
_triton.backends = types.ModuleType("triton.backends")
_triton.backends.compiler = types.ModuleType("triton.backends.compiler")
_triton.Config = type("Config", (), {})  # torch._inductor.runtime.triton_compat
_triton.jit = lambda *a, **kw: (lambda fn: fn)  # @triton.jit decorator
_triton.autotune = lambda *a, **kw: (lambda fn: fn)  # @triton.autotune decorator
_triton.heuristics = lambda *a, **kw: (lambda fn: fn)
_triton.language.constexpr = lambda *a, **kw: (lambda fn: fn)
_triton.language.autotune = _triton.autotune
_triton.language.heuristics = _triton.heuristics

for _name, _mod in [
    ("triton", _triton),
    ("triton.language", _triton.language),
    ("triton.compiler", _triton.compiler),
    ("triton.compiler.compiler", _triton.compiler.compiler),
    ("triton.backends", _triton.backends),
    ("triton.backends.compiler", _triton.backends.compiler),
]:
    sys.modules.setdefault(_name, _mod)

# ---------------------------------------------------------------------------
# 2. Stub flash_attn (CUDA-only).
# ---------------------------------------------------------------------------
for _mod in (
    "flash_attn",
    "flash_attn.flash_attn_varlen_func",
    "flash_attn.flash_attn_with_kvcache",
):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# ---------------------------------------------------------------------------
# 3. Mock torch.compile as a no-op BEFORE fork activation.py triggers it.
#    activation.py has `@torch.compile` on SiluAndMul at class-definition
#    time, which would pull in torch._inductor → triton. With the stubs
#    above this would technically succeed, but a no-op is cleaner for
#    unit tests (no GPU, no CUDA graph capture).
# ---------------------------------------------------------------------------
import torch  # noqa: E402

_orig_compile = torch.compile


def _no_compile(*args, **kwargs):
    if args and len(args) == 1 and callable(args[0]):
        return args[0]

    def decorator(fn):
        return fn

    return decorator


torch.compile = _no_compile

# ---------------------------------------------------------------------------
# 4. pytest fixtures
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_dist(monkeypatch):
    """Make fork layers constructible without a real process group."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
