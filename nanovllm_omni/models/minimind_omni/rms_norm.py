"""Fused RMSNorm monkey-patch for MiniMind-O remote code.

Replaces ``MiniMindBlock``'s 8-step ``RMSNorm`` Python body with a
single ``torch.ops.aten._fused_rms_norm`` call. Numerically equivalent
to the upstream formula:

    x = x * rsqrt(mean(x^2) + eps)
    y = weight * x

Verified bit-exact vs the upstream implementation for the float16
weights and inputs MiniMind-O uses.

The patch is class-level so every layer instance routes through the
fused op the first time the model is touched. Safe to call multiple
times.
"""

from __future__ import annotations

from typing import Any

_FUSED_MARKER = "_nanovllm_fused_rms"


def _fused_rms_forward(self: Any, x: Any) -> Any:
    import torch

    y, _ = torch.ops.aten._fused_rms_norm(x, list(self.weight.shape), self.weight, self.eps)
    return y


def enable_fused_rmsnorm(model: Any) -> int:
    """Monkey-patch every RMSNorm instance on the model to the fused op."""
    patched = 0
    seen: set[type] = set()
    for module in model.modules():
        cls = type(module)
        if cls.__name__ != "RMSNorm":
            continue
        if cls in seen:
            continue
        if getattr(cls, _FUSED_MARKER, False):
            seen.add(cls)
            continue
        cls.forward = _fused_rms_forward  # type: ignore[assignment]
        setattr(cls, _FUSED_MARKER, True)
        seen.add(cls)
        patched += 1
    return patched


__all__ = ["enable_fused_rmsnorm"]
