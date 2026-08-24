"""Fused RMSNorm monkey-patch for MiniMind-O remote code.

Replaces the upstream ``RMSNorm`` Python body with a single
``torch.ops.aten._fused_rms_norm`` call while preserving the upstream
fp32-in-fp16-out precision pattern.

The upstream implementation casts ``x`` to float32 before the
normalization and casts the result back to ``x.dtype``:

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

where ``self.norm`` is ``x * rsqrt(mean(x^2) + eps)``. The naive
``torch.ops.aten._fused_rms_norm(x, ...)`` runs the reduction in ``x``'s
dtype (float16 for MiniMind-O after ``.half()``). For one call the
difference is at fp16 ULP scale, but 8 MiniMind-O RMSNorms per layer
``* 12 layers * 30+ decode steps`` accumulates into a measurable
shift in the multinomial text/audio logits, breaking the bit-exact
fixed-seed reproduction the project relies on.

The patch below mirrors the upstream cast pattern: upcast to fp32 for
the reduction, multiply by the (fp16) weight, then cast back. The
fused op still eliminates the Python-side power / mean / rsqrt / mul
launches; only the precision pattern changes.

Safe to call multiple times -- the marker is class-level so a
second ``enable_fused_rmsnorm`` on the same model is a no-op.
"""

from __future__ import annotations

from typing import Any

_FUSED_MARKER = "_nanovllm_fused_rms"


def _fused_rms_forward(self: Any, x: Any) -> Any:
    import torch

    in_dtype = x.dtype
    # The fused op expects ``weight`` to share the input dtype, but the
    # upstream RMSNorm explicitly casts ``x`` to fp32 for the reduction.
    # We do the same: cast input up for the reduction and the weight up
    # for the multiply, then cast the result back to ``x.dtype``.
    weight = self.weight.to(torch.float32) if self.weight.dtype != torch.float32 else self.weight
    y, _ = torch.ops.aten._fused_rms_norm(x.to(torch.float32), list(weight.shape), weight, self.eps)
    return y.to(in_dtype)


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
