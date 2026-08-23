"""Fused RoPE monkey-patch for MiniMind-O remote code.

The upstream ``apply_rotary_pos_emb`` computes ``q * cos + rotate_half(q) * sin``.
``rotate_half`` is ``torch.cat((-x[..., D/2:], x[..., :D/2]), dim=-1)`` which is
one extra launch per call plus a memory round-trip.

Because ``freqs_cos = cat(cos_half, cos_half)`` and ``freqs_sin = cat(sin_half,
sin_half)`` in precompute_freqs_cis, the formula can be rewritten without a
concat / cat at all:

    q_rot[..., :D/2] = q[..., :D/2] * cos - q[..., D/2:] * sin
    q_rot[..., D/2:] = q[..., D/2:] * cos + q[..., :D/2] * sin

This replaces 7 launches per (q, k) pair with 6, dropping the cat entirely.

Verified bit-exact vs the upstream implementation for the float16 inputs
MiniMind-O uses (max abs diff: 0.0 in fp16).
"""

from __future__ import annotations

from typing import Any

_FUSED_ROPE_MARKER = "_nanovllm_fused_rope"


def _fused_apply_rotary_pos_emb(
    q: Any, k: Any, cos: Any, sin: Any, unsqueeze_dim: int = 1
) -> tuple[Any, Any]:
    import torch

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_half = q.shape[-1] // 2
    q_cos = cos[..., :q_half]
    q_sin = sin[..., :q_half]
    q1 = q[..., :q_half]
    q2 = q[..., q_half:]
    q_rot = torch.empty_like(q)
    q_rot[..., :q_half] = q1 * q_cos - q2 * q_sin
    q_rot[..., q_half:] = q2 * q_cos + q1 * q_sin

    k_half = k.shape[-1] // 2
    k_cos = cos[..., :k_half]
    k_sin = sin[..., :k_half]
    k1 = k[..., :k_half]
    k2 = k[..., k_half:]
    k_rot = torch.empty_like(k)
    k_rot[..., :k_half] = k1 * k_cos - k2 * k_sin
    k_rot[..., k_half:] = k2 * k_cos + k1 * k_sin

    return q_rot, k_rot


def enable_fused_rope(model: Any) -> int:
    """Patch the upstream ``apply_rotary_pos_emb`` module function to the fused form.

    Optionally wraps it with ``torch.compile(dynamic=True)`` so Inductor can
    fuse the 6 elementwise launches into 1–2 kernels. ``dynamic=True`` avoids
    recompilation as the sequence length grows during decoding.
    """
    try:
        module = __import__(
            type(model.thinker.layers[0].self_attn).__module__, fromlist=["apply_rotary_pos_emb"]
        )
    except (ImportError, AttributeError):
        return 0
    if getattr(module, _FUSED_ROPE_MARKER, False):
        return 0
    fn = _fused_apply_rotary_pos_emb
    try:
        import torch

        fn = torch.compile(fn, dynamic=True)
    except Exception:
        fn = _fused_apply_rotary_pos_emb
    module.apply_rotary_pos_emb = fn
    module._nanovllm_fused_rope = True
    return 1


__all__ = ["enable_fused_rope"]
