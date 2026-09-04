"""Equivalence test: `_fused_apply_rotary_pos_emb` (nanovllm attention.py)
must match the upstream `apply_rotary_pos_emb` / `rotate_half`
(minimind-3o model_minimind.py) numerically.

Motivation: this is the E7/E12 optimization (docs/perf/
minimind-omni-under-500ms.md) — a fused RoPE that removes `rotate_half`'s
``torch.cat`` launch (+ saves launches under torch.compile), worth ~11ms.
The upstream `rotate_half` does ``torch.cat((-x[...half:], x[..., :half]))``
then ``q*cos + rotate_half(q)*sin``. The fused form pre-expands ``cos/sin``
to ``cat(half, half)`` and applies the algebraic identity directly:

    out[:, :half] = q[:, :half] * cos_half - q[:, half:] * sin_half
    out[:, half:] = q[:, half:] * cos_half + q[:, :half] * sin_half

The docstring claims bit-exactness (max abs diff 0.0, fp16). This test
locks that claim for both fp32 and fp16 so a future refactor can't break
the optimization silently. CPU-only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional  # noqa: F401  (kept for readability)

# Reuse the live implementation — the thing under test.
from nanovllm_omni.models.minimind_omni.attention import _fused_apply_rotary_pos_emb


def _upstream_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Mirror of model_minimind.py apply_rotary_pos_emb + rotate_half."""

    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def _precompute_freqs_cis(dim: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate upstream precompute_freqs_cis (model_minimind.py lines 73-77)."""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # repeat to dim
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def _build_inputs(
    dtype: torch.dtype, seq: int, head_dim: int = 128, n_heads: int = 4, pos: int = 0
):
    torch.manual_seed(42)
    # Real call site (attention.py _attention_forward): query is
    # [B, seq_len, n_local_heads, head_dim] (4D) after qkv projection +
    # reshape. Both upstream and fused receive this 4D tensor.
    q = torch.randn(1, seq, n_heads, head_dim, dtype=dtype)
    k = torch.randn(1, seq, n_heads, head_dim, dtype=dtype)
    cos, sin = _precompute_freqs_cis(head_dim, end=pos + seq)
    # Upstream constructs position_embeddings as (freqs_cos[start:end],
    # freqs_sin[start:end]) — 2D [seq, head_dim], no batch dim. The RoPE
    # functions unsqueeze(1) internally.
    c = cos[pos : pos + seq]  # [seq, head_dim]
    s = sin[pos : pos + seq]
    return q, k, c.to(dtype), s.to(dtype)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_fused_matches_upstream_fp32() -> None:
    q, k, cos, sin = _build_inputs(torch.float32, seq=8, pos=0)
    up_q, up_k = _upstream_apply_rotary_pos_emb(q, k, cos, sin)
    fused_q, fused_k = _fused_apply_rotary_pos_emb(q, k, cos, sin)
    assert fused_q.shape == up_q.shape
    assert fused_k.shape == up_k.shape
    assert (
        _max_abs_diff(fused_q, up_q) < 1e-5
    ), f"fp32 Q RoPE diverges from upstream: {_max_abs_diff(fused_q, up_q)}"
    assert (
        _max_abs_diff(fused_k, up_k) < 1e-5
    ), f"fp32 K RoPE diverges from upstream: {_max_abs_diff(fused_k, up_k)}"


def test_fused_matches_upstream_fp16() -> None:
    q, k, cos, sin = _build_inputs(torch.float16, seq=8, pos=0)
    up_q, up_k = _upstream_apply_rotary_pos_emb(q, k, cos, sin)
    fused_q, fused_k = _fused_apply_rotary_pos_emb(q, k, cos, sin)
    # fp16 ulp; the docstring claims 0.0. Allow one ulp of fp16 slack
    # (head dim 128, modest magnitudes) but keep the bound tight.
    dq = _max_abs_diff(fused_q, up_q)
    dk = _max_abs_diff(fused_k, up_k)
    assert dq < 1e-3, f"fp16 Q RoPE diverges from upstream: {dq}"
    assert dk < 1e-3, f"fp16 K RoPE diverges from upstream: {dk}"


def test_fused_matches_upstream_across_offsets() -> None:
    """RoPE is position-dependent; verify equivalence at several past-KV
    offsets (positions 0, 9, 23)."""
    for pos in (0, 9, 23):
        q, k, cos, sin = _build_inputs(torch.float32, seq=4, pos=pos)
        up_q, _ = _upstream_apply_rotary_pos_emb(q, k, cos, sin)
        fused_q, _ = _fused_apply_rotary_pos_emb(q, k, cos, sin)
        assert _max_abs_diff(fused_q, up_q) < 1e-5, f"pos={pos}: RoPE diverges"


def test_fused_identity_holds_half_dim() -> None:
    """Head dim must be even (RoPE splits in half); odd dims are not RoPE
    targets but the fused form should still not crash with odd lengths —
    upstream would produce a malformed cat. We only assert even-dim path is
    correct, and that the fused form is defined for the real head_dim used
    by minimind-3o (128, even)."""
    # head_dim for minimind-3o: config.head_dim is 128 (even) -> rotate half valid.
    q, k, cos, sin = _build_inputs(torch.float32, seq=2, head_dim=128)
    fused_q, _ = _fused_apply_rotary_pos_emb(q, k, cos, sin)
    assert fused_q.shape[-1] == 128
    # Values must be finite.
    assert torch.isfinite(fused_q).all()


if __name__ == "__main__":
    import sys

    checks = [
        test_fused_matches_upstream_fp32,
        test_fused_matches_upstream_fp16,
        test_fused_matches_upstream_across_offsets,
        test_fused_identity_holds_half_dim,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
