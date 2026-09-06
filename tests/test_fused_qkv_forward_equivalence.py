"""Equivalence test: fused QKV attention forward (1 matmul via
`_fused_attention_forward`) must produce numerically identical output to
the unfused forward (3 separate q/k/v matmuls via `_sdpa_forward`).

This is the core guarantee of the E6/E24 fusion (docs/perf/
minimind-omni-under-500ms.md): fusing q_proj+k_proj+v_proj into a single
qkv_proj must NOT change the attention output. #9 already verifies the
weight concatenation is right ([q;k;v] concat) and all instances get
fused; this test verifies the *forward* output is unchanged end-to-end,
for both decode (Q=1) and prefill (Q=N) branch shapes, using the real
attention.py forward functions on minimal stub modules.

CPU-only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402 -- after importorskip

from nanovllm_omni.optim import attention as attn_mod  # noqa: E402 -- after importorskip


class _RMSNorm(nn.Module):
    """Identity stand-in for q_norm/k_norm: must not break the fused path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _make_stub_attention(n_heads: int, n_kv_heads: int, head_dim: int, feat_dim: int):
    """Minimal Attention-like module satisfying _attention_forward's needs:
    flash, is_causal, dropout, attn_dropout, resid_dropout, o_proj, head_dim,
    n_local_heads, n_local_kv_heads, and q/k/v projections."""
    stub = nn.Module()
    stub.n_local_heads = n_heads
    stub.n_local_kv_heads = n_kv_heads
    stub.head_dim = head_dim
    stub.flash = True
    stub.is_causal = True
    stub.dropout = 0.0
    stub.training = False
    stub.attn_dropout = nn.Dropout(0.0)
    stub.resid_dropout = nn.Dropout(0.0)
    stub.o_proj = nn.Linear(n_heads * head_dim, feat_dim, bias=False)
    stub.q_proj = nn.Linear(feat_dim, n_heads * head_dim, bias=False)
    stub.k_proj = nn.Linear(feat_dim, n_kv_heads * head_dim, bias=False)
    stub.v_proj = nn.Linear(feat_dim, n_kv_heads * head_dim, bias=False)
    stub.q_norm = _RMSNorm(head_dim)
    stub.k_norm = _RMSNorm(head_dim)
    stub.n_rep = max(1, n_heads // n_kv_heads)
    return stub


def _rotary_identity(q, k, cos, sin):
    """No-op RoPE for the equivalence test (both paths share the same one,
    so it cancels out; keeps the test focused on projection fusion)."""
    return q, k


def _run_forward(attn, x, past_key_value, sequence_len):
    """Run the attention forward with the shared tail on CPU. Both fused and
    unfused paths delegate the tail to `_attention_forward`."""
    from nanovllm_omni.optim.attention import (
        _fused_attention_forward,
        _sdpa_forward,
    )

    # Stub the module's apply_rotary_pos_emb (used inside _attention_forward)
    # to the identity for determinism.
    orig_import = attn_mod._import_upstream
    attn_mod._import_upstream = lambda _m: (  # type: ignore[attr-defined]
        type(
            "M",
            (),
            {
                "apply_rotary_pos_emb": staticmethod(_rotary_identity),
                "repeat_kv": staticmethod(_repeat_kv),
            },
        )()
    )
    try:
        fn = _fused_attention_forward if hasattr(attn, "qkv_proj") else _sdpa_forward
        out, past = fn(
            attn,
            x,
            position_embeddings=(None, None),
            past_key_value=past_key_value,
            use_cache=True,
            attention_mask=None,
        )
    finally:
        attn_mod._import_upstream = orig_import
    return out, past


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq, n_kv, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq, n_kv, n_rep, head_dim)
        .reshape(batch, seq, n_kv * n_rep, head_dim)
    )


def _build_x(batch: int, seq: int, feat_dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq, feat_dim)


def test_fused_matches_unfused_decode_q1() -> None:
    """Decode shape (Q=1) with past KV — the running-hot path in generate."""
    feat_dim, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16

    # Build two identical stubs; fuse one.
    unfused = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    fused = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    # Copy weights so the two are bit-identical before fusion.
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(fused, name).weight.data.copy_(getattr(unfused, name).weight.data)
    from nanovllm_omni.optim.attention import _fuse_attention_qkv

    _fuse_attention_qkv(fused)  # installs qkv_proj + fused forward

    x = _build_x(1, 1, feat_dim, seed=1)
    past_k = torch.randn(1, 5, n_kv_heads, head_dim)
    past_v = torch.randn(1, 5, n_kv_heads, head_dim)
    past_kv = (past_k, past_v)

    out_unfused, _ = _run_forward(unfused, x, past_kv, sequence_len=1)
    out_fused, _ = _run_forward(fused, x, past_kv, sequence_len=1)

    assert out_unfused.shape == out_fused.shape, (out_unfused.shape, out_fused.shape)
    max_diff = (out_unfused - out_fused).abs().max().item()
    assert max_diff < 1e-5, f"decode fused output diverges from unfused by {max_diff}"


def test_fused_matches_unfused_prefill() -> None:
    """Prefill shape (Q=N sequence, no past KV)."""
    feat_dim, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
    unfused = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    fused = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(fused, name).weight.data.copy_(getattr(unfused, name).weight.data)
    from nanovllm_omni.optim.attention import _fuse_attention_qkv

    _fuse_attention_qkv(fused)

    x = _build_x(1, 4, feat_dim, seed=2)
    out_unfused, _ = _run_forward(unfused, x, None, sequence_len=4)
    out_fused, _ = _run_forward(fused, x, None, sequence_len=4)
    max_diff = (out_unfused - out_fused).abs().max().item()
    assert max_diff < 1e-5, f"prefill fused output diverges by {max_diff}"


def test_fused_forward_selected_by_install() -> None:
    """After fusion the module's forward must be the fused one (this is what
    makes the E24 dedupe fix observable at runtime)."""
    feat_dim, n_heads, n_kv_heads, head_dim = 32, 2, 1, 16
    stub = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    from nanovllm_omni.optim.attention import (
        _fuse_attention_qkv,
        _fused_attention_forward,
    )

    assert not hasattr(stub, "qkv_proj"), "precondition: no qkv_proj yet"
    _fuse_attention_qkv(stub)
    assert hasattr(stub, "qkv_proj")
    assert stub.forward is not None
    # The installed bound forward functools.partial is the fused fn; verify by
    # checking the bound __name__ via func.
    bound = stub.forward
    func = getattr(bound, "__func__", None)
    assert func is not None and func.__name__ == _fused_attention_forward.__name__


if __name__ == "__main__":
    import sys

    checks = [
        test_fused_matches_unfused_decode_q1,
        test_fused_matches_unfused_prefill,
        test_fused_forward_selected_by_install,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
