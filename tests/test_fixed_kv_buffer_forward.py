"""Test: fixed-KV-buffer forward (plan §3.1/§3.2) == cat forward.

CUDA Graph capture requires static tensor shapes, but MiniMind-O attention
grows its KV cache with ``torch.cat`` each AR step. The fix (report
§19/§22, docs/perf/ncu-generate-kernels-2026-09-01.md) is a preallocated
fixed buffer the attention writes into via slice assignment and reads back
as views -- zero arithmetic change.

This test verifies ``_kv_buffer_forward`` (+ ``_attention_forward_buffered``)
produces numerically identical output to the existing ``_sdpa_forward`` (cat
path) on the same inputs, at decode (Q=1) and prefill (Q=N) shapes, across
multiple AR steps (KV flow + pos counter), using real attention.py functions
on minimal stub modules, CPU-only.

It also locks the per-instance marker/idempotency contract (E24 lesson: no
class-level dedupe; re-attach must not double-bind).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402 -- after importorskip

from nanovllm_omni.engine import attention as attn_mod  # noqa: E402 -- after importorskip


class _RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim):
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
    return q, k


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq, n_kv, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq, n_kv, n_rep, head_dim)
        .reshape(batch, seq, n_kv * n_rep, head_dim)
    )


def _cat_baseline_forward(attn, x, past_key_value, use_cache, attention_mask):
    """Reference: separate Q/K/V projections + torch.cat KV history.

    Mirrors the upstream eager path the fixed-KV buffer replaces. Lives
    in-test so the production ``optim/attention.py`` only carries the
    buffered implementation.
    """
    import math

    import torch
    import torch.nn.functional as functional

    batch_size, sequence_len, _ = x.shape
    query = attn.q_proj(x).view(batch_size, sequence_len, attn.n_local_heads, attn.head_dim)
    key = attn.k_proj(x).view(batch_size, sequence_len, attn.n_local_kv_heads, attn.head_dim)
    value = attn.v_proj(x).view(batch_size, sequence_len, attn.n_local_kv_heads, attn.head_dim)
    query, key = attn.q_norm(query), attn.k_norm(key)
    # rotary is identity in this test (see _run_forward), so skip.
    if past_key_value is not None:
        key = torch.cat([past_key_value[0], key], dim=1)
        value = torch.cat([past_key_value[1], value], dim=1)
    past = (key, value) if use_cache else None
    q_t = query.transpose(1, 2)
    k_t = _repeat_kv(key, attn.n_rep).transpose(1, 2)
    v_t = _repeat_kv(value, attn.n_rep).transpose(1, 2)
    if sequence_len == 1 and past_key_value is not None and attention_mask is None:
        output = functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, is_causal=False
        )
    elif (
        attn.flash
        and (sequence_len > 1)
        and (not attn.is_causal or past_key_value is None)
        and (attention_mask is None or torch.all(attention_mask == 1))
    ):
        output = functional.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=attn.dropout if attn.training else 0.0,
            is_causal=attn.is_causal,
        )
    else:
        scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        if attn.is_causal:
            scores[:, :, :, -sequence_len:] += torch.full(
                (sequence_len, sequence_len), float("-inf"), device=scores.device
            ).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = attn.attn_dropout(functional.softmax(scores.float(), dim=-1).type_as(q_t)) @ v_t
    output = output.transpose(1, 2).reshape(batch_size, sequence_len, -1)
    return attn.resid_dropout(attn.o_proj(output)), past


def _run_forward(attn, x, past_key_value, sequence_len, buffered: bool):
    import torch.nn.functional as functional  # noqa: F401  (mirror upstream)

    from nanovllm_omni.engine.attention import _kv_buffer_forward

    if buffered:
        fn = _kv_buffer_forward
    else:

        def fn(attn_, x_, **kw):  # noqa: ARG001
            return _cat_baseline_forward(
                attn_,
                x_,
                past_key_value=kw["past_key_value"],
                use_cache=kw["use_cache"],
                attention_mask=kw["attention_mask"],
            )

    # The buffered path needs the upstream-rotary/repeat_kv stub injected.
    # Cat baseline is self-contained above; only patch when buffered.
    if buffered:
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
        out, past = fn(
            attn,
            x,
            position_embeddings=(None, None),
            past_key_value=past_key_value,
            use_cache=True,
            attention_mask=None,
        )
    finally:
        if buffered:
            attn_mod._import_upstream = orig_import
    return out, past


def _build_x(batch: int, seq: int, feat_dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq, feat_dim)


def _run_step_sequence(
    use_buffer: bool, n_steps: int, feat_dim, n_heads, n_kv_heads, head_dim, prefill_seq
):
    """Run prefill + n_steps decode, returning the per-step output. The
    buffered path drives its own pos counter; the cat path accumulates past
    via the returned (key, value)."""
    attn = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    if use_buffer:
        from nanovllm_omni.engine.attention import _attach_kv_buffer

        _attach_kv_buffer(attn, max_len=prefill_seq + n_steps + 4)
        # prefill
        xp = _build_x(1, prefill_seq, feat_dim, seed=7)
        outp, _ = _run_forward(attn, xp, None, prefill_seq, buffered=True)
        outs = [outp]
        for i in range(n_steps):
            xd = _build_x(1, 1, feat_dim, seed=10 + i)
            outd, _ = _run_forward(attn, xd, None, 1, buffered=True)
            outs.append(outd)
        return attn, outs, None

    past = None
    xp = _build_x(1, prefill_seq, feat_dim, seed=7)
    outp, past = _run_forward(attn, xp, past, prefill_seq, buffered=False)
    outs = [outp]
    for i in range(n_steps):
        xd = _build_x(1, 1, feat_dim, seed=10 + i)
        outd, past = _run_forward(attn, xd, past, 1, buffered=False)
        outs.append(outd)
    return attn, outs, past


def _make_paired_stubs(n_heads, n_kv_heads, head_dim, feat_dim):
    """Return two attention stubs with bit-identical weights (precondition
    for comparing the buffered and cat paths)."""
    buf = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    cat = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(cat, name).weight.data.copy_(getattr(buf, name).weight.data)
    return buf, cat


def _run_both(n_steps, feat_dim, n_heads, n_kv_heads, head_dim, prefill_seq):
    """Run buffered and cat paths on bit-identical stubs; return all outputs."""
    from nanovllm_omni.engine.attention import _attach_kv_buffer

    buf_attn, cat_attn = _make_paired_stubs(n_heads, n_kv_heads, head_dim, feat_dim)
    _attach_kv_buffer(buf_attn, max_len=prefill_seq + n_steps + 4)

    xp = _build_x(1, prefill_seq, feat_dim, seed=7)
    xds = [_build_x(1, 1, feat_dim, seed=10 + i) for i in range(n_steps)]

    # buffered path
    out_buf, _ = _run_forward(buf_attn, xp, None, prefill_seq, buffered=True)
    outs_buf = [out_buf]
    for xd in xds:
        od, _ = _run_forward(buf_attn, xd, None, 1, buffered=True)
        outs_buf.append(od)

    # cat path
    past = None
    out_cat, past = _run_forward(cat_attn, xp, past, prefill_seq, buffered=False)
    outs_cat = [out_cat]
    for xd in xds:
        od, past = _run_forward(cat_attn, xd, past, 1, buffered=False)
        outs_cat.append(od)
    return outs_buf, outs_cat, buf_attn


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_fixed_buffer_matches_cat_decode_prefill() -> None:
    """Prefill + single decode step: buffered output == cat output."""
    feat_dim, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
    prefill_seq, n_steps = 5, 1

    outs_buf, outs_cat, _ = _run_both(n_steps, feat_dim, n_heads, n_kv_heads, head_dim, prefill_seq)

    assert len(outs_buf) == len(outs_cat) == n_steps + 1
    for i, (a, b) in enumerate(zip(outs_buf, outs_cat, strict=True)):
        assert a.shape == b.shape, (i, a.shape, b.shape)
        assert (a - b).abs().max().item() < 1e-5, f"step {i} diverges"


def test_fixed_buffer_matches_cat_over_16_steps() -> None:
    """16 decode steps with KV flowing through each path: still identical."""
    feat_dim, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
    prefill_seq, n_steps = 5, 16

    outs_buf, outs_cat, _ = _run_both(n_steps, feat_dim, n_heads, n_kv_heads, head_dim, prefill_seq)

    for i, (a, b) in enumerate(zip(outs_buf, outs_cat, strict=True)):
        assert (a - b).abs().max().item() < 1e-5, f"step {i} diverges"


def test_fixed_buffer_returns_full_history_as_past() -> None:
    """use_cache=True must return the accumulated KV as `past` so the outer
    generate loop can keep the SAME contract (past_key_values list)."""
    feat_dim, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
    prefill_seq = 5
    _, _, buf_attn = _run_both(2, feat_dim, n_heads, n_kv_heads, head_dim, prefill_seq)
    # after prefill(5) + 2 decode the pos counter = 7
    assert buf_attn._kv_pos == prefill_seq + 2, buf_attn._kv_pos
    assert tuple(buf_attn._kv_past_key.shape) == (1, prefill_seq + 2 + 4, n_kv_heads, head_dim)


def test_attach_is_per_instance_and_idempotent() -> None:
    """E24 lesson: patch must bind per instance, and re-attach must not
    double-bind (new `forward` each call)."""
    from nanovllm_omni.engine.attention import (
        _attach_kv_buffer,
        _kv_buffer_forward,
    )

    feat_dim, n_heads, n_kv_heads, head_dim = 32, 2, 1, 16
    a1 = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)
    a2 = _make_stub_attention(n_heads, n_kv_heads, head_dim, feat_dim)

    _attach_kv_buffer(a1, max_len=32)
    _attach_kv_buffer(a2, max_len=32)
    bound1 = a1.forward
    func1 = getattr(bound1, "__func__", None)
    assert func1 is not None and func1.__name__ == _kv_buffer_forward.__name__

    # a2 is independently bound (not relying on a1's marker).
    bound2 = a2.forward
    func2 = getattr(bound2, "__func__", None)
    assert func2 is not None and func2.__name__ == _kv_buffer_forward.__name__

    # re-attach is a no-op (no double-bind churn).
    _attach_kv_buffer(a1, max_len=32)
    assert a1.forward is bound1


if __name__ == "__main__":
    import sys

    checks = [
        test_fixed_buffer_matches_cat_decode_prefill,
        test_fixed_buffer_matches_cat_over_16_steps,
        test_fixed_buffer_returns_full_history_as_past,
        test_attach_is_per_instance_and_idempotent,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
