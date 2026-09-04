"""CPU test: does SDPA with padded KV + additive attention mask produce
bit-identical output to SDPA with exact-length (cat) KV?

This de-risks CUDA Graph capture strategy A (docs/perf/
cuda-graph-capture-plan-2026-09-01.md): pad KV cache to max_len so shapes
are static, pass an additive attention_mask that zeroes-out padded
positions. For this to be a valid swap, the padded output's unpadded slice
must equal the exact-length output. If it does, a single static-shape CUDA
Graph (strategy A) is numerically correct — no dynamic-shape re-capture
(strategy B) needed.

No GPU required. Pure CPU math on scaled_dot_product_attention.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as functional  # noqa: E402 -- after importorskip

# Decode shape: Q has seq_len=1 (is_causal=False, attends to all past).
BATCH = 1
N_HEADS = 4
HEAD_DIM = 64
CUR_LEN = 7  # current KV length in an AR step
MAX_LEN = 16  # pre-allocated buffer size


def _build_decode_tensors(cur_len: int, max_len: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM)
    k_exact = torch.randn(BATCH, N_HEADS, cur_len, HEAD_DIM)
    v_exact = torch.randn(BATCH, N_HEADS, cur_len, HEAD_DIM)

    # Padded to max_len: valid KV in [0, cur_len), padding zeros after.
    k_padded = torch.zeros(BATCH, N_HEADS, max_len, HEAD_DIM)
    v_padded = torch.zeros(BATCH, N_HEADS, max_len, HEAD_DIM)
    k_padded[:, :, :cur_len] = k_exact
    v_padded[:, :, :cur_len] = v_exact
    return q, (k_exact, v_exact), (k_padded, v_padded)


def _exact_sdpa(q, k_exact, v_exact):
    """Baseline: is_causal=False on exact-length KV (the current code path)."""
    return functional.scaled_dot_product_attention(
        q, k_exact, v_exact, dropout_p=0.0, is_causal=False
    )


def _padded_sdpa(q, k_padded, v_padded, cur_len):
    max_len = k_padded.shape[-2]
    """Strategy A: additive mask -inf on padded positions [cur_len, max_len).

    HF attention_mask convention here: 1 = attend, 0 = masked. We build the
    additive 4D mask in the SDPA additive format (0 for keep, -inf for pad)
    directly.
    """
    # attn_mask shape for SDPA: [B, 1 (heads broadcast), 1 (q len), max_len]
    mask_logical = torch.zeros(BATCH, 1, 1, max_len)
    mask_logical[:, :, :, :cur_len] = 1.0  # 1 = attend valid positions
    additive = torch.where(mask_logical.bool(), 0.0, float("-inf"))
    assert torch.equal(additive[:, :, :, :cur_len], torch.zeros_like(additive[:, :, :, :cur_len]))
    assert (additive[:, :, :, cur_len:] == float("-inf")).all()
    return functional.scaled_dot_product_attention(
        q, k_padded, v_padded, attn_mask=additive, dropout_p=0.0, is_causal=False
    )


def test_padded_sdpa_slice_equals_exact() -> None:
    """Output of padded+mask SDPA at [:cur_len] must equal exact KV SDPA."""
    q, (k_exact, v_exact), (k_padded, v_padded) = _build_decode_tensors(CUR_LEN, MAX_LEN)
    out_exact = _exact_sdpa(q, k_exact, v_exact)
    out_padded = _padded_sdpa(q, k_padded, v_padded, CUR_LEN)
    # SDPA output shape: [B, H, 1, D] — no seq dim to slice, so compare whole.
    assert out_exact.shape == out_padded.shape, (out_exact.shape, out_padded.shape)
    max_diff = (out_exact - out_padded).abs().max().item()
    # Softmax with -inf mask positions contributes exactly 0; fp16/fp32 should
    # match to float tolerance. We're in fp32 on CPU; allow 1e-5.
    assert max_diff < 1e-5, f"padded output diverges from exact by {max_diff}"


def test_padded_attention_with_zero_value_pad() -> None:
    """Even though padded positions have zero K/V, the mask (not zeroing) is
    what excludes them; verify masking is doing the work (sanity: an
    unmasked SDPA on zero-padded K massively diverges, proving mask matters)."""
    q, (k_exact, v_exact), (k_padded, v_padded) = _build_decode_tensors(CUR_LEN, MAX_LEN)
    out_masked = _padded_sdpa(q, k_padded, v_padded, CUR_LEN)
    out_exact = _exact_sdpa(q, k_exact, v_exact)
    # Unmasked padded version: attends to zero K with non-masked slots.
    out_unmasked = functional.scaled_dot_product_attention(
        q, k_padded, v_padded, dropout_p=0.0, is_causal=False
    )
    # Masked should be close to exact; unmasked should clearly differ.
    d_masked = (out_masked - out_exact).abs().max().item()
    d_unmasked = (out_unmasked - out_exact).abs().max().item()
    assert d_masked < 1e-5, f"masked should match exact, diverges {d_masked}"
    assert (
        d_unmasked > 0.1
    ), f"unmasked should diverge from exact (mask does the work), got {d_unmasked}"


def test_mask_across_varying_cur_len() -> None:
    """Equivalent at multiple KV lengths: cur_len = 1, 5, 9, 15 (the lengths a
    decode loop visits when max_len=16)."""
    for cur in (1, 5, 9, 15):
        q, (k_exact, v_exact), (k_padded, v_padded) = _build_decode_tensors(cur, MAX_LEN, seed=cur)
        out_exact = _exact_sdpa(q, k_exact, v_exact)
        out_padded = _padded_sdpa(q, k_padded, v_padded, cur)
        d = (out_exact - out_padded).abs().max().item()
        assert d < 1e-5, f"cur_len={cur}: padded diverges by {d}"


def test_mask_needs_float_min_not_zero() -> None:
    """The additive mask must use -inf, not 0, for padded positions (0 would
    add a zero attention logit that survives softmax, changing the result)."""
    q, (k_exact, v_exact), (k_padded, v_padded) = _build_decode_tensors(CUR_LEN, MAX_LEN)
    # Wrong: use 0.0 additive (softmax sees both valid and padded).
    wrong_mask = torch.zeros(BATCH, 1, 1, MAX_LEN)
    out_wrong = functional.scaled_dot_product_attention(
        q, k_padded, v_padded, attn_mask=wrong_mask, dropout_p=0.0, is_causal=False
    )
    out_exact = _exact_sdpa(q, k_exact, v_exact)
    d = (out_wrong - out_exact).abs().max().item()
    assert d > 0.1, (
        f"zero-mask should diverge (adds padded attn mass), got {d}, meaning "
        f"masking semantics unexpectedly matched."
    )


if __name__ == "__main__":
    import sys

    checks = [
        test_padded_sdpa_slice_equals_exact,
        test_padded_attention_with_zero_value_pad,
        test_mask_across_varying_cur_len,
        test_mask_needs_float_min_not_zero,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
