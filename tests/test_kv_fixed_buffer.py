"""Correctness test for the fixed-length KV cache buffer mechanism.

Motivation (docs/perf/ncu-generate-kernels-2026-09-01.md §6.4): CUDA Graph
capture is the only remaining lever to cut the 11 367 cudaLaunchKernel
calls per generate. But CUDA Graph requires static tensor shapes — the KV
cache cannot grow via ``torch.cat`` each AR step (E14 crashed exactly on
this). The fix is a pre-allocated fixed-length KV buffer that the forward
writes into with slice assignment and reads back as views.

This test validates that the fixed-buffer slice-write + slice-view pattern
is *numerically identical* to the current ``torch.cat`` KV update, on CPU.
If a future CUDA Graph implementation switches to the buffer, it must not
change the numbers the attention sees.

No GPU required. CPU-only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def kv_update_cat(
    past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    """Current implementation: grow KV via torch.cat each step."""
    if past_kv is None:
        k, v = new_key, new_value
    else:
        pk, pv = past_kv
        k = torch.cat([pk, new_key], dim=1)
        v = torch.cat([pv, new_value], dim=1)
    return (k, v), k.shape[1]


def kv_update_buffered(
    buffer_k: torch.Tensor,
    buffer_v: torch.Tensor,
    pos: int,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    """Fixed-buffer version: write into a slice of a pre-allocated buffer,
    return a view of the used prefix."""
    seq_len = new_key.shape[1]
    buffer_k[:, pos : pos + seq_len] = new_key
    buffer_v[:, pos : pos + seq_len] = new_value
    # Views into the buffer; same logical content as the cat version, same
    # shapes, but no allocation / no new kernel launch of a grow-op.
    return (buffer_k[:, : pos + seq_len], buffer_v[:, : pos + seq_len]), pos + seq_len


def _simulate_ar_steps(n_steps: int, max_len: int, n_kv_heads: int, head_dim: int, batch: int):
    """Run both KV strategies over n_steps AR decodes and compare outputs."""
    torch.manual_seed(42)
    # Pre-allocated buffer sized to max AR sequence.
    buffer_k = torch.zeros(batch, max_len, n_kv_heads, head_dim)
    buffer_v = torch.zeros(batch, max_len, n_kv_heads, head_dim)

    cat_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    buf_position = 0
    all_equal = True
    seq_lens: list[int] = []

    for step in range(n_steps):
        new_k = torch.randn(batch, 1, n_kv_heads, head_dim)
        new_v = torch.randn(batch, 1, n_kv_heads, head_dim)
        # Cat version to compare against.
        cat_kv, clen = kv_update_cat(cat_kv, new_k, new_v)
        buf_kv, blen = kv_update_buffered(buffer_k, buffer_v, buf_position, new_k, new_v)
        seq_lens.append(blen)

        # Both must produce the same length and same values.
        assert clen == blen, f"step {step}: cat len {clen} != buffer len {blen}"
        same_k = torch.equal(cat_kv[0], buf_kv[0])
        same_v = torch.equal(cat_kv[1], buf_kv[1])
        if not (same_k and same_v):
            all_equal = False
            print(f"step {step}: MISMATCH k_same={same_k} v_same={same_v}")
            break
        buf_position = blen

    return all_equal, seq_lens


def test_fixed_buffer_matches_cat_over_16_steps() -> None:
    """16 AR steps (matches max_tokens=16) must produce identical KV."""
    n_steps, max_len, n_kv_heads, head_dim, batch = 16, 16, 4, 64, 1
    ok, seq_lens = _simulate_ar_steps(n_steps, max_len, n_kv_heads, head_dim, batch)
    assert ok, "fixed-buffer KV diverged from cat KV"
    assert seq_lens == list(range(1, n_steps + 1)), f"unexpected seq len growth: {seq_lens}"


def test_fixed_buffer_respects_buffer_boundary() -> None:
    """Writes must stay within the pre-allocated buffer (no OOB)."""
    n_steps, max_len, n_kv_heads, head_dim, batch = 16, 16, 4, 64, 1
    buffer_k = torch.zeros(batch, max_len, n_kv_heads, head_dim)
    buffer_v = torch.zeros(batch, max_len, n_kv_heads, head_dim)
    pos = 0
    for _ in range(n_steps):
        new_k = torch.randn(batch, 1, n_kv_heads, head_dim)
        new_v = torch.randn(batch, 1, n_kv_heads, head_dim)
        _, pos = kv_update_buffered(buffer_k, buffer_v, pos, new_k, new_v)
    # pos should be exactly max_len after 16 single-token steps.
    assert pos == max_len


def test_fixed_buffer_multi_token_append() -> None:
    """Prefill writes N tokens in one call; the buffer handles multi-token
    appends (prefill is sequence_len>1, not just decode Q=1)."""
    batch, n_kv_heads, head_dim = 1, 4, 64
    max_len = 32
    buffer_k = torch.zeros(batch, max_len, n_kv_heads, head_dim)
    buffer_v = torch.zeros(batch, max_len, n_kv_heads, head_dim)

    # Prefill of 10 tokens.
    prefill_k = torch.randn(batch, 10, n_kv_heads, head_dim)
    prefill_v = torch.randn(batch, 10, n_kv_heads, head_dim)
    ref_kv, _ = kv_update_cat(None, prefill_k, prefill_v)
    buf_kv, blen = kv_update_buffered(buffer_k, buffer_v, 0, prefill_k, prefill_v)
    assert blen == 10
    assert torch.equal(ref_kv[0], buf_kv[0])
    assert torch.equal(ref_kv[1], buf_kv[1])

    # Then 3 decode steps of 1 token each.
    pos = blen
    for _ in range(3):
        dk = torch.randn(batch, 1, n_kv_heads, head_dim)
        dv = torch.randn(batch, 1, n_kv_heads, head_dim)
        ref_kv, _ = kv_update_cat(ref_kv, dk, dv)
        buf_kv, pos = kv_update_buffered(buffer_k, buffer_v, pos, dk, dv)
    assert pos == 13
    assert torch.equal(ref_kv[0], buf_kv[0])
    assert torch.equal(ref_kv[1], buf_kv[1])


if __name__ == "__main__":
    import sys

    checks = [
        test_fixed_buffer_matches_cat_over_16_steps,
        test_fixed_buffer_respects_buffer_boundary,
        test_fixed_buffer_multi_token_append,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
