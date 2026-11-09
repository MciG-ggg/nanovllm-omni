"""The self-authored MLX Metal kernels, checked against the dense path.

``mlx`` is a Mac-track extra and is not in the project venv, so every test
here is behind ``importorskip``. Nothing in this file touches
``MlxStageRunner``: this round validates the kernels alone.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
cache_module = pytest.importorskip("mlx_lm.models.cache")

from nanovllm_omni.backend.mlx_kernel import (  # noqa: E402
    DecodeShape,
    build,
    kernels,
)

BLOCK_SIZE = kernels.BLOCK_SIZE
HEAD_DIM = 96


def _shape(num_heads: int = 8, num_kv_heads: int = 4) -> DecodeShape:
    """MiniMind-O's attention shape: 8 heads, 4 KV heads, head_dim 96."""
    return DecodeShape(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=HEAD_DIM)


def _pool(block_table: list[list[int]], context_lens: list[int], pool_rows: int):
    """A pool plus a dense reference cache holding the same rows, block-table order."""
    batch = len(context_lens)
    mx.random.seed(0)
    shape = _shape()
    keys = mx.random.normal((pool_rows, shape.num_kv_heads, HEAD_DIM)).astype(mx.float16)
    values = mx.random.normal((pool_rows, shape.num_kv_heads, HEAD_DIM)).astype(mx.float16)
    query = mx.random.normal((batch, shape.num_heads, HEAD_DIM)).astype(mx.float16)
    table = mx.array(block_table, dtype=mx.int32)
    lens = mx.array(context_lens, dtype=mx.int32)
    return shape, query, keys, values, table, lens


def _dense_reference(query, keys, values, block_table, context_lens, shape):
    """Same keys through ``BatchKVCache`` + SDPA, the arm the MLX backend runs."""
    batch = query.shape[0]
    longest = max(context_lens)
    left_padding = [longest - length for length in context_lens]
    cache = cache_module.BatchKVCache(left_padding)
    feed_keys = mx.zeros((batch, shape.num_kv_heads, longest, HEAD_DIM), mx.float16)
    feed_values = mx.zeros((batch, shape.num_kv_heads, longest, HEAD_DIM), mx.float16)
    for row in range(batch):
        for position in range(context_lens[row]):
            block = block_table[row][position // BLOCK_SIZE]
            slot = block * BLOCK_SIZE + position % BLOCK_SIZE
            feed_keys[row, :, left_padding[row] + position] = keys[slot]
            feed_values[row, :, left_padding[row] + position] = values[slot]

    dense_keys, dense_values = cache.update_and_fetch(feed_keys, feed_values)
    group = shape.num_heads // shape.num_kv_heads
    dense_keys = mx.repeat(dense_keys, group, axis=1)
    dense_values = mx.repeat(dense_values, group, axis=1)
    # The pad columns are masked, exactly as the MLX backend relies on.
    positions = mx.arange(longest)[None, :]
    keep = positions >= mx.array(left_padding)[:, None]
    attended = mx.fast.scaled_dot_product_attention(
        query[:, :, None, :],
        dense_keys,
        dense_values,
        scale=HEAD_DIM**-0.5,
        mask=keep[:, None, None, :],
    )
    mx.eval(attended)
    return attended[:, :, 0, :]


def test_paged_decode_matches_dense_batch_kv_cache():
    block_table = [[0, 2, 5, -1], [1, 3, -1, -1], [4, 6, 7, -1]]
    context_lens = [600, 257, 300]

    shape, query, keys, values, table, lens = _pool(block_table, context_lens, 8 * BLOCK_SIZE)
    paged = kernels.paged_decode_attention(query, keys, values, table, lens, shape)
    dense = _dense_reference(query, keys, values, block_table, context_lens, shape)
    mx.eval(paged)

    difference = mx.abs(paged.astype(mx.float32) - dense.astype(mx.float32))
    assert float(difference.max()) < 2e-3


def _poisoned_pool(shape: DecodeShape, pool_rows: int):
    """A pool whose byte pattern would blow up any read that lands on it."""
    keys = mx.full((pool_rows, shape.num_kv_heads, HEAD_DIM), 3.0e4, mx.float16)
    values = mx.full((pool_rows, shape.num_kv_heads, HEAD_DIM), 3.0e4, mx.float16)
    return keys, values


def test_block_table_holes_are_never_read():
    # ``context_len`` says 512 keys, but every block-table row is -1 past the
    # first two blocks, so the positions those entries cover are holes. The
    # pool is poison everywhere: a read that trusts -1 produces 3e4 and the
    # finite-and-bounded assertions catch it.
    shape = _shape()
    keys, values = _poisoned_pool(shape, 4 * BLOCK_SIZE)
    for block in (0, 1):
        start = block * BLOCK_SIZE
        keys = keys.at[start : start + BLOCK_SIZE].multiply(0.0)
        values = values.at[start : start + BLOCK_SIZE].multiply(0.0)
    table = mx.array([[0, 1, -1, -1]], dtype=mx.int32)
    query = mx.ones((1, shape.num_heads, HEAD_DIM), mx.float16)

    attended = kernels.paged_decode_attention(
        query, keys, values, table, mx.array([512], mx.int32), shape
    )
    mx.eval(attended)

    # The two live blocks hold zeros, so a hole-free read averages to zero.
    assert float(mx.abs(attended).max()) == 0.0


def test_a_hole_is_skipped_not_absorbed():
    # Same 512 positions, all live, with one real (non-zero) key and the rest
    # zero: the result is that key's value scaled by its softmax weight, which
    # is strictly positive. Turning one position into a hole must change the
    # answer -- if -1 were read as a block, the pool's poison would swamp it.
    shape = _shape()
    keys = mx.zeros((4 * BLOCK_SIZE, shape.num_kv_heads, HEAD_DIM), mx.float16)
    values = mx.zeros((4 * BLOCK_SIZE, shape.num_kv_heads, HEAD_DIM), mx.float16)
    keys = keys.at[7].add(mx.ones((shape.num_kv_heads, HEAD_DIM), mx.float16))
    values = values.at[7].add(mx.ones((shape.num_kv_heads, HEAD_DIM), mx.float16))
    query = mx.ones((1, shape.num_heads, HEAD_DIM), mx.float16)
    lens = mx.array([512], mx.int32)

    live = kernels.paged_decode_attention(
        query, keys, values, mx.array([[0, 1, -1, -1]], mx.int32), lens, shape
    )
    # Block 0 becomes a hole, so the single non-zero key at row 7 is gone.
    holed = kernels.paged_decode_attention(
        query, keys, values, mx.array([[-1, 1, -1, -1]], mx.int32), lens, shape
    )
    mx.eval(live, holed)

    assert float(live.max()) > 0.0
    assert float(mx.abs(holed).max()) == 0.0


def test_store_kv_writes_only_live_slots():
    shape = _shape()
    pool_rows = 8 * BLOCK_SIZE
    keys = mx.zeros((pool_rows, shape.num_kv_heads, HEAD_DIM), mx.float16)
    values = mx.zeros((pool_rows, shape.num_kv_heads, HEAD_DIM), mx.float16)
    fresh_keys = mx.array(
        [mx.full((shape.num_kv_heads, HEAD_DIM), value, mx.float16) for value in (1.0, 2.0, 3.0)]
    )
    fresh_values = fresh_keys + 3.0
    slots = mx.array([5, 300, -1], dtype=mx.int32)

    written_keys, written_values = kernels.store_kv(
        fresh_keys, fresh_values, keys, values, slots, shape
    )
    mx.eval(written_keys, written_values)

    assert written_keys[5, 0, 0].item() == pytest.approx(1.0)
    assert written_keys[300, 0, 0].item() == pytest.approx(2.0)
    assert written_values[300, 0, 0].item() == pytest.approx(5.0)
    # slot -1 is not a slot: the third token's 3.0 must appear nowhere.
    assert int((written_keys[:, 0, 0] == 3.0).sum().item()) == 0
    assert bool(mx.all(written_keys[6] == 0).item())
    # A metal_kernel cannot write in place; the pools it was handed stay put.
    assert bool(mx.all(keys == 0).item())


def test_store_then_decode_sees_the_new_rows():
    shape = _shape()
    context_len = 300
    pool_rows = 8 * BLOCK_SIZE
    keys = mx.zeros((pool_rows, shape.num_kv_heads, HEAD_DIM), mx.float16)
    values = mx.zeros((pool_rows, shape.num_kv_heads, HEAD_DIM), mx.float16)
    fresh_keys = mx.ones((1, shape.num_kv_heads, HEAD_DIM), mx.float16)
    fresh_values = mx.ones((1, shape.num_kv_heads, HEAD_DIM), mx.float16) * 2.0
    slot = context_len - 1

    stored_keys, stored_values = kernels.store_kv(
        fresh_keys,
        fresh_values,
        keys,
        values,
        mx.array([slot], dtype=mx.int32),
        shape,
    )
    table = mx.array([[0, 1, -1]], dtype=mx.int32)
    query = mx.ones((1, shape.num_heads, HEAD_DIM), mx.float16)
    attended = kernels.paged_decode_attention(
        query, stored_keys, stored_values, table, mx.array([context_len], mx.int32), shape
    )
    mx.eval(attended)

    # Every key is zero except the stored one, so the output is a weighted
    # average of 2.0 and 0.0 -- never above the stored value.
    assert float(mx.abs(attended).max()) <= 2.0 + 1e-3
    assert bool(mx.all(mx.isfinite(attended)).item())


def test_empty_context_attends_to_nothing():
    shape, query, keys, values, table, lens = _pool([[0, -1]], [0], 2 * BLOCK_SIZE)
    attended = kernels.paged_decode_attention(query, keys, values, table, lens, shape)
    mx.eval(attended)

    assert float(mx.abs(attended).max()) == 0.0
    assert bool(mx.all(mx.isfinite(attended)).item())


@pytest.mark.skipif(build.shutil.which("xcrun") is None, reason="needs the Metal toolchain")
def test_generated_source_compiles_with_the_metal_toolchain():
    assert build.build(shape=_shape()) == ["paged_decode_attention", "store_kv"]
