"""The MLX attention arms, measured on one MiniMind-O thinker decode step.

Three arms, same math, same shapes, one layer each:

- ``dense``: what the MLX backend runs today. The batch is left-padded to its
  longest member, and one masked SDPA covers the whole padded cache.
- ``paged``: our Metal kernel. A block table points into a KV pool, one query
  per sequence, and the scan covers ``context_len`` keys.
- ``bucketed``: dense again, but the batch is split into length buckets first,
  so every bucket pads to its own longest member.

The question these answer: how much padding waste does the dense arm pay
before the paged arm wins? ADR 0027 has the numbers.

Every constant input is built before the timed region, so the numbers are
kernel time and not graph-building time. The evaluation floor is measured
next to the results.

Scope: the attention subsystem only (KV write plus attention), not the whole
decode step. Every other layer of the model is identical in the three arms.

``mlx`` is a Mac-track extra, so it is imported inside the functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

BLOCK_SIZE = 256
"""Slots per block, matching the fork's ``Sequence.block_size``."""

POOL_CEILING_BLOCKS = 64
"""Pool ceiling from ADR 0023's decode attention scan cost, in blocks."""

LAYERS = 8
"""MiniMind-O thinker layers; the per-step cost is one layer times this."""

MEAN_LENGTH = 1024
"""Mean context length of the synthetic batch, in tokens."""

REAL_TOKENS = 256
"""Real content per sequence in the padding experiment."""

PADDED_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384]
"""Padded lengths of the padding experiment, in tokens."""

PATTERNS: dict[str, list[float]] = {
    "uniform": [1.0],
    "spread2": [0.75, 0.875, 1.0, 1.125, 1.25, 0.875, 1.0, 1.125],
    "spread4": [0.5, 0.75, 1.0, 1.5, 2.0, 0.625, 0.875, 1.25],
    "bimodal": [4.0, 0.25, 0.25, 0.375, 0.5, 0.25, 0.3125, 0.4375],
}
"""Length patterns, as multiples of ``MEAN_LENGTH``.

Every pattern keeps the mean near ``MEAN_LENGTH``, so the spread -- and
therefore the padding waste -- is the only thing that changes.
"""

BUCKET_RATIO = 1.25
"""A bucket takes the next sequence while its length stays within this ratio."""


@dataclass(frozen=True)
class Arm:
    """One arm's measured cost for a single decode step of one layer."""

    name: str
    attention_ms: float
    write_ms: float
    buckets: int = 1

    @property
    def total_ms(self) -> float:
        """Attention plus KV write, the cost one layer adds to a decode step."""
        return self.attention_ms + self.write_ms


def decode_shape() -> Any:
    """MiniMind-O's attention shape: 8 heads, 4 KV heads, head_dim 96."""
    from ..backend.mlx_kernel import DecodeShape

    return DecodeShape(num_heads=8, num_kv_heads=4, head_dim=96)


def lengths_for(pattern: str, batch: int) -> list[int]:
    """The context lengths of one batch under ``pattern``."""
    fractions = PATTERNS[pattern]
    lengths = []
    for index in range(batch):
        length = MEAN_LENGTH * fractions[index % len(fractions)]
        lengths.append(max(1, int(round(length))))
    return lengths


def padding_waste(lengths: list[int]) -> float:
    """Share of the padded cache the dense arm pays for and never uses."""
    longest = max(lengths)
    return 1.0 - sum(lengths) / (len(lengths) * longest)


def buckets_for(lengths: list[int]) -> list[list[int]]:
    """Split the batch into length buckets, each padded to its own longest."""
    buckets: list[list[int]] = []
    for length in sorted(lengths):
        if buckets and length <= buckets[-1][0] * BUCKET_RATIO:
            buckets[-1].append(length)
        else:
            buckets.append([length])
    return buckets


def _pool(lengths: list[int], shape: Any, seed: int):
    """A KV pool plus the block table that gives each sequence its rows."""
    import mlx.core as mx

    blocks_per_sequence = [max(1, -(-length // BLOCK_SIZE)) for length in lengths]
    pool_blocks = sum(blocks_per_sequence)
    mx.random.seed(seed)
    keys = mx.random.normal((pool_blocks * BLOCK_SIZE, shape.num_kv_heads, shape.head_dim)).astype(
        mx.float16
    )
    values = mx.random.normal(
        (pool_blocks * BLOCK_SIZE, shape.num_kv_heads, shape.head_dim)
    ).astype(mx.float16)
    table: list[list[int]] = []
    next_block = 0
    for count in blocks_per_sequence:
        table.append(list(range(next_block, next_block + count)))
        next_block += count
    width = max(len(row) for row in table)
    padded = [row + [-1] * (width - len(row)) for row in table]
    mx.eval(keys, values)
    return keys, values, padded, mx.array(lengths, dtype=mx.int32)


def _dense_view(pool_keys, pool_values, lengths, table_rows, shape):
    """The dense view of a pool: the same keys, right-aligned and left-padded."""
    import mlx.core as mx

    batch = len(lengths)
    longest = max(lengths)
    keys = mx.zeros((batch, shape.num_kv_heads, longest, shape.head_dim), mx.float16)
    values = mx.zeros((batch, shape.num_kv_heads, longest, shape.head_dim), mx.float16)
    for row in range(batch):
        left = longest - lengths[row]
        positions = mx.arange(lengths[row])
        blocks = mx.array(table_rows[row], dtype=mx.int32)[positions // BLOCK_SIZE]
        slots = blocks * BLOCK_SIZE + positions % BLOCK_SIZE
        keys[row, :, left : left + lengths[row]] = pool_keys[slots].transpose(1, 0, 2)
        values[row, :, left : left + lengths[row]] = pool_values[slots].transpose(1, 0, 2)
    mx.eval(keys, values)
    return keys, values


def _dense_mask(lengths: list[int], longest: int):
    """The keep-mask of the dense arm, built once per configuration."""
    import mlx.core as mx

    left = mx.array([longest - length for length in lengths])[:, None]
    positions = mx.arange(longest)[None, :]
    return positions >= left


def _dense_attention_call(query, keys, values, keep, shape):
    """One masked SDPA over the padded cache; the arm the MLX backend runs."""
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(
        query[:, :, None, :],
        keys,
        values,
        scale=shape.head_dim**-0.5,
        mask=keep[:, None, None, :],
    )


def _paged_attention_call(query, pool_keys, pool_values, table, lens, shape):
    """One paged attention call, with every constant input already built."""
    import mlx.core as mx

    from ..backend.mlx_kernel import kernels

    scale = mx.array([shape.head_dim**-0.5], dtype=mx.float32)
    meta = mx.array([table.shape[1]], dtype=mx.int32)
    batch, num_heads, _ = query.shape
    return kernels.paged_decode_attention_kernel(
        inputs=[query, pool_keys, pool_values, table, lens, scale, meta],
        template=shape.template(query.dtype),
        grid=(batch * num_heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[query.dtype],
    )[0]


def _time(call, runs: int, warmup: int) -> float:
    """Best wall of one call, in milliseconds.

    The minimum is the least noisy estimate on a shared Mac: a run that loses
    the core to another process only makes the number larger.
    """
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(call())
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        mx.eval(call())
        samples.append((time.perf_counter() - start) * 1000.0)
    return min(samples)


def eval_floor(runs: int, warmup: int) -> float:
    """Cost of one measured call when the call does nothing."""
    import mlx.core as mx

    tiny = mx.zeros((1,), mx.float32)
    mx.eval(tiny)
    return _time(lambda: tiny, runs=runs, warmup=warmup)


def measure_dense(lengths: list[int], shape: Any, runs: int, warmup: int) -> Arm:
    """The dense arm: one SDPA over the padded cache, plus one token write."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    pool_keys, pool_values, table_rows, _ = _pool(lengths, shape, seed=0)
    keys, values = _dense_view(pool_keys, pool_values, lengths, table_rows, shape)
    keep = _dense_mask(lengths, keys.shape[2])
    query = mx.random.normal((len(lengths), shape.num_heads, shape.head_dim)).astype(mx.float16)
    mx.eval(query)

    attention_ms = _time(
        lambda: _dense_attention_call(query, keys, values, keep, shape), runs=runs, warmup=warmup
    )

    # The dense write is one token appended to a left-padded BatchKVCache:
    # the path mlx_lm takes on every decode step.
    left_padding = [max(lengths) - length for length in lengths]
    cache = BatchKVCache(left_padding)
    cache.update_and_fetch(keys, values)
    fresh = mx.zeros((len(lengths), shape.num_kv_heads, 1, shape.head_dim), mx.float16)
    mx.eval(fresh)
    write_ms = _time(lambda: cache.update_and_fetch(fresh, fresh), runs=runs, warmup=warmup)
    return Arm(name="dense", attention_ms=attention_ms, write_ms=write_ms)


def measure_bucketed(lengths: list[int], shape: Any, runs: int, warmup: int) -> Arm:
    """The bucketed arm: dense per bucket, so each bucket pads to its own longest."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    buckets = buckets_for(lengths)
    prepared = []
    for index, bucket in enumerate(buckets):
        pool_keys, pool_values, table_rows, _ = _pool(bucket, shape, seed=index)
        keys, values = _dense_view(pool_keys, pool_values, bucket, table_rows, shape)
        keep = _dense_mask(bucket, keys.shape[2])
        query = mx.random.normal((len(bucket), shape.num_heads, shape.head_dim)).astype(mx.float16)
        mx.eval(query)
        prepared.append((bucket, keys, values, query, keep))

    attention_ms = 0.0
    for item in prepared:
        attention_ms += _time(
            lambda item=item: _dense_attention_call(item[3], item[1], item[2], item[4], shape),
            runs=runs,
            warmup=warmup,
        )
    write_ms = 0.0
    for item in prepared:
        bucket = item[0]
        cache = BatchKVCache([max(bucket) - length for length in bucket])
        cache.update_and_fetch(item[1], item[2])
        fresh = mx.zeros((len(bucket), shape.num_kv_heads, 1, shape.head_dim), mx.float16)
        mx.eval(fresh)
        write_ms += _time(
            lambda cache=cache, fresh=fresh: cache.update_and_fetch(fresh, fresh),
            runs=runs,
            warmup=warmup,
        )
    return Arm(name="bucketed", attention_ms=attention_ms, write_ms=write_ms, buckets=len(buckets))


def measure_paged(lengths: list[int], shape: Any, runs: int, warmup: int, pool_blocks: int) -> Arm:
    """The paged arm: our Metal kernel over a pool, plus the pool write."""
    import mlx.core as mx

    from ..backend.mlx_kernel import kernels

    pool_keys, pool_values, table_rows, lens = _pool(lengths, shape, seed=0)
    table = mx.array(table_rows, dtype=mx.int32)
    query = mx.random.normal((len(lengths), shape.num_heads, shape.head_dim)).astype(mx.float16)
    mx.eval(query)

    attention_ms = _time(
        lambda: _paged_attention_call(query, pool_keys, pool_values, table, lens, shape),
        runs=runs,
        warmup=warmup,
    )

    pool_rows = pool_blocks * BLOCK_SIZE
    write_keys = mx.zeros((pool_rows, shape.num_kv_heads, shape.head_dim), mx.float16)
    write_values = mx.zeros((pool_rows, shape.num_kv_heads, shape.head_dim), mx.float16)
    fresh_keys = mx.zeros((len(lengths), shape.num_kv_heads, shape.head_dim), mx.float16)
    fresh_values = mx.zeros_like(fresh_keys)
    slots = mx.array(list(range(len(lengths))), dtype=mx.int32)
    mx.eval(write_keys, write_values, fresh_keys, fresh_values, slots)
    write_ms = _time(
        lambda: kernels.store_kv(fresh_keys, fresh_values, write_keys, write_values, slots, shape),
        runs=runs,
        warmup=warmup,
    )
    return Arm(name="paged", attention_ms=attention_ms, write_ms=write_ms)


def check_agreement(lengths: list[int], shape: Any) -> float:
    """Max absolute difference between the dense arm and the paged arm."""
    import mlx.core as mx

    pool_keys, pool_values, table_rows, lens = _pool(lengths, shape, seed=1)
    keys, values = _dense_view(pool_keys, pool_values, lengths, table_rows, shape)
    keep = _dense_mask(lengths, keys.shape[2])
    query = mx.random.normal((len(lengths), shape.num_heads, shape.head_dim)).astype(mx.float16)
    mx.eval(query)
    dense = _dense_attention_call(query, keys, values, keep, shape)[:, :, 0, :]

    table = mx.array(table_rows, dtype=mx.int32)
    paged = _paged_attention_call(query, pool_keys, pool_values, table, lens, shape)
    mx.eval(dense, paged)
    difference = mx.abs(dense.astype(mx.float32) - paged.astype(mx.float32))
    return float(difference.max())


def padding_experiment(shape: Any, batch: int, runs: int, warmup: int) -> list[dict]:
    """What the dense arm pays for padding: same content, growing padded length."""
    import mlx.core as mx

    rows = []
    for padded in PADDED_LENGTHS:
        lengths = [REAL_TOKENS] * batch
        pool_keys, pool_values, table_rows, _ = _pool(lengths, shape, seed=0)
        keys = mx.zeros((batch, shape.num_kv_heads, padded, shape.head_dim), mx.float16)
        values = mx.zeros((batch, shape.num_kv_heads, padded, shape.head_dim), mx.float16)
        for row in range(batch):
            left = padded - REAL_TOKENS
            positions = mx.arange(REAL_TOKENS)
            blocks = mx.array(table_rows[row], dtype=mx.int32)[positions // BLOCK_SIZE]
            slots = blocks * BLOCK_SIZE + positions % BLOCK_SIZE
            keys[row, :, left:padded] = pool_keys[slots].transpose(1, 0, 2)
            values[row, :, left:padded] = pool_values[slots].transpose(1, 0, 2)
        keep = _dense_mask(lengths, padded)
        query = mx.random.normal((batch, shape.num_heads, shape.head_dim)).astype(mx.float16)
        mx.eval(keys, values, query)

        attention_ms = _time(
            lambda query=query, keys=keys, values=values, keep=keep: _dense_attention_call(
                query, keys, values, keep, shape
            ),
            runs=runs,
            warmup=warmup,
        )
        rows.append(
            {
                "padded_tokens": padded,
                "real_tokens": REAL_TOKENS,
                "waste_pct": round((1.0 - REAL_TOKENS / padded) * 100, 1),
                "dense_attn_ms": round(attention_ms, 3),
                "step_dense_ms": round(attention_ms * LAYERS, 3),
            }
        )
    return rows


__all__ = [
    "Arm",
    "BUCKET_RATIO",
    "LAYERS",
    "MEAN_LENGTH",
    "PADDED_LENGTHS",
    "PATTERNS",
    "POOL_CEILING_BLOCKS",
    "REAL_TOKENS",
    "buckets_for",
    "check_agreement",
    "decode_shape",
    "eval_floor",
    "lengths_for",
    "measure_bucketed",
    "measure_dense",
    "measure_paged",
    "padding_experiment",
    "padding_waste",
]
