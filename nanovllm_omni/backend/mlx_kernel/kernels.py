"""The two Metal kernels the MLX track runs: paged decode attention + KV store.

``paged_attention.metal`` holds the bodies; this module supplies the
signatures ``mx.fast.metal_kernel`` asks for in Python and the per-step
launch. Nothing here is wired into ``MlxStageRunner`` yet.

The pool budget below is measured, not chosen. On an M2 Air a scatter of
k/v into a paged pool costs one full pool copy -- MLX has no in-place
scatter for an array that is still referenced -- so the write side is
priced by pool size alone:

    pool           write cost / layer     8 layers / step
    8 blocks  (1.5 MB)     0.06 ms            0.45 ms
    64 blocks (12 MB)      0.46 ms            3.7 ms
    256 blocks (48 MB)     1.48 ms           11.9 ms

A whole decode step is 2.3-4.0 ms, so a pool larger than 64 blocks spends
more on the cache than on the forward pass. ``DEFAULT_MAX_BLOCKS`` is that
64-block row: at block_size 256 it holds 16k tokens, which is the point
where paging stops paying for itself on this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

SOURCE_PATH = Path(__file__).with_name("paged_attention.metal")

DEFAULT_MAX_BLOCKS = 64
"""Pool ceiling from the measured budget above, in blocks of ``block_size``."""

BLOCK_SIZE = 256
"""Slots per block, matching the fork's ``Sequence.block_size``."""


def _split(source: str, begin: str, end: str) -> str:
    """Return the text between the ``begin``/``end`` marker lines."""
    start = source.index(begin) + len(begin)
    return source[start : source.index(end)].strip()


def source_fragments(path: Path = SOURCE_PATH) -> tuple[str, str, dict[str, str]]:
    """Split the ``.metal`` file into (header, signature note, kernel bodies)."""
    text = path.read_text()
    header = _split(text, "// HEADER_BEGIN\n", "// HEADER_END")
    bodies = {}
    for name in ("paged_decode_attention", "store_kv"):
        marker = f"// BODY_BEGIN {name}\n"
        bodies[name] = _split(text, marker, f"// BODY_END {name}")
    return header, text, bodies


def _make_kernels() -> tuple[object, object]:
    """Build both ``metal_kernel`` objects from the shared ``.metal`` source."""
    header, _, bodies = source_fragments()
    decode = mx.fast.metal_kernel(
        name="paged_decode_attention",
        input_names=["q", "k_cache", "v_cache", "block_table", "context_lens", "scale", "meta"],
        output_names=["out"],
        source=bodies["paged_decode_attention"],
        header=header,
    )
    store = mx.fast.metal_kernel(
        name="store_kv",
        input_names=["k", "v", "k_cache", "v_cache", "slot_mapping", "meta"],
        output_names=["k_out", "v_out"],
        source=bodies["store_kv"],
        header=header,
    )
    return decode, store


paged_decode_attention_kernel, store_kv_kernel = _make_kernels()


@dataclass(frozen=True)
class DecodeShape:
    """The model constants a decode launch bakes into the kernel."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = BLOCK_SIZE

    def template(self, dtype: object) -> list:
        """Template arguments: model constants as ints, plus the element dtype."""
        return [
            ("D", self.head_dim),
            ("NUM_HEADS", self.num_heads),
            ("NUM_KV_HEADS", self.num_kv_heads),
            ("BLOCK_SIZE", self.block_size),
            ("T", dtype),
        ]


def paged_decode_attention(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    shape: DecodeShape,
) -> mx.array:
    """Attend one query per sequence over the keys ``block_table`` names."""
    batch, num_heads, head_dim = q.shape
    scale = mx.array([head_dim**-0.5], dtype=mx.float32)
    meta = mx.array([block_table.shape[1]], dtype=mx.int32)
    return paged_decode_attention_kernel(
        inputs=[q, k_cache, v_cache, block_table, context_lens, scale, meta],
        template=shape.template(q.dtype),
        grid=(batch * num_heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def store_kv(
    k: mx.array,
    v: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    slot_mapping: mx.array,
    shape: DecodeShape,
) -> tuple[mx.array, mx.array]:
    """Return the pool with ``slot_mapping``'s rows replaced by ``k``/``v``."""
    pool_rows = k_cache.shape[0]
    meta = mx.array([pool_rows, slot_mapping.shape[0]], dtype=mx.int32)
    k_out, v_out = store_kv_kernel(
        inputs=[k, v, k_cache, v_cache, slot_mapping, meta],
        template=[
            ("D", shape.head_dim),
            ("NUM_KV_HEADS", shape.num_kv_heads),
            ("T", k.dtype),
        ],
        grid=(pool_rows * shape.num_kv_heads * shape.head_dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[k_cache.shape, v_cache.shape],
        output_dtypes=[k_cache.dtype, v_cache.dtype],
    )
    return k_out, v_out


__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_MAX_BLOCKS",
    "DecodeShape",
    "paged_decode_attention",
    "source_fragments",
    "store_kv",
]
