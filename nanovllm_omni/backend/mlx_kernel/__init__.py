"""Self-authored Metal kernels for the MLX track (ADR 0021 / 0023).

This round implements decode only: ``paged_attention.metal`` holds the
source, ``kernels.py`` launches it, ``build.py`` gates it through
``xcrun metal``. Nothing here is wired into ``MlxStageRunner`` yet.
"""

from __future__ import annotations

from .kernels import (
    BLOCK_SIZE,
    DEFAULT_MAX_BLOCKS,
    DecodeShape,
    paged_decode_attention,
    source_fragments,
    store_kv,
)

__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_MAX_BLOCKS",
    "DecodeShape",
    "paged_decode_attention",
    "source_fragments",
    "store_kv",
]
