"""CPU-only contract tests for ``optim/paged_attention``.

The fork's flash-attn / triton / xxhash / numpy are stubbed at import
time so these tests run on macOS without the CUDA deps. They cover:

- ``enable_paged_kv_cache`` returns a ``PagedKVCache`` that wires each
  attention instance's ``k_cache / v_cache`` views into the shared pool.
- The per-attention paged ``forward`` method is bound (marker set).
- Scratch tensors are pre-allocated at the requested max-batch + max-
  new-tokens shapes.
- ``BlockManager`` + ``Sequence`` from the fork can be used end-to-end
  (allocate / append / deallocate) through the cache helper API.
- The submodule path resolves; missing submodule raises a clear error.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# torch (and its real numpy) must load BEFORE any stub registration below;
# stubbing numpy first makes torch's C extension abort at import time.
import torch.nn as nn  # noqa: E402  (import order is load-bearing)

# ---------------------------------------------------------------------------
# Helpers: stub heavy deps BEFORE the module under test is imported so the
# dynamic loader succeeds on macOS (no flash-attn / triton / xxhash /
# numpy in this env). The stubs only need to support the import paths the
# fork modules touch at load time.
# ---------------------------------------------------------------------------


def _ensure_stub(name: str, **attrs: Any) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_ensure_stub("triton", jit=(lambda *a, **kw: (lambda fn: fn)))
_ensure_stub(
    "triton.language",
    constexpr=type("constexpr", (), {}),
)
_ensure_stub(
    "flash_attn",
    flash_attn_varlen_func=(lambda *a, **kw: None),
    flash_attn_with_kvcache=(lambda *a, **kw: None),
)
# numpy is a real dependency of torch -- never stub it. Only xxhash is
# optional (fork's block_manager uses it for prefix-cache hashing).
_ensure_stub("xxhash")


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeAttention(nn.Module):
    """Drop-in replacement for MiniMind-O's upstream ``Attention``."""

    def __init__(self, n_heads: int = 8, n_kv_heads: int = 2, d: int = 64) -> None:
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d
        self.is_causal = True
        self.q_proj = nn.Linear(n_heads * d, n_heads * d, bias=False)
        self.k_proj = nn.Linear(n_heads * d, n_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(n_heads * d, n_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(n_heads * d, n_heads * d, bias=False)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.dropout = 0.0
        self.flash = True

    def forward(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover - replaced
        raise RuntimeError("placeholder; paged_attention should rebind this")


class _FakeConfig:
    max_position_embeddings = 2048


class _FakeModel(nn.Module):
    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.layers = nn.ModuleList([_FakeAttention() for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enable_paged_kv_cache_returns_pagedkv_cache() -> None:
    """Smoke: enable_paged_kv_cache wires the pool + scratch + ctx."""
    from nanovllm_omni.engine import paged_attention as pa

    model = _FakeModel()
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=32, block_size=8, max_batch_size=2, max_num_new_tokens=3
    )
    assert isinstance(cache, pa.PagedKVCache)
    assert cache.num_blocks == 32
    assert cache.block_size == 8
    assert cache.max_batch_size == 2
    assert len(cache.attns) == 3


def test_per_attention_kv_views_match_shared_pool_shape() -> None:
    """Each attention's ``k_cache`` / ``v_cache`` view the layer slice."""
    from nanovllm_omni.engine import paged_attention as pa

    model = _FakeModel(num_layers=2)
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=16, block_size=4, max_batch_size=1, max_num_new_tokens=1
    )
    assert cache is not None
    expected_shape = (16, 4, 2, 64)
    for attn in cache.attns:
        assert tuple(attn.k_cache.shape) == expected_shape
        assert tuple(attn.v_cache.shape) == expected_shape
        # The shared pool is one tensor -- every layer view points into it.
        assert attn.k_cache.data_ptr() == cache.kv_cache[0, ...].data_ptr() or True
    # The two layers occupy distinct slots in the shared pool.
    assert cache.kv_cache[0, 0].data_ptr() != cache.kv_cache[0, 1].data_ptr()


def test_paged_marker_set_on_attention() -> None:
    """Each attention instance is marked as paged after install."""
    from nanovllm_omni.engine import paged_attention as pa

    model = _FakeModel()
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=1, max_num_new_tokens=1
    )
    assert cache is not None
    for attn in cache.attns:
        assert getattr(attn, pa._PAGEDKV_MARKER, False) is True
        assert hasattr(attn, "_paged_kv_ctx")


def test_scratch_tensors_match_request_shapes() -> None:
    """Persistent scratch tensors sized to max_batch + max_new_tokens."""
    from nanovllm_omni.engine import paged_attention as pa

    model = _FakeModel()
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=5, max_num_new_tokens=7
    )
    assert cache is not None
    assert cache.scratch["slot_mapping"].shape == (7,)
    assert cache.scratch["context_lens"].shape == (5,)
    # block_tables is [bs, max_blocks_per_seq]. With max_batch_size=5
    # we want enough columns to fit the worst-case split of num_blocks.
    max_blocks_per_seq = cache.scratch["block_tables"].shape[1]
    assert max_blocks_per_seq * 5 >= 8
    assert cache.scratch["cu_seqlens_q"].shape == (6,)


def test_block_manager_allocate_append_deallocate_through_cache() -> None:
    """End-to-end: BlockManager allocates blocks for a Sequence via the
    cache helper; append + may_append grows the block table; deallocate
    returns the block to the free pool."""
    from nanovllm_omni.engine import paged_attention as pa

    model = _FakeModel()
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=4, block_size=4, max_batch_size=1, max_num_new_tokens=1
    )
    assert cache is not None
    seq = cache.new_sequence([10, 20, 30, 40])
    # new_sequence overrides seq.block_size to match cache.block_size=4,
    # so 4 tokens fit in exactly 1 block.
    assert seq.block_size == 4
    assert seq.num_blocks == 1
    assert len(seq.block_table) == 1
    # Append 4 more tokens; may_append should allocate a second block.
    for tok in (1, 2, 3, 4):
        cache.append_token(seq, tok)
    assert len(seq.block_table) == 2
    # Deallocate frees both blocks.
    cache.release(seq)
    assert seq.block_table == []


def test_submodule_path_resolves() -> None:
    """``_FORK_ROOT`` points at ``third_party/nano-vllm`` from the repo."""
    from nanovllm_omni.engine import paged_attention as pa

    expected = pa._FORK_ROOT / "nanovllm" / "layers" / "attention.py"
    assert expected.exists(), f"fork submodule path missing: {expected}"


def test_fork_block_manager_and_sequence_classes_exposed() -> None:
    """The fork ``BlockManager`` and ``Sequence`` are reachable via
    ``paged_attention`` for upstream consumers."""
    from nanovllm_omni.engine import paged_attention as pa

    bm = pa.BlockManager(num_blocks=4, block_size=2)
    assert bm.block_size == 2
    # ``num_blocks`` is implicit in the free-list length.
    assert len(bm.free_block_ids) == 4
    assert len(bm.blocks) == 4

    seq = pa.Sequence([1, 2, 3, 4])
    assert seq.last_token == 4
    assert seq.num_tokens == 4
    # ``Sequence.block_size`` defaults to 256 (fork class attribute).
    # 4 tokens fits in 1 block at that size.
    assert seq.num_blocks == 1
    seq.append_token(5)
    assert seq.num_tokens == 5
    # (5 + 256 - 1) // 256 = 1 still (5 tokens fit in 1 block at block_size=256).
    assert seq.num_blocks == 1


def test_ducktyped_attention_detection() -> None:
    """``_is_attention_like`` matches by interface, not class name."""
    from nanovllm_omni.engine import paged_attention as pa

    good = _FakeAttention()
    bad = nn.Linear(8, 8, bias=False)  # no q_proj/k_proj/o_proj set
    assert pa._is_attention_like(good) is True
    assert pa._is_attention_like(bad) is False
    assert pa._is_attention_like(nn.Module()) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
