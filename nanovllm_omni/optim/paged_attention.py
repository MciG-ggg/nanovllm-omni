from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

"""Paged-KV attention via the MciG-ggg/nano-vllm fork (submodule).

This module is the **position-independent successor** to
``optim/attention.enable_fixed_kv_buffer``. The fork is the foundation:
``BlockManager`` allocates block IDs from a shared pool, ``Sequence``
tracks per-request block tables, and each attention layer's
``module.k_cache / module.v_cache`` are views into a single
``[2, num_layers, num_blocks, block_size, n_kv, d]`` tensor (K on axis 0,
V on axis 1). ``flash_attn_with_kvcache`` reads the relevant history
through ``block_table + context_lens`` so the K/V tensor shape is
constant across AR steps -- a single CUDA Graph serves every position
of a request, and per-bucket graphs handle batching.

Why this is needed
------------------
``enable_fixed_kv_buffer`` uses a contiguous ``[B, max_len, n_kv, d]``
buffer and ``[:, : _kv_pos]`` slicing for the KV history. CUDA Graphs
freeze the K/V tensor shapes at capture time, so each AR position needs
its own per-step graph (``n_steps - 1`` graphs, all paying warmup +
capture + VRAM). The fork's paged cache keeps K/V shape constant
across AR steps: every new token writes one slot via ``slot_mapping``
into a ``[num_blocks, block_size, n_kv, d]`` pool, and the SDPA backend
reads the relevant tokens through ``block_table`` + ``context_lens``.
A single captured graph then serves every AR step of a request, and a
small set of bucket graphs (keyed on actual batch size) serves a
batched serve loop.

Batched serving, not single-request
-----------------------------------
We deliberately do **not** constrain the design to B=1. ``enable_paged_kv_cache``
allocates a shared block pool sized for ``gpu_memory_utilization`` worth
of KV; ``BlockManager`` hands out block IDs to ``Sequence`` instances,
and a single ``prepare_decode`` call lays out ``slot_mapping[B]``,
``context_lens[B]``, ``block_tables[B, max_blocks]`` for a batched
forward. ``Omni.generate`` is still B=1 today, but the paged layer
already speaks ``B>=1`` so adding a scheduler that batches multiple
generate() calls requires no attention changes.

Submodule strategy
------------------
The fork is pinned at ``third_party/nano-vllm/`` (see ``.gitmodules``).
We **dynamically load** the three files we need
(``nanovllm.layers.attention`` + ``nanovllm.engine.block_manager`` +
``nanovllm.engine.sequence``) via ``importlib`` so that macOS-side
imports don't require ``flash_attn`` / ``triton`` -- those deps only
need to be installed on the WSL inference box. The fork's
``store_kvcache`` Triton kernel + ``Attention`` SDPA helpers are re-used
directly; the surrounding plumbing (per-layer pool wiring, batched
Context, AR-step block management) is written here.

Bridged contracts
-----------------
- ``enable_paged_kv_cache(model, num_blocks, block_size)`` allocates
  the shared ``[2, num_layers, num_blocks, block_size, n_kv, d]`` pool
  and rebinds every attention instance's ``forward`` to the paged
  version. Returns ``None`` when the model has no compatible
  attention instances or the submodule is not importable.
- The rewritten ``attn.forward(x, position_embeddings, past_key_value=None,
  use_cache=False, attention_mask=None)`` keeps the upstream signature so
  the upstream ``MiniMindBlock`` calls it without changes.
- ``use_cache=True`` returns ``(output, None)`` -- the actual KV lives
  in the shared block pool; ``past_key_value=None`` is the contract the
  graphed decoder already relies on.
- ``PagedKVCache.runtime(seqs)`` builds a ``PagedKVContext`` per
  forward call, suitable for graph capture.
"""
# Per-instance marker: this attention has a paged KV cache installed.
_PAGEDKV_MARKER = "_nanovllm_paged_kv"

# Submodule path -- third_party/nano-vllm/.
_FORK_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "nano-vllm"


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """Register a stub ``sys.modules[name]`` with the given attrs so an
    import path resolves without touching the real package."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_fork_module(
    short_name: str,
    rel_path: str,
    extra_stubs: dict[str, Any] | None = None,
) -> Any:
    """Load a fork module via ``importlib.util`` so we don't go through
    ``nanovllm/__init__.py`` (which pulls in transformers + LLMEngine and
    is not importable on macOS without the full vLLM env).

    Stubs heavy deps first so the source parses + key symbols exist even
    when the CUDA deps are absent. Kernel calls are only made at
    runtime; on macOS we never call them, the module just needs to
    import.
    """
    if short_name in sys.modules:
        return sys.modules[short_name]

    # 1) Stub triton + flash_attn if absent.
    if "triton" not in sys.modules:
        _stub_module("triton", jit=(lambda *a, **kw: (lambda fn: fn)))
    if "triton.language" not in sys.modules:
        _stub_module("triton.language", constexpr=type("constexpr", (), {}))
    if "flash_attn" not in sys.modules:
        _stub_module(
            "flash_attn",
            flash_attn_varlen_func=(lambda *a, **kw: None),
            flash_attn_with_kvcache=(lambda *a, **kw: None),
        )

    # 2) Stub nanovllm package tree so cross-module imports resolve.
    if "nanovllm" not in sys.modules:
        pkg = _stub_module("nanovllm")
        pkg.__path__ = [str(_FORK_ROOT / "nanovllm")]  # type: ignore[attr-defined]
    if "nanovllm.utils" not in sys.modules:
        utils_pkg = _stub_module("nanovllm.utils")
        utils_pkg.__path__ = [str(_FORK_ROOT / "nanovllm" / "utils")]  # type: ignore[attr-defined]
    if "nanovllm.engine" not in sys.modules:
        eng_pkg = _stub_module("nanovllm.engine")
        eng_pkg.__path__ = [str(_FORK_ROOT / "nanovllm" / "engine")]  # type: ignore[attr-defined]

    # 3) Per-file stubs before exec_module. Only register stubs for
    #    modules that don't already exist so we don't clobber a real
    #    torch-installed numpy / xxhash / sampling_params.
    def _maybe_stub(name: str, attrs: dict[str, Any]) -> None:
        if name not in sys.modules:
            _stub_module(name, **attrs)

    if extra_stubs:
        for full_name, attrs in extra_stubs.items():
            _maybe_stub(full_name, attrs)
    _maybe_stub("xxhash", {})
    _maybe_stub("numpy", {})

    spec = importlib.util.spec_from_file_location(short_name, str(_FORK_ROOT / rel_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"paged_attention: cannot load fork module {rel_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[short_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # 4) Patch the just-loaded module's stubs to expose the minimum
    #    surface the fork modules reference (numpy.array / xxhash.xxh64).
    if "numpy" in sys.modules:
        np = sys.modules["numpy"]
        if not hasattr(np, "array"):
            np.array = lambda *a, **kw: b""  # type: ignore[attr-defined]
        if not hasattr(np, "asarray"):
            np.asarray = lambda *a, **kw: b""  # type: ignore[attr-defined]
    if "xxhash" in sys.modules:
        xxhash_mod = sys.modules["xxhash"]

        class _StubHash:
            def __init__(self, *a: Any, **kw: Any) -> None: ...
            def update(self, *a: Any, **kw: Any) -> None: ...
            def intdigest(self) -> int:
                return 0

        if not hasattr(xxhash_mod, "xxh64"):
            def _xxh64(*a: Any, **kw: Any) -> _StubHash:
                return _StubHash()

            xxhash_mod.xxh64 = _xxh64  # type: ignore[attr-defined]

    return mod


# Eagerly resolve at import time so the rest of the module can name
# these symbols directly. If the submodule is missing (e.g.
# ``git submodule update --init`` was skipped), surface a clear error
# pointing at .gitmodules.
try:
    _fork_attention = _load_fork_module(
        "nanovllm_fork_layers_attention",
        "nanovllm/layers/attention.py",
    )
    # block_manager and sequence both reference SamplingParams at import.
    # Build a stub class once and inject it under ``nanovllm.sampling_params``
    # so ``from nanovllm.sampling_params import SamplingParams`` resolves.

    class _StubSamplingParams:
        temperature: float = 1.0
        max_tokens: int = 64
        ignore_eos: bool = False

        def __post_init__(self) -> None: ...

    if "nanovllm.sampling_params" not in sys.modules:
        _stub_module("nanovllm.sampling_params", SamplingParams=_StubSamplingParams)
    else:
        sys.modules["nanovllm.sampling_params"].SamplingParams = _StubSamplingParams

    _fork_block_manager = _load_fork_module(
        "nanovllm_fork_engine_block_manager",
        "nanovllm/engine/block_manager.py",
    )
    _fork_sequence = _load_fork_module(
        "nanovllm_fork_engine_sequence",
        "nanovllm/engine/sequence.py",
    )
except (FileNotFoundError, ModuleNotFoundError) as exc:
    raise RuntimeError(
        "paged_attention requires the MciG-ggg/nano-vllm submodule; run "
        "`git submodule update --init --recursive` from the repo root."
    ) from exc

store_kvcache = _fork_attention.store_kvcache
store_kvcache_kernel = _fork_attention.store_kvcache_kernel
ForkAttention = _fork_attention.Attention
BlockManager = _fork_block_manager.BlockManager
Sequence = _fork_sequence.Sequence


class PagedKVContext:
    """Per-forward-call batched metadata, fork-style.

    Holds CUDA Graph-stable input tensors (``slot_mapping``,
    ``context_lens``, ``block_tables``). All fields are pre-allocated
    once at capture time; only ``.fill_`` + slice ``[:bs]`` happen per
    replay. ``cu_seqlens_*`` / ``max_seqlen_*`` are prefill-only.

    ``block_table`` is a ``[B, max_blocks]`` int32 tensor. ``slot_mapping``
    is a ``[num_new_tokens]`` int32 tensor: ``slot_mapping[t]`` is the
    flat offset into the per-layer pool for token t. ``context_lens``
    is ``[B]`` int32: number of K/V tokens each request attends to.
    """

    __slots__ = (
        "is_prefill",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "slot_mapping",
        "context_lens",
        "block_tables",
    )

    def __init__(
        self,
        *,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        is_prefill: bool = False,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
    ) -> None:
        self.is_prefill = is_prefill
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables


class PagedKVCache:
    """Owns the paged KV block pool across every attention instance.

    Allocation model (mirrors fork ``ModelRunner.allocate_kv_cache``):
    one ``[2, num_layers, num_blocks, block_size, n_kv, d]`` tensor
    holds every layer's K (axis 0) and V (axis 1). Each attention
    instance's ``module.k_cache / module.v_cache`` becomes a view into
    its layer's slot. ``BlockManager`` hands out block IDs that share
    the same numerical meaning across every layer (block ID 5 in layer 0
    corresponds to the same logical block as block ID 5 in layer 1).
    """

    def __init__(
        self,
        model: Any,
        num_blocks: int,
        block_size: int,
        max_batch_size: int = 1,
    ) -> None:

        self.model = model
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.attns: list[Any] = []
        # K and V are stored as views into a single ``[2, L, P, B, H, D]``
        # tensor so the allocator walks one memory range. The 2-axis
        # shape mirrors fork's ``kv_cache[2, num_layers, num_blocks,
        # block_size, n_kv, d]`` layout.
        self.kv_cache: torch.Tensor | None = None
        # Per-forward metadata scratch tensors (CUDA-Graph-friendly
        # addresses).
        self.scratch: dict[str, torch.Tensor] = {}
        # Batched BlockManager (one global pool, shared across all
        # layers via block IDs). Initialized in ``enable_paged_kv_cache``.
        self.block_manager: BlockManager | None = None
        # Active batch's Sequence list, kept on the cache for the
        # graphed decoder's AR-step bookkeeping.
        self.active_seqs: list[Sequence] = []

    def init_pool(self, attns: list[Any], max_blocks: int, block_size: int, dtype: Any, device: Any) -> torch.Tensor:
        """Allocate the ``[2, num_layers, num_blocks, block_size, n_kv, d]``
        pool and bind every attention instance's ``k_cache / v_cache`` to
        its layer view."""
        import torch

        n_kv = attns[0].n_local_kv_heads
        d = attns[0].head_dim
        num_layers = len(attns)
        self.kv_cache = torch.zeros(
            2, num_layers, max_blocks, block_size, n_kv, d,
            dtype=dtype, device=device,
        )
        for layer_id, attn in enumerate(attns):
            attn.k_cache = self.kv_cache[0, layer_id]  # [num_blocks, block_size, n_kv, d]
            attn.v_cache = self.kv_cache[1, layer_id]
        return self.kv_cache

    def init_scratch(
        self,
        max_batch_size: int,
        max_blocks_per_seq: int,
        max_num_new_tokens: int,
        device: Any,
    ) -> None:
        """Pre-allocate persistent metadata tensors. Reused across
        requests; only ``.fill_ + slice`` happens per forward."""
        import torch

        self.scratch["slot_mapping"] = torch.full(
            (max_num_new_tokens,), -1, dtype=torch.int32, device=device,
        )
        self.scratch["context_lens"] = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device,
        )
        self.scratch["block_tables"] = torch.zeros(
            max_batch_size, max_blocks_per_seq, dtype=torch.int32, device=device,
        )
        # Prefill-only scratch -- sized for the largest prompt we'll ever
        # see. ``cu_seqlens_*`` start with 0 and end with
        # ``sum(prompt_lens)``; we keep them as ``[max_batch_size + 1]``
        # to cover arbitrary batching.
        self.scratch["cu_seqlens_q"] = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device,
        )
        self.scratch["cu_seqlens_k"] = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device,
        )

    def new_sequence(self, token_ids: list[int], block_size: int | None = None) -> Sequence:
        """Allocate a new ``Sequence`` with blocks from the BlockManager.

        Fork's ``Sequence`` has a class-level ``block_size = 256``; we
        override per-instance to match the cache's block_size so prefix
        hashing and ``num_blocks`` agree with the BlockManager.
        """
        block_size = block_size or self.block_size
        seq = Sequence(token_ids)
        seq.block_size = block_size
        # Compute num_cached_blocks (prefix cache hit count) + allocate.
        num_cached = self.block_manager.can_allocate(seq)  # type: ignore[union-attr]
        if num_cached < 0:
            raise RuntimeError(
                f"PagedKVCache: not enough blocks for seq of length "
                f"{len(token_ids)} (have {len(self.block_manager.free_block_ids)} free)"  # type: ignore[union-attr]
            )
        self.block_manager.allocate(seq, num_cached)  # type: ignore[union-attr]
        return seq

    def append_token(self, seq: Sequence, token_id: int) -> None:
        """Append a decoded token to ``seq``; BlockManager may allocate a
        new block if we crossed a block boundary."""
        seq.append_token(token_id)
        self.block_manager.may_append(seq)  # type: ignore[union-attr]

    def release(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)  # type: ignore[union-attr]


def _paged_attention_forward(
    self: Any,
    x: Any,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any = None,
    use_cache: bool = False,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    """Rewrite of upstream ``Attention.forward`` for paged KV.

    Mirrors ``optim/attention._kv_buffer_forward`` for the projection
    head (q/k/v projections + q_norm/k_norm + RoPE) and then defers the
    SDPA backend to ``flash_attn_varlen_func`` (prefill) or
    ``flash_attn_with_kvcache`` (decode). KV storage is the shared
    ``[num_blocks, block_size, n_kv, d]`` block pool bound as
    ``self.k_cache / self.v_cache``.

    Returns ``(output, None)`` -- the actual KV lives in the block pool;
    we never propagate past_key_value.
    """
    import math

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    module = __import__(type(self).__module__, fromlist=["apply_rotary_pos_emb"])
    batch_size, sequence_len, _ = x.shape

    # 1. Projections + norms + RoPE.
    query = self.q_proj(x).view(batch_size, sequence_len, self.n_local_heads, self.head_dim)
    key = self.k_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    value = self.v_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    query, key = self.q_norm(query), self.k_norm(key)
    query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)

    ctx = self._paged_kv_ctx
    k_cache = self.k_cache  # [num_blocks, block_size, n_kv, d]
    v_cache = self.v_cache

    if ctx.is_prefill:
        # Prefill path: write all new tokens into the block pool via
        # fork's ``store_kvcache`` Triton kernel, then run
        # ``flash_attn_varlen_func`` against the cache.
        store_kvcache(key, value, k_cache, v_cache, ctx.slot_mapping)
        # flash_attn_varlen_func expects q/k/v shaped [N, H, D] (flat).
        q_flat = query.reshape(-1, self.n_local_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        out = flash_attn_varlen_func(
            q_flat,
            k_cache,
            v_cache,
            max_seqlen_q=ctx.max_seqlen_q,
            cu_seqlens_q=ctx.cu_seqlens_q,
            max_seqlen_k=ctx.max_seqlen_k,
            cu_seqlens_k=ctx.cu_seqlens_k,
            softmax_scale=scale,
            causal=self.is_causal,
            block_table=ctx.block_tables,
        )
        # out: [N, H, D] -> [B, seq, H, D]
        out = out.view(batch_size, sequence_len, self.n_local_heads, self.head_dim)
    else:
        # Decode path: write one new token per request via store_kvcache,
        # then call flash_attn_with_kvcache which reads the relevant
        # history through block_table + context_lens.
        store_kvcache(key, value, k_cache, v_cache, ctx.slot_mapping)
        # q: [B, 1, H, D] -- flash_attn expects 4D
        q4d = query.view(batch_size, sequence_len, self.n_local_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        out = flash_attn_with_kvcache(
            q4d,
            k_cache,
            v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            softmax_scale=scale,
            causal=self.is_causal,
        )
        # out: [B, 1, H, D] -> [B, 1, H, D] (already correct shape)
        out = out.view(batch_size, sequence_len, self.n_local_heads, self.head_dim)

    output = out.reshape(batch_size, sequence_len, -1)
    output = self.resid_dropout(self.o_proj(output))
    return output, None


def _attach_paged_kv(attn: Any) -> None:
    """Rewrite the upstream ``Attention.forward`` to the paged version.

    Per-instance marker; idempotent. Reads ``self.k_cache`` /
    ``self.v_cache`` and ``self._paged_kv_ctx`` at forward time -- both
    are bound by ``enable_paged_kv_cache`` before any forward runs.
    """
    if getattr(attn, _PAGEDKV_MARKER, False):
        return
    # Match by duck-typed interface, not class name. MiniMind-O's
    # ``Attention`` (in vendored ``model_minimind.py``) has the full set;
    # our adapter also accepts any module exposing the same projections.
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "n_local_kv_heads")
    if not all(hasattr(attn, name) for name in required):
        return
    attn.forward = _paged_attention_forward.__get__(attn, type(attn))
    setattr(attn, _PAGEDKV_MARKER, True)


def _is_attention_like(m: Any) -> bool:
    """Duck-typed attention detection.

    Avoids name-matching (``type(m).__name__ in (...)``) so the adapter
    works on MiniMind-O's ``Attention``, any subclass, and test
    substitutes that expose the same projection + head-dim interface.
    """
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "n_local_kv_heads")
    return all(hasattr(m, name) for name in required)


def enable_paged_kv_cache(
    model: Any,
    num_blocks: int | None = None,
    block_size: int = 16,
    max_batch_size: int = 1,
    max_num_new_tokens: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> PagedKVCache | None:
    """Install paged-KV attention on every instance.

    Parameters
    ----------
    model
        The MiniMind-O model (loaded, on the target device).
    num_blocks
        Number of KV blocks in the shared pool. Defaults to a free-VRAM
        estimate (``gpu_memory_utilization * total - used - peak``).
    block_size
        Tokens per block. 16 is a good local-serving default (small
        block = finer allocation, less wasted space per request).
    max_batch_size
        Maximum number of concurrent requests the served layer should
        accept. Sized for the largest ``B`` the CUDA Graph buckets
        will capture.
    max_num_new_tokens
        Maximum new tokens per step (= 1 for normal AR; can be > 1 for
        speculative). Sizes the persistent ``slot_mapping`` scratch.
    gpu_memory_utilization
        Fraction of total VRAM to dedicate to KV cache.

    Returns the ``PagedKVCache`` that owns the shared block pool, or
    ``None`` when the model has no compatible attention instances.
    """
    import torch

    attns = [m for m in model.modules() if _is_attention_like(m)]
    if not attns:
        return None
    first = attns[0]
    dev = first.q_proj.weight.device
    dtype = first.q_proj.weight.dtype
    n_kv = first.n_local_kv_heads
    d = first.head_dim
    num_layers = len(attns)

    if num_blocks is None:
        # Free-VRAM estimate -- fork's approach.
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # block_bytes: K + V across num_layers, block_size tokens, n_kv heads, d head_dim
        block_bytes = (
            2 * num_layers * block_size * n_kv * d * dtype.itemsize
        )
        budget_bytes = int(total * gpu_memory_utilization) - used - peak + current
        num_blocks = max(1, budget_bytes // block_bytes)

    cache = PagedKVCache(model, num_blocks, block_size, max_batch_size=max_batch_size)
    cache.attns = attns

    # 1. Allocate shared block pool + wire per-layer views.
    cache.init_pool(attns, num_blocks, block_size, dtype, dev)

    # 2. Allocate scratch tensors (slot_mapping / context_lens /
    #    block_tables / cu_seqlens).
    max_blocks_per_seq = (num_blocks + max_batch_size - 1) // max_batch_size + 1
    cache.init_scratch(
        max_batch_size=max_batch_size,
        max_blocks_per_seq=max_blocks_per_seq,
        max_num_new_tokens=max_num_new_tokens,
        device=dev,
    )
    cache._ctx = PagedKVContext(
        slot_mapping=cache.scratch["slot_mapping"],
        context_lens=cache.scratch["context_lens"],
        block_tables=cache.scratch["block_tables"],
        cu_seqlens_q=cache.scratch["cu_seqlens_q"],
        cu_seqlens_k=cache.scratch["cu_seqlens_k"],
    )

    # 3. BlockManager (one shared allocator).
    cache.block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)

    # 4. Rewrite every attention's forward to the paged version + bind
    #    the per-attention ctx pointer (shared across layers).
    for attn in attns:
        attn._paged_kv_ctx = cache._ctx
        with suppress(AttributeError, TypeError):
            _attach_paged_kv(attn)

    return cache


__all__ = [
    "PagedKVCache",
    "PagedKVContext",
    "BlockManager",
    "Sequence",
    "enable_paged_kv_cache",
    "store_kvcache",
    "store_kvcache_kernel",
]
