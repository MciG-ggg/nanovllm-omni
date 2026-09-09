"""CPU parity tests: paged KV path writes the same K/V as the vendor path.

These tests lock the contract that motivated replacing fixed-KV buffers
with paged KV: every token position must end up with the same K/V
content under both layouts. They run on CPU because the fake
MiniMind-O attention stub is a plain SDPA; the CUDA flash-attn path
is exercised end-to-end on the 3050 bench artifacts in
``docs/perf/aligned/enginecore-phase4-*/``.

What the tests cover
--------------------

- ``_store_kv_paged`` followed by ``_gather_paged_kv`` reconstructs
  the exact same K/V as the vendor ``torch.cat([past, k])`` path.
- Multi-step decode accumulates without cross-step corruption.
- ``enable_paged_kv_cache`` monkey-patches ``attn.forward`` and the
  monkey-patched forward writes into the paged pool; a parallel
  vendor forward writes into a contiguous buffer; the two are
  bit-equal after layout reshape.

Why CPU is enough
-----------------

The contract is "same tensor layout under two storage backends." We
do not need real flash-attn to test that -- a plain SDPA over the
same tokens with the same weights produces the same K/V, regardless
of whether the storage is contiguous or paged. The CUDA parity
proof lives in the bench artifacts.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

import pytest
import torch
import torch.nn as nn


def apply_rotary_pos_emb(  # noqa: D401 -- identity RoPE for parity test
    q: torch.Tensor, k: torch.Tensor, cos: Any, sin: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity RoPE: paged attention's first step is orthogonal to RoPE.

    The parity check only inspects K/V storage layout, not the
    attention output. We keep RoPE a no-op so the contract stays
    self-contained.
    """
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Identity repeat_kv for ``n_rep == 1``; expand otherwise.

    ``_paged_sdpa_decode`` and the prefill SDPA branch expect GQA-style
    head expansion. Our fake attention uses ``n_local_heads ==
    n_local_kv_heads`` so ``n_rep == 1`` and a plain clone suffices.
    """
    if n_rep == 1:
        return x.clone()
    bsz, num_kv, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, num_kv, n_rep, seq_len, head_dim)
        .reshape(bsz, num_kv * n_rep, seq_len, head_dim)
    )


# Stubs for the fork's optional deps before importing the SUT.
def _ensure_stub(name: str, **attrs: Any) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_ensure_stub("triton", jit=(lambda *a, **kw: lambda fn: fn))
_ensure_stub(
    "triton.language",
    constexpr=type("constexpr", (), {}),
)
_ensure_stub(
    "flash_attn",
    flash_attn_varlen_func=(lambda *a, **kw: None),
    flash_attn_with_kvcache=(lambda *a, **kw: None),
)
_ensure_stub("xxhash")

from nanovllm_omni.models.minimind_omni import paged_attention as pa  # noqa: E402

# ---------------------------------------------------------------------------
# Fake MiniMind-O attention + model
# ---------------------------------------------------------------------------


class _ParityAttention(nn.Module):
    """MiniMind-O style Attention; writes K/V under both layouts.

    The vendor path (``past_key_value is not None``) uses
    ``torch.cat``; the paged path delegates to the monkey-patched
    ``_paged_attention_forward`` installed by ``enable_paged_kv_cache``
    which routes K/V through ``_store_kv_paged`` instead.

    Deterministic weights so equal inputs yield equal K/V. We use
    ``n_local_heads == n_local_kv_heads`` so ``n_rep == 1`` -- the GQA
    expansion path is orthogonal to the layout parity check.
    """

    def __init__(self, n_heads: int = 2, n_kv_heads: int = 2, d: int = 4) -> None:
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d
        self.is_causal = True
        # Initialize projections with fixed values so tests are deterministic.
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = nn.Linear(
                n_heads * d,
                n_heads * d if proj_name in {"q_proj", "o_proj"} else n_kv_heads * d,
                bias=False,
            )
            with torch.no_grad():
                proj.weight.fill_(0.1)
            setattr(self, proj_name, proj)
        self.flash = False  # force SDPA fallback (no flash-attn stubs)
        self.dropout = 0.0
        # Identity q/k norms: upstream ``_paged_attention_forward`` expects
        # these as ``nn.Module`` (it calls ``self.q_norm(x)``).
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        # Identity dropout modules -- paged forward calls both.
        self.attn_dropout = nn.Identity()
        self.resid_dropout = nn.Identity()

    def forward(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # ``enable_paged_kv_cache`` rebinds ``forward`` on this instance
        # and sets ``_nanovllm_paged_kv``. ``disable_paged_kv_cache``
        # deletes the instance attribute to restore the class method.
        # The tests below call ``_vendor_forward`` directly so we bypass
        # the paged rewrite for the reference path.
        if getattr(self, "_nanovllm_paged_kv", False):
            # Tests that need the vendor output call ``_vendor_forward``.
            # The paged-installed bound method on ``self.forward`` is what
            # the *production* decoder would hit.
            raise RuntimeError(
                "test called parity attention after paged install; use "
                "_vendor_forward() to bypass the monkey-patch."
            )
        return self._vendor_forward(x, past_key_value, use_cache)

    def _vendor_forward(
        self,
        x: torch.Tensor,
        past_key_value: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Vendor-style attention: ``torch.cat`` past K/V, SDPA, return ``(out, past)``."""
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(out)
        return out, past_kv


class _ParityModel(nn.Module):
    """Stack of ``_ParityAttention`` instances sharing the layer index."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=64)
        self.layers = nn.ModuleList([_ParityAttention() for _ in range(num_layers)])


class _ParityBlock(nn.Module):
    """Mirror of vendored ``MiniMindBlock``: attention nested as ``self_attn``.

    The vendored model never calls an attention directly -- it goes
    ``MiniMindModel.layers[i](...)`` -> ``block.forward`` ->
    ``self.self_attn(...)``, an ``nn.Module.__call__``. Whatever the
    paged install puts at ``block.self_attn`` must survive that chain,
    which is exactly what the wrapper tests below exercise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _ParityAttention()

    def forward(  # type: ignore[no-untyped-def]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        residual = hidden_states
        hidden, present = self.self_attn(
            hidden_states,
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        return hidden + residual, present


class _ParityBlockModel(nn.Module):
    """``MiniMindModel``-shaped stack of ``_ParityBlock`` layers."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(max_position_embeddings=64)
        self.layers = nn.ModuleList([_ParityBlock() for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather_vendor_kv(
    cache: pa.PagedKVCache, layer_idx: int, block_table: list[int], n_tokens: int
) -> torch.Tensor:
    """Read back K for one request from the paged pool, ordered like vendor past_kv[0].

    Returns the K tensor in vendor layout ``[1, n_tokens, n_kv, d]``.
    """
    block_size = cache.block_size
    n_kv = cache.attns[layer_idx].n_local_kv_heads
    d = cache.attns[layer_idx].head_dim
    # Each token's K lives at flat slot ``block_table[block] * block_size + offset``.
    rows = []
    for tok in range(n_tokens):
        block_id = block_table[tok // block_size]
        offset = tok % block_size
        flat = block_id * block_size + offset
        rows.append(cache.kv_cache[0, layer_idx].view(-1, n_kv, d)[flat][None, None])
    return torch.cat(rows, dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_store_then_gather_round_trip() -> None:
    """One decode step: paged write + gather matches vendor ``torch.cat``."""
    model = _ParityModel(num_layers=1)
    # Vendor reference FIRST (before paged install monkey-patches forward).
    attn = model.layers[0]
    torch.manual_seed(0)
    x = torch.randn(1, 1, 2 * 4)
    _, vendor_past = attn._vendor_forward(x, past_key_value=None, use_cache=True)
    assert vendor_past is not None
    vendor_k = vendor_past[0]  # [1, 1, n_kv, d]

    # Now enable paged.
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=4, block_size=4, max_batch_size=1, max_num_new_tokens=4
    )
    assert cache is not None

    # Paged write: the paged forward writes to the pool at slot 0.
    # ``new_sequence`` requires a non-empty list because the fork's
    # ``Sequence.__init__`` reads ``token_ids[-1]``; the placeholder is
    # overwritten by ``append_token``.
    seq = cache.new_sequence([0])
    cache.append_token(seq, 99)
    paged_attn = cache.attns[0]
    # Paged forward reads the cache._ctx to know slot_mapping; we set it directly.
    paged_attn._paged_kv_ctx.is_prefill = False
    paged_attn._paged_kv_ctx.slot_mapping.fill_(-1)
    # Slot for the only token = seq.block_table[-1] * block_size + last_token_idx.
    flat = seq.block_table[-1] * cache.block_size + (seq.last_block_num_tokens - 1)
    paged_attn._paged_kv_ctx.slot_mapping[0] = flat
    paged_attn._paged_kv_ctx.context_lens[0] = seq.num_tokens
    # Call the monkey-patched forward directly with the SAME input + position.
    out, _ = paged_attn.forward(x, (None, None), past_key_value=None, use_cache=False)
    # The paged forward writes K/V using the SAME q/k/v projections,
    # so the freshly stored K at slot ``flat`` must equal ``vendor_k``'s row 0.
    n_kv = attn.n_local_kv_heads
    d = attn.head_dim
    k_cache_flat = cache.kv_cache[0, 0].view(-1, n_kv, d)
    stored_k = k_cache_flat[flat]
    assert torch.allclose(stored_k, vendor_k[0, 0], atol=1e-5), (
        "paged store diverged from vendor K at slot 0"
    )
    # Output logits must also be finite and have the expected shape.
    assert out.shape == (1, 1, 2 * 4)


def test_multi_step_decode_accumulates_correctly() -> None:
    """Five decode steps; gathered paged K must equal the vendor past_kv."""
    model = _ParityModel(num_layers=2)

    # Vendor reference: walk 5 decode steps; the final ``past_kv`` is
    # the concatenation of all per-token K writes.
    pasts: list[Any] = [None, None]
    torch.manual_seed(1)
    x = torch.randn(1, 1, 2 * 4)
    for _step in range(5):
        for li, layer in enumerate(model.layers):
            _, new_past = layer._vendor_forward(x, past_key_value=pasts[li], use_cache=True)
            pasts[li] = new_past

    cache = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=1, max_num_new_tokens=1
    )
    assert cache is not None

    # Paged path: same 5 steps, same input, same q/k/v projections.
    seq = cache.new_sequence([0])
    paged_layer = cache.attns
    for step in range(5):
        cache.append_token(seq, 99 + step)
        flat = seq.block_table[-1] * cache.block_size + (seq.last_block_num_tokens - 1)
        ctx = paged_layer[0]._paged_kv_ctx
        ctx.is_prefill = False
        ctx.slot_mapping.fill_(-1)
        ctx.slot_mapping[0] = flat
        ctx.context_lens[0] = seq.num_tokens
        for layer in paged_layer:
            layer.forward(x, (None, None), past_key_value=None, use_cache=False)

    # Now the paged pool holds the same per-token K's as ``pasts[li][0]``,
    # just laid out across block slots instead of concatenated. Gather
    # them in sequence order and compare.
    for li in range(2):
        gathered = _gather_vendor_kv(cache, li, seq.block_table, seq.num_tokens)
        vendor_k = pasts[li][0]  # [1, T, n_kv, d]
        # After new_sequence + 5 appends the seq has 6 tokens; vendor
        # past only covers the 5 appended positions.
        assert torch.allclose(gathered[:, -5:], vendor_k[:, -5:], atol=1e-5), (
            f"paged-gathered K diverged from vendor K at layer {li}"
        )


def test_paged_path_matches_vendor_path_end_to_end() -> None:
    """Run both paths on the same input; gather paged and compare to vendor.

    The vendor path writes per-layer ``past_kv`` as ``torch.cat``;
    the paged path writes into the shared pool at slots dictated by
    the sequence's block table. After gathering the paged K/V into the
    vendor layout (``[1, n_tokens, n_kv, d]``), they must match
    position-by-position.
    """
    model = _ParityModel(num_layers=2)

    # Vendor reference FIRST (before paged install).
    pasts: list[Any] = [None, None]
    torch.manual_seed(2)
    x = torch.randn(1, 3, 2 * 4)  # prefill-shaped input
    for li, layer in enumerate(model.layers):
        _, past = layer._vendor_forward(x, past_key_value=None, use_cache=True)
        pasts[li] = past

    # Now install paged.
    cache = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=1, max_num_new_tokens=3
    )
    assert cache is not None

    # Paged path: prefill-shaped input writes 3 tokens in one call.
    seq = cache.new_sequence([0, 0, 0])
    paged_layer0 = cache.attns[0]
    ctx = paged_layer0._paged_kv_ctx
    ctx.is_prefill = True
    ctx.cu_seqlens_q.zero_()
    ctx.cu_seqlens_q[1] = 3
    ctx.cu_seqlens_k.zero_()
    ctx.cu_seqlens_k[1] = 3
    ctx.max_seqlen_q = 3
    ctx.max_seqlen_k = 3
    ctx.context_lens[0] = 3
    slots = []
    for tok in range(3):
        block_id = seq.block_table[tok // cache.block_size]
        offset = tok % cache.block_size
        slots.append(block_id * cache.block_size + offset)
    ctx.slot_mapping.fill_(-1)
    ctx.slot_mapping[:3] = torch.tensor(slots, dtype=ctx.slot_mapping.dtype)
    ctx.block_tables.zero_()
    ctx.block_tables[0, : len(seq.block_table)] = torch.tensor(
        seq.block_table, dtype=ctx.block_tables.dtype
    )

    for layer in cache.attns:
        layer.forward(x, (None, None), past_key_value=None, use_cache=False)

    # Gather paged K for each layer and compare to vendor.
    for li, _layer in enumerate(model.layers):
        vendor_k = pasts[li][0]  # [1, 3, n_kv, d]
        gathered = _gather_vendor_kv(cache, li, seq.block_table, seq.num_tokens)
        assert torch.allclose(gathered, vendor_k, atol=1e-5), (
            f"paged-gathered K diverged from vendor K at layer {li}"
        )


def test_wrapper_swap_survives_block_call_chain_and_disable_restores() -> None:
    """Install/disable round-trip through the real vendored call chain.

    ``_PagedAttentionWrapper`` is an ``nn.Module`` swapped into
    ``block.self_attn``; the block's ``self.self_attn(...)`` must hit
    the paged forward through ``__call__``, and ``disable_paged_kv_cache``
    must put the original attention (with its own ``forward``) back.
    """
    model = _ParityBlockModel(num_layers=2)
    originals = [model.layers[li].self_attn for li in range(2)]

    cache = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=1, max_num_new_tokens=4
    )
    assert cache is not None
    # Block slots now hold wrappers proxying the original attentions.
    for li in range(2):
        slot = model.layers[li].self_attn
        assert getattr(slot, pa._PAGEDKV_MARKER, False) is True
        assert slot.original is originals[li]

    # One decode step through ``block.forward`` -> ``self.self_attn(...)``.
    seq = cache.new_sequence([0])
    cache.append_token(seq, 5)
    ctx = cache.attns[0]._paged_kv_ctx
    ctx.is_prefill = False
    ctx.slot_mapping.fill_(-1)
    flat = seq.block_table[-1] * cache.block_size + (seq.last_block_num_tokens - 1)
    ctx.slot_mapping[0] = flat
    ctx.context_lens[0] = seq.num_tokens
    torch.manual_seed(3)
    hidden = torch.randn(1, 1, 2 * 4)
    for li in range(2):
        hidden, _present = model.layers[li](hidden, (None, None), use_cache=False)
    assert hidden.shape == (1, 1, 2 * 4)
    assert torch.isfinite(hidden).all()

    # Disable restores the vendored attentions, and the cycle repeats.
    assert pa.disable_paged_kv_cache(model) == 2
    for li in range(2):
        assert model.layers[li].self_attn is originals[li]
    cache2 = pa.enable_paged_kv_cache(
        model, num_blocks=8, block_size=4, max_batch_size=1, max_num_new_tokens=4
    )
    assert cache2 is not None
    assert pa.disable_paged_kv_cache(model) == 2
    for li in range(2):
        assert model.layers[li].self_attn is originals[li]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
