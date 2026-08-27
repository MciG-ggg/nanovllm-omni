"""Project-owned attention runtime tweaks for MiniMind-O.

Four monkey-patches applied at model load via ``bundle.load_minimind_omni_bundle``:

  1. ``enable_sdpa_decode``    -- replace manual attention with
     ``F.scaled_dot_product_attention``; decode branch uses
     ``is_causal=False`` (Q length=1 + K/V past => causal mask would
     collapse attention to K[0]).
  2. ``enable_fused_projections`` -- 3 matmuls -> 1 for attention QKV,
     2 -> 1 for MLP gate-up.
  3. ``enable_fused_rmsnorm``  -- ``aten._fused_rms_norm`` with the
     upstream fp32-in / fp32-weight / fp16-out precision pattern.
  4. ``enable_fused_rope``     -- algebraic-identity rotate-half +
     ``torch.compile(dynamic=True)``.

All patches are zero-dep, zero vendor; the upstream model code is
unchanged. Critical fix: ``enable_fused_projections`` patches every
*instance*, not just the first per class -- the earlier
``seen_attn``/``seen_mlp`` dedupe left 11 of 12 attention layers
un-fused, masking much of the stack's claimed wall-clock benefit.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

# Idempotency markers: one per patch.
_SDPA_MARKER = "_nanovllm_sdpa_decode"
_FUSED_RMS_MARKER = "_nanovllm_fused_rms"
_FUSED_ROPE_MARKER = "_nanovllm_fused_rope"

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# shared attention tail
# ---------------------------------------------------------------------------


def _import_upstream(module_name: str) -> Any:
    """Resolve the vendored upstream module by dotted name.

    Uses ``__import__`` (not a static import) so this module never couples
    to the vendored model code at import time; we only touch it when a
    monkey-patch is applied at model load.
    """
    return __import__(module_name, fromlist=["apply_rotary_pos_emb"])


def _attention_forward(
    self: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any,
    use_cache: bool,
    attention_mask: Any,
    batch_size: int,
    sequence_len: int,
) -> tuple[Any, Any]:
    """Shared post-projection tail of the two attention forwards.

    ``query / key / value`` are already projected, normed and reshaped to
    ``[batch, seq, n_heads, head_dim]`` by the caller (single-proj or
    fused-qkv proj path).

    The decode branch (Q length=1 plus the full K/V past) uses
    ``is_causal=False``. PyTorch SDPA's ``is_causal=True`` is only valid
    when Q, K, V share one sequence length; for Q length=1 it builds a
    ``[1, N]`` lower-triangular mask that attends only to K[0] (the BOS
    position). That collapses decode logits onto a single token and turns
    the subsequent multinomial sampling into noise -- the "garbled
    MiniMind-O audio" bug. With Q length=1 the current token genuinely
    attends to all past positions, so ``is_causal=False`` is correct.
    """
    import math

    import torch
    import torch.nn.functional as functional

    module = _import_upstream(type(self).__module__)
    query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_value is not None:
        key = torch.cat([past_key_value[0], key], dim=1)
        value = torch.cat([past_key_value[1], value], dim=1)
    past = (key, value) if use_cache else None

    query = query.transpose(1, 2)
    key = module.repeat_kv(key, self.n_rep).transpose(1, 2)
    value = module.repeat_kv(value, self.n_rep).transpose(1, 2)

    if sequence_len == 1 and past_key_value is not None and attention_mask is None:
        output = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
    elif (
        self.flash
        and (sequence_len > 1)
        and (not self.is_causal or past_key_value is None)
        and (attention_mask is None or torch.all(attention_mask == 1))
    ):
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )
    else:
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.is_causal:
            scores[:, :, :, -sequence_len:] += torch.full(
                (sequence_len, sequence_len), float("-inf"), device=scores.device
            ).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = (
            self.attn_dropout(functional.softmax(scores.float(), dim=-1).type_as(query)) @ value
        )

    output = output.transpose(1, 2).reshape(batch_size, sequence_len, -1)
    return self.resid_dropout(self.o_proj(output)), past


def _sdpa_forward(
    self: Any,
    x: Any,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any = None,
    use_cache: bool = False,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    """Attention forward with separate Q/K/V projections + SDPA decode."""
    batch_size, sequence_len, _ = x.shape
    query = self.q_proj(x).view(batch_size, sequence_len, self.n_local_heads, self.head_dim)
    key = self.k_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    value = self.v_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    query, key = self.q_norm(query), self.k_norm(key)
    return _attention_forward(
        self,
        query,
        key,
        value,
        position_embeddings=position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask,
        batch_size=batch_size,
        sequence_len=sequence_len,
    )


def _fused_attention_forward(
    self: Any,
    x: Any,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any = None,
    use_cache: bool = False,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    """Attention forward with fused QKV projection + SDPA decode."""
    import torch

    batch_size, sequence_len, _ = x.shape
    qkv = self.qkv_proj(x)
    head_q = self.n_local_heads * self.head_dim
    head_k = self.n_local_kv_heads * self.head_dim
    query, key, value = torch.split(qkv, [head_q, head_k, head_k], dim=-1)
    query = query.reshape(batch_size, sequence_len, self.n_local_heads, self.head_dim)
    key = key.reshape(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    value = value.reshape(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    query, key = self.q_norm(query), self.k_norm(key)
    return _attention_forward(
        self,
        query,
        key,
        value,
        position_embeddings=position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask,
        batch_size=batch_size,
        sequence_len=sequence_len,
    )


def enable_sdpa_decode(model: Any) -> None:
    """Install ``_sdpa_forward`` on every attention class in thinker / talker."""
    classes = {
        stack[0].self_attn.__class__ for stack in (model.thinker.layers, model.talker.layers)
    }
    for cls in classes:
        if getattr(cls, _SDPA_MARKER, False):
            continue
        cls.forward = _sdpa_forward
        setattr(cls, _SDPA_MARKER, True)


# ---------------------------------------------------------------------------
# 2. QKV / gate-up fusion
# ---------------------------------------------------------------------------


def _fused_mlp_forward(self: Any, x: Any) -> Any:
    gu = self.gate_up_proj(x)
    intermediate = gu.shape[-1] // 2
    gate = gu[..., :intermediate]
    up = gu[..., intermediate:]
    return self.down_proj(self.act_fn(gate) * up)


def _fuse_attention_qkv(attn: Any) -> None:
    import torch
    import torch.nn as nn

    if hasattr(attn, "qkv_proj"):
        return

    q_w = attn.q_proj.weight.data
    k_w = attn.k_proj.weight.data
    v_w = attn.v_proj.weight.data
    combined = torch.cat([q_w, k_w, v_w], dim=0)
    qkv_proj = nn.Linear(combined.shape[1], combined.shape[0], bias=False).to(
        dtype=q_w.dtype, device=q_w.device
    )
    qkv_proj.weight.data = combined
    attn.qkv_proj = qkv_proj
    attn.forward = _fused_attention_forward.__get__(attn, type(attn))


def _fuse_mlp_gate_up(mlp: Any) -> None:
    import torch
    import torch.nn as nn

    if hasattr(mlp, "gate_up_proj"):
        return

    g_w = mlp.gate_proj.weight.data
    u_w = mlp.up_proj.weight.data
    combined = torch.cat([g_w, u_w], dim=0)
    gate_up_proj = nn.Linear(combined.shape[1], combined.shape[0], bias=False).to(
        dtype=g_w.dtype, device=g_w.device
    )
    gate_up_proj.weight.data = combined
    mlp.gate_up_proj = gate_up_proj
    mlp.forward = _fused_mlp_forward.__get__(mlp, type(mlp))


def enable_fused_projections(model: Any) -> None:
    """Fuse QKV in attention and gate-up in MLP. Skips MoE MLPs.

    Patches every *instance* (not just the first one per class).
    """
    for module in model.modules():
        cls_name = type(module).__name__
        # ponytail: monkey-patching heterogeneous upstream code needs a
        # defensive skip; narrower than bare ``Exception`` so shape / dtype
        # errors still surface.
        if cls_name == "Attention":
            with suppress(AttributeError, TypeError):
                _fuse_attention_qkv(module)
        elif cls_name == "FeedForward":
            with suppress(AttributeError, TypeError):
                _fuse_mlp_gate_up(module)


# ---------------------------------------------------------------------------
# 3. Fused RMSNorm
# ---------------------------------------------------------------------------


def _fused_rms_forward(self: Any, x: Any) -> Any:
    import torch

    in_dtype = x.dtype
    weight = self.weight if self.weight.dtype == torch.float32 else self.weight.to(torch.float32)
    y, _ = torch.ops.aten._fused_rms_norm(x.to(torch.float32), list(weight.shape), weight, self.eps)
    return y.to(in_dtype)


def enable_fused_rmsnorm(model: Any) -> None:
    for module in model.modules():
        cls = type(module)
        if cls.__name__ != "RMSNorm":
            continue
        if getattr(cls, _FUSED_RMS_MARKER, False):
            continue
        cls.forward = _fused_rms_forward  # type: ignore[assignment]
        setattr(cls, _FUSED_RMS_MARKER, True)


# ---------------------------------------------------------------------------
# 4. Fused RoPE (algebraic identity + torch.compile)
# ---------------------------------------------------------------------------


def _rotate_half(t: Any, cos: Any, sin: Any) -> Any:
    """Apply RoPE without ``rotate_half``'s torch.cat launch.

    ``cos`` / ``sin`` are precomputed as ``cat(half, half)`` so the
    rotate-half formula collapses from 7 elementwise launches to 6.
    Bit-exact vs upstream for fp16 inputs (max abs diff: 0.0).
    """
    import torch

    half = t.shape[-1] // 2
    t1 = t[..., :half]
    t2 = t[..., half:]
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    out = torch.empty_like(t)
    out[..., :half] = t1 * cos_half - t2 * sin_half
    out[..., half:] = t2 * cos_half + t1 * sin_half
    return out


def _fused_apply_rotary_pos_emb(
    q: Any, k: Any, cos: Any, sin: Any, unsqueeze_dim: int = 1
) -> tuple[Any, Any]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)


def enable_fused_rope(model: Any) -> None:
    """Patch upstream ``apply_rotary_pos_emb`` to the algebraic form.

    Optionally wraps it with ``torch.compile(dynamic=True)`` so Inductor
    can fuse the 6 elementwise launches into 1-2 kernels.
    """
    import torch

    try:
        attn = model.thinker.layers[0].self_attn
        module = _import_upstream(type(attn).__module__)
    except (AttributeError, IndexError, ImportError) as exc:
        _log.warning("enable_fused_rope: could not resolve upstream RoPE module: %s", exc)
        return
    if getattr(module, _FUSED_ROPE_MARKER, False):
        return
    try:
        fn = torch.compile(_fused_apply_rotary_pos_emb, dynamic=True)
    except Exception:
        # ponytail: torch.compile can fail on first import (no inductor
        # backend, no CUDA, etc.). Fall back to the eager fused form --
        # still saves the cat launch.
        fn = _fused_apply_rotary_pos_emb
    module.apply_rotary_pos_emb = fn
    setattr(module, _FUSED_ROPE_MARKER, True)


__all__ = [
    "enable_fused_projections",
    "enable_fused_rope",
    "enable_fused_rmsnorm",
    "enable_sdpa_decode",
]
