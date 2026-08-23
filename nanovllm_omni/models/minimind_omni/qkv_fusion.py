"""Fused QKV / gate-up projection monkey-patch for MiniMind-O.

Each attention layer normally runs three independent matmuls for
``q_proj``, ``k_proj``, ``v_proj``. We can replace them with a single
matmul whose weight is the row-wise concatenation of the three
projections. The combined output is split back into the three
streams. This collapses 3 launches into 1 per attention call.

Same trick applied to the MLP ``gate_proj`` / ``up_proj`` (2 → 1).

Verified correctness for float16 weights / inputs by comparing the
fused output against eager.
"""

from __future__ import annotations

from typing import Any

_FUSED_QKV_MARKER = "_nanovllm_fused_qkv"
_FUSED_MLP_MARKER = "_nanovllm_fused_mlp"


def _fused_attention_forward(
    self: Any,
    x: Any,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any = None,
    use_cache: bool = False,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    import math

    import torch
    import torch.nn.functional as functional

    batch_size, seq_len, _ = x.shape
    qkv = self.qkv_proj(x)
    head_q = self.n_local_heads * self.head_dim
    head_k = self.n_local_kv_heads * self.head_dim
    query, key, value = torch.split(qkv, [head_q, head_k, head_k], dim=-1)
    query = query.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim)
    key = key.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
    value = value.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
    query, key = self.q_norm(query), self.k_norm(key)

    module = __import__(type(self).__module__, fromlist=["apply_rotary_pos_emb"])
    query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_value is not None:
        key = torch.cat([past_key_value[0], key], dim=1)
        value = torch.cat([past_key_value[1], value], dim=1)
    past = (key, value) if use_cache else None

    query = query.transpose(1, 2)
    key = module.repeat_kv(key, self.n_rep).transpose(1, 2)
    value = module.repeat_kv(value, self.n_rep).transpose(1, 2)

    if seq_len == 1 and past_key_value is not None and attention_mask is None:
        output = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
    elif (
        self.flash
        and (seq_len > 1)
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
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device
            ).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = (
            self.attn_dropout(functional.softmax(scores.float(), dim=-1).type_as(query)) @ value
        )

    output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    return self.resid_dropout(self.o_proj(output)), past


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


def enable_fused_projections(model: Any) -> int:
    """Fuse QKV in attention and gate-up in MLP. Skips MoE MLPs.

    Patches every *instance* (not just the first one per class). The earlier
    ``seen_attn``/``seen_mlp`` dedupe inadvertently left 11 of 12 layers
    un-fused, masking much of E6's claimed wall-clock benefit.
    """
    patched = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "Attention":
            try:
                _fuse_attention_qkv(module)
                patched += 1
            except Exception:
                pass
        elif cls_name == "FeedForward":
            try:
                _fuse_mlp_gate_up(module)
                patched += 1
            except Exception:
                pass
    return patched


__all__ = ["enable_fused_projections"]
