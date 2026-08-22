"""Project-owned attention runtime tweaks for MiniMind-O."""

from __future__ import annotations

from typing import Any

_SDPA_MARKER = "_nanovllm_sdpa_decode"


def _sdpa_forward(
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
    query = self.q_proj(x).view(batch_size, seq_len, self.n_local_heads, self.head_dim)
    key = self.k_proj(x).view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
    value = self.v_proj(x).view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
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
            query, key, value, dropout_p=0.0, is_causal=False
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


def enable_sdpa_decode(model: Any) -> int:
    """Use PyTorch SDPA for one-token KV-cache decode on MiniMind attention."""
    classes = {
        layer.self_attn.__class__
        for stack in (model.thinker.layers, model.talker.layers)
        for layer in (stack[0],)
    }
    patched = 0
    for attention_class in classes:
        if getattr(attention_class, _SDPA_MARKER, False):
            continue
        attention_class.forward = _sdpa_forward
        setattr(attention_class, _SDPA_MARKER, True)
        patched += 1
    return patched


__all__ = ["enable_sdpa_decode"]
