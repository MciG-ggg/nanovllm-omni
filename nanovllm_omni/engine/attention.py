"""Fixed-KV-buffer attention (CUDA-Graph capture pillar).

Installs a per-instance preallocated KV buffer + buffered forward on every
attention module so the AR-step KV history has static shape across decode
steps. Used by the CUDA-Graph decoder (``optim/cuda_graph.py``); opt-in
only -- bundle loading does NOT call it automatically.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

# Instance marker: attention has a fixed KV buffer + buffered forward.
_KVBUFFER_MARKER = "_nanovllm_kv_buffer"


def _import_upstream(module_name: str) -> Any:
    """Resolve the vendored upstream module by dotted name.

    Kept as a module-level helper because
    ``tests/test_fixed_kv_buffer_forward.py`` monkey-patches it to inject a
    stub ``apply_rotary_pos_emb`` / ``repeat_kv`` for the buffered path;
    inlining the call site would force the test to patch ``__import__``
    instead, which is harder to scope.
    """
    return __import__(module_name, fromlist=["apply_rotary_pos_emb"])


def _attention_forward_buffered(
    self: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    position_embeddings: tuple[Any, Any],
    use_cache: bool,
    attention_mask: Any,
    batch_size: int,
    sequence_len: int,
) -> tuple[Any, Any]:
    """Same tail as the eager attention forward but the KV history lives in
    a preallocated fixed buffer (``self._kv_past_key/value``) instead of a
    per-step ``torch.cat``.

    Keeps tensor shapes static across AR steps (CUDA Graph requirement)
    with zero arithmetic change.
    """
    import math

    import torch
    import torch.nn.functional as functional

    module = _import_upstream(type(self).__module__)
    query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)

    # --- fixed-buffer KV update (replaces torch.cat([past, cur], dim=1)) ---
    self._kv_past_key[:, self._kv_pos : self._kv_pos + sequence_len] = key
    self._kv_past_value[:, self._kv_pos : self._kv_pos + sequence_len] = value
    self._kv_pos += sequence_len
    past = (
        (self._kv_past_key[:, : self._kv_pos], self._kv_past_value[:, : self._kv_pos])
        if use_cache
        else None
    )
    k_hist = self._kv_past_key[:, : self._kv_pos]
    v_hist = self._kv_past_value[:, : self._kv_pos]

    query = query.transpose(1, 2)
    key = module.repeat_kv(k_hist, self.n_rep).transpose(1, 2)
    value = module.repeat_kv(v_hist, self.n_rep).transpose(1, 2)

    # No ``past_key_value is not None`` guard here on purpose: the fixed
    # buffer always has >= 1 prepended history row by the time decode runs
    # (graphed path always prefills first, _kv_pos >= 1), so decode never
    # SDPA's against an empty history.
    if sequence_len == 1 and attention_mask is None:
        output = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
    elif (
        self.flash
        and (sequence_len > 1)
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


def _kv_buffer_forward(
    self: Any,
    x: Any,
    position_embeddings: tuple[Any, Any],
    past_key_value: Any = None,
    use_cache: bool = False,
    attention_mask: Any = None,
) -> tuple[Any, Any]:
    """Projection tail: separate QKV (the only remaining projection path) +
    buffered history.

    ``past_key_value`` is accepted for API parity but ignored -- the buffer
    holds the authoritative history.
    """

    batch_size, sequence_len, _ = x.shape
    query = self.q_proj(x).view(batch_size, sequence_len, self.n_local_heads, self.head_dim)
    key = self.k_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    value = self.v_proj(x).view(batch_size, sequence_len, self.n_local_kv_heads, self.head_dim)
    query, key = self.q_norm(query), self.k_norm(key)
    return _attention_forward_buffered(
        self,
        query,
        key,
        value,
        position_embeddings=position_embeddings,
        use_cache=use_cache,
        attention_mask=attention_mask,
        batch_size=batch_size,
        sequence_len=sequence_len,
    )


def _attach_kv_buffer(attn: Any, max_len: int) -> None:
    """Allocate fixed KV buffers on the attention instance + bind forward.

    Per-instance (class-level dedupe would break per-instance patches).
    Idempotent: re-attach only resets buffer capacity, never double-binds.
    Buffer seq axis is dim 1 (KV layout (B, seq, n_kv, d)).
    """
    import torch

    if getattr(attn, _KVBUFFER_MARKER, False):
        return
    if not (hasattr(attn, "q_proj") and hasattr(attn, "k_proj")):
        return  # not a projection attention; leave untouched
    b = 1  # single-request local serving (batch handled by caller if needed)
    n_kv = attn.n_local_kv_heads
    d = attn.head_dim
    dev = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    attn._kv_past_key = torch.zeros(b, max_len, n_kv, d, dtype=dtype, device=dev)
    attn._kv_past_value = torch.zeros(b, max_len, n_kv, d, dtype=dtype, device=dev)
    attn._kv_pos = 0
    attn.forward = _kv_buffer_forward.__get__(attn, type(attn))
    setattr(attn, _KVBUFFER_MARKER, True)


def enable_fixed_kv_buffer(model: Any, max_len: int | None = None) -> None:
    """Install fixed-KV-buffer attention on every instance.

    ``max_len`` defaults to the model's max_position_embeddings. The
    buffer must cover prompt_len + max_new_tokens; callers should validate
    before generating.
    """
    if max_len is None:
        max_len = int(getattr(model.config, "max_position_embeddings", 2048))
    for module in model.modules():
        if type(module).__name__ in ("Attention", "MiniMindAttention"):
            with suppress(AttributeError, TypeError):
                _attach_kv_buffer(module, max_len)


__all__ = ["enable_fixed_kv_buffer"]
