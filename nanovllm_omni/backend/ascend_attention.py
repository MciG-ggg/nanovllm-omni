"""Ascend attention: contiguous KV plus ``torch_npu`` fused attention.

Mirrors the fork ``Attention`` interface (``__init__`` / ``forward`` /
``k_cache`` / ``v_cache``) so ``ThinkerAttention`` can take this class
through ``attention_cls``. The fork still assigns its paged KV tensors
onto this module; this class ignores them and keeps its own contiguous
per-sequence cache -- milestone 1 is one sequence, no prefix cache.

``torch_npu`` is imported inside ``forward`` so importing this module on
a machine without Ascend does not fail.
"""

from __future__ import annotations

import torch
from nanovllm.utils.context import get_context
from torch import nn
from torch.nn import functional


class AscendAttention(nn.Module):
    """Contiguous-KV attention for the Ascend track, one sequence at a time."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            if context.block_tables is not None:
                raise NotImplementedError(
                    "prefix cache / chunked prefill is not wired on Ascend yet"
                )
            if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() > 2:
                raise NotImplementedError("AscendAttention is single-sequence until milestone 2")
            # Fresh prefill: the chunk is the whole sequence.
            self.k_cache, self.v_cache = k, v
            causal = True
        else:
            if context.context_lens is None or context.context_lens.numel() > 1:
                raise NotImplementedError("AscendAttention is single-sequence until milestone 2")
            if self.k_cache.numel() == 0:
                raise RuntimeError("decode step ran before any prefill")
            # Simplest correct append. A preallocated buffer is the first
            # thing to replace once the card is available.
            self.k_cache = torch.cat([self.k_cache, k], dim=0)
            self.v_cache = torch.cat([self.v_cache, v], dim=0)
            # The single query is the last position, so every cached key is
            # visible; a causal mask here would align top-left and hide them.
            causal = False

        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return _sdpa(q, self.k_cache, self.v_cache, self.num_heads, self.scale, causal)
        return _npu_fusion_attention(
            q, self.k_cache, self.v_cache, self.num_heads, self.scale, causal
        )


def _npu_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """TND varlen fused attention; q, k, v are ``[T, H, D]`` / ``[T, H_kv, D]``."""
    import torch_npu

    output = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        num_heads,
        "TND",
        scale=scale,
        keep_prob=1.0,
        actual_seq_qlen=[q.shape[0]],
        actual_seq_kvlen=[k.shape[0]],
        sparse_mode=3 if causal else 0,
    )
    return output[0]


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """SDPA fallback for prefill (causal) and decode (full attention)."""
    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k.transpose(0, 1).unsqueeze(0)
    v4 = v.transpose(0, 1).unsqueeze(0)
    repeat = num_heads // k.shape[1]
    if repeat != 1:
        k4 = k4.repeat_interleave(repeat, dim=1)
        v4 = v4.repeat_interleave(repeat, dim=1)
    out = functional.scaled_dot_product_attention(q4, k4, v4, is_causal=causal, scale=scale)
    return out.squeeze(0).transpose(0, 1)


__all__ = ["AscendAttention"]
