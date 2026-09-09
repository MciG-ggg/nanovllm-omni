"""MiniMind-O talker stage.

Owns the ``TalkerAttention``, ``TalkerBlock``, and ``MiniMindTalker``
model components using fork layers (``nanovllm.layers.*``).

Public symbols: ``MiniMindTalker``, ``_talker_stage``.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from torch import nn


class TalkerAttention(nn.Module):
    """Attention for the talker MTP blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class TalkerMLP(nn.Module):
    """MLP for talker blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class TalkerBlock(nn.Module):
    """Pre-norm transformer block for talker MTP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_position: int = 4096,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        self.self_attn = TalkerAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            max_position,
            head_dim,
            rms_norm_eps,
            rope_theta,
        )
        self.mlp = TalkerMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMindTalker(nn.Module):
    """MiniMind talker stage: MTP blocks for audio code prediction.

    Takes bridge hidden states from the thinker and produces audio logits
    for 8 codebook channels.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        intermediate_size: int | None = None,
        max_position: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000,
        audio_vocab_size: int = 2048,
        num_audio_heads: int = 8,
        talker_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if intermediate_size is None:
            intermediate_size = math.ceil(hidden_size * 8 / 3 / 256) * 256
        self.talker_hidden_size = talker_hidden_size or hidden_size

        self.embed_proj = nn.Linear(hidden_size, self.talker_hidden_size, bias=False)
        self.text_scale = nn.Parameter(torch.ones(1))
        self.audio_scale = nn.Parameter(torch.ones(1))
        self.embed_tokens = nn.Embedding(audio_vocab_size, self.talker_hidden_size)

        self.layers = nn.ModuleList(
            [
                TalkerBlock(
                    self.talker_hidden_size,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    max_position,
                    None,
                    rms_norm_eps,
                    rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(self.talker_hidden_size, eps=rms_norm_eps)
        self.audio_head = nn.Linear(
            self.talker_hidden_size,
            num_audio_heads * audio_vocab_size,
            bias=False,
        )
        self.num_audio_heads = num_audio_heads
        self.audio_vocab_size = audio_vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_codes: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run talker MTP.

        Args:
            hidden_states: Bridge hidden states from thinker [seq_len, hidden_size]
            text_codes: Audio token IDs [seq_len]
            positions: Position indices for RoPE [seq_len]
        Returns:
            Audio logits [seq_len, num_audio_heads, audio_vocab_size]
        """
        h = self.embed_proj(hidden_states) * self.text_scale
        h = h + self.embed_tokens(text_codes) * self.audio_scale
        residual = None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        h, _ = self.norm(h, residual)
        logits = self.audio_head(h)
        return logits.view(-1, self.num_audio_heads, self.audio_vocab_size)


# ---------------------------------------------------------------------------
#  Stage factory (called by pipeline.py via dotted-path resolution)
# ---------------------------------------------------------------------------


def _talker_stage(deploy, args):
    """Stage 1 factory — fork ``ModelRunner`` + shared ``SharedBlockManager``.

    Returns a ``TalkerStage`` that reuses the thinker's ``SharedBlockManager``
    (per ADR-002: logical block IDs are shared across stages; physical KV
    tensors are per-stage because thinker and talker have different layer
    counts). The MTP decode loop that consumes ``TalkerInputPayload`` and
    emits audio codes is Phase 4 territory.
    """
    from ._engine import TalkerStage

    return TalkerStage(deploy, args)
