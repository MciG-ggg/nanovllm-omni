"""MiniMind-O talker stage.

Owns the ``TalkerAttention``, ``TalkerBlock``, ``TalkerHead``,
``TalkerEmbedding``, and ``MiniMindTalker`` model components.

Architecture matches the vendored HF ``MiniMindOmni.talker`` (see
``model_omni.py`` in the snapshot): a 4-layer pre-norm transformer
with 8 codebook heads (one base Linear + 8 LoRA-style adapters each)
that together predict the 8 audio codebook channels frame by frame.

Public symbols: ``MiniMindTalker``, ``_talker_stage``.

ponytail: matches vendor 1:1; no speculative generalisation. The eager
SDPA attention class is local so we don't pull fork's paged-attention
path into a model whose forward shape is fixed (full sequence each
step).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from torch import nn
from torch.nn import functional

# ---------------------------------------------------------------------------
# Vendor TalkerHead / TalkerEmbedding (verbatim shape, name-for-name).
# ---------------------------------------------------------------------------


class TalkerHead(nn.Module):
    """8 codebook heads: one base Linear + 8 adapter Linears (LoRA-style).

    ``forward(x)`` returns a list of 8 logits tensors of shape
    ``[N, audio_vocab_size]`` — one per codebook channel. The vendor
    pipeline samples each list element independently to draw the 8
    audio codes for one frame.
    """

    def __init__(
        self, in_features: int, out_features: int, num_layers: int = 8, rank: int = 256
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, rank, bias=False),
                    nn.GELU(),
                    nn.Linear(rank, out_features, bias=False),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    """Embedding with base + 8 adapters. Averages across 8 channels.

    Input ``x`` has shape ``[B, num_layers, T]`` (one channel per
    codebook). The vendor formula is::

        base_out = base(x)  # [B, num_layers, T, H]
        return (sum_i (base_out[:, i, :] + adapters[i](x[:, i, :]))
                / num_layers)

    The result is ``[B, T, H]`` — each time step gets an average over
    the 8 codebook embeddings.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, num_layers: int = 8, rank: int = 256
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(num_embeddings, rank),
                    nn.GELU(),
                    nn.Linear(rank, embedding_dim, bias=False),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        return (
            sum(base_out[:, i, :] + self.adapters[i](x[:, i, :]) for i in range(self.num_layers))
            / self.num_layers
        )


# ---------------------------------------------------------------------------
# Inner transformer block (uses fork layers; no paged KV cache).
# ---------------------------------------------------------------------------


class _EagerAttention(nn.Module):
    """SDPA attention — eager full-sequence path.

    Fork's ``Attention`` reads the module-level ``Context`` to pick
    between paged prefill / paged decode. The talker runs the full
    sequence each step (no KV cache), so we bypass that machinery
    and use ``F.scaled_dot_product_attention`` directly.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        # GQA: replicate k/v to match num_heads when num_kv_heads != num_heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # SDPA signature is [B, H, N, D]; treat the whole batch as B=1.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        o = functional.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
        return o.squeeze(0).transpose(0, 1).reshape(n, self.num_heads * q.shape[-1])


class TalkerAttention(nn.Module):
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
            hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False
        )
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim, max_position=max_position, base=rope_theta
        )
        self.attn = _EagerAttention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
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
        return self.o_proj(o)


class TalkerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class TalkerBlock(nn.Module):
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
            hidden_size, num_heads, num_kv_heads, max_position, head_dim, rms_norm_eps, rope_theta
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


# ---------------------------------------------------------------------------
# MiniMindTalker — vendor TalkerModule architecture.
# ---------------------------------------------------------------------------


class MiniMindTalker(nn.Module):
    """Talker: bridge hidden + audio codes → 8 codebook logits per step.

    Architecture matches the vendored ``MiniMindOmni.talker``
    (``TalkerModule`` in ``model_omni.py``).  ``forward`` is eager and
    runs the full sequence each step — no KV cache.  The vendor uses
    KV cache for speed; we skip it for code-size parity with the
    teaching goal.  If the per-step cost becomes painful, swap in a
    paged cache; the rest of the API stays the same.

    Args:
        bridge_states: ``[B, T, hidden_size]`` from the thinker
            ``bridge_layer`` (the thinker's own hidden_size, NOT
            ``talker_hidden_size``; ``embed_proj`` projects it).
        audio_codes: ``[B, 8, T]`` — channel ``i`` is the i-th
            codebook's history. Pad positions (before the model has
            produced a code) are filled with ``audio_pad_token`` (2049).
        positions: ``[T]`` absolute RoPE positions.
        spk_emb: optional ``[B, spk_emb_size]``. When provided, it is
            inserted at the first position of the sequence (vendor
            convention; we do not implement the ``audio_spk_token``
            mask, the caller decides whether to include it).

    Returns:
        list of 8 logits tensors, each ``[B, T, audio_vocab_size]``.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        talker_hidden = getattr(config, "talker_hidden_size", None) or config.hidden_size
        audio_vocab = getattr(config, "audio_vocab_size", 2112)
        num_layers = getattr(config, "num_talker_hidden_layers", 4)
        # Attention geometry — taken from the thinker config (same model family).
        num_heads = getattr(config, "num_attention_heads", 8)
        num_kv_heads = getattr(config, "num_key_value_heads", 4)
        head_dim = getattr(config, "head_dim", None) or talker_hidden // num_heads
        # The vendor ``TalkerModule`` reuses the thinker MLP width.
        intermediate_size = getattr(config, "intermediate_size", 2432)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        max_position = getattr(config, "max_position_embeddings", 32768)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        spk_emb_size = getattr(config, "spk_emb_size", 192)
        text_hidden = config.hidden_size  # thinker hidden (the bridge comes from here)

        self.talker_hidden_size = talker_hidden
        self.audio_vocab_size = audio_vocab
        self.num_codebooks = 8
        self.config = config

        self.layers = nn.ModuleList(
            [
                TalkerBlock(
                    talker_hidden,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    max_position,
                    head_dim,
                    rms_norm_eps,
                    rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(talker_hidden, eps=rms_norm_eps)
        self.lm_head = TalkerHead(talker_hidden, audio_vocab)
        self.embed_tokens = TalkerEmbedding(audio_vocab, talker_hidden)
        # codec_proj: take audio-side hidden (post-embed_tokens) and project to talker space.
        # embed_proj: take text-side hidden (from thinker bridge) and project to talker space.
        self.codec_proj = nn.Sequential(
            nn.Linear(talker_hidden, talker_hidden),
            nn.GELU(),
            nn.Linear(talker_hidden, talker_hidden),
            RMSNorm(talker_hidden, eps=rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(text_hidden, text_hidden),
            nn.GELU(),
            nn.Linear(text_hidden, talker_hidden),
            RMSNorm(talker_hidden, eps=rms_norm_eps),
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        # Renamed from ``spk_proj`` to avoid the substring
        # ``k_proj`` (which the fork loader maps to ``qkv_proj``).
        self.speaker_proj = nn.Linear(spk_emb_size, talker_hidden, bias=False)

    def forward(
        self,
        bridge_states: torch.Tensor,
        audio_codes: torch.Tensor,
        positions: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        # Fork layers are 2D-friendly ([N, H]); flatten batch dim so the
        # rms_norm add-residual path broadcasts correctly.
        if bridge_states.dim() == 3:
            b, t, _ = bridge_states.shape
            bridge_states = bridge_states.reshape(b * t, -1)
        text_h = self.embed_proj(bridge_states) * self.text_scale  # [N, H]
        audio_h = self.codec_proj(self.embed_tokens(audio_codes)) * self.audio_scale
        if audio_h.dim() == 3:
            b, t, _ = audio_h.shape
            audio_h = audio_h.reshape(b * t, -1)
        hidden = text_h + audio_h  # elementwise add — vendor requires same length
        if spk_emb is not None:
            spk_h = self.speaker_proj(spk_emb)
            if spk_h.dim() == 2:
                spk_h = spk_h.reshape(-1, spk_h.shape[-1])
            hidden = torch.cat([spk_h, hidden], dim=0)
            positions = torch.cat([positions.new_zeros(1), positions + 1], dim=0)

        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
#  Stage factory (called by pipeline.py via dotted-path resolution)
# ---------------------------------------------------------------------------


def _talker_stage(deploy, args):
    """Stage 1 factory — loads ``MiniMindTalker`` via fork ``load_model``.

    The MTP decode loop that consumes ``TalkerInputPayload`` and emits
    audio codes lives in ``TalkerStage.__call__`` (see ``_engine.py``).
    """
    from ._engine import TalkerStage

    return TalkerStage(deploy, args)
