"""MiniMind talker: dense rewrite of the vendor ``TalkerModule`` (ADR 0017/0022)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

# TalkerHead / TalkerEmbedding defaults in the vendor class: one head per
# audio codebook, each a rank-256 adapter over a shared base.
NUM_CODEBOOKS = 8
_ADAPTER_RANK = 256
_WEIGHT_PREFIX = "talker."


class RMSNorm(nn.Module):
    """Vendor RMSNorm: normalize in fp32, scale, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendor RoPE: ``cos``/``sin`` are ``[T, head_dim]``, broadcast per head."""
    query = (query * cos.unsqueeze(1) + _rotate_half(query) * sin.unsqueeze(1)).to(query.dtype)
    key = (key * cos.unsqueeze(1) + _rotate_half(key) * sin.unsqueeze(1)).to(key.dtype)
    return query, key


def _repeat_key_value(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Repeat each KV head ``num_groups`` times (vendor ``repeat_kv``)."""
    if num_groups == 1:
        return x
    batch, seq_len, num_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(batch, seq_len, num_heads, num_groups, head_dim)
    return x.reshape(batch, seq_len, num_heads * num_groups, head_dim)


def _rope_tables(
    head_dim: int, max_positions: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendor ``precompute_freqs_cis``; no YaRN branch because this checkpoint has none."""
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    angles = torch.outer(torch.arange(max_positions), freqs).float()
    freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return freqs_cos, freqs_sin


class _LayerKVCache:
    """One layer's settled keys/values, ``[B, length, num_key_value_heads, head_dim]``."""

    def __init__(self) -> None:
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None


class TalkerKVCache:
    """Windowed KV cache for one request (ADR 0022)."""

    def __init__(self, num_layers: int) -> None:
        self.layers = [_LayerKVCache() for _ in range(num_layers)]

    @property
    def length(self) -> int:
        """Settled columns, which is also the absolute RoPE offset of the next window."""
        first = self.layers[0].key
        return 0 if first is None else first.shape[1]


def _append_to_cache(
    cache: _LayerKVCache, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge this step's columns into ``cache`` and keep only the settled ones.

    The window carries the newest columns, which a later step recomputes, so
    the trailing ``NUM_CODEBOOKS - 1`` columns stay out of the cache.
    """
    if cache.key is not None:
        key = torch.cat([cache.key, key], dim=1)
        value = torch.cat([cache.value, value], dim=1)
    keep = max(0, key.shape[1] - (NUM_CODEBOOKS - 1))
    cache.key = key[:, :keep].contiguous()
    cache.value = value[:, :keep].contiguous()
    return key, value


def _window_attention_mask(window_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    """Causal inside the window; all cached columns are visible to all of it.

    ``[window_len, key_len]``, True where a query may attend.
    """
    first_cached = key_len - window_len
    queries = first_cached + torch.arange(window_len, device=device)[:, None]
    return torch.arange(key_len, device=device)[None, :] <= queries


class TalkerAttention(nn.Module):
    """Grouped-query attention with per-head RMSNorm and RoPE, dense full sequence."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(
            config.talker_hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.talker_hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.talker_hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.talker_hidden_size, bias=False
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: _LayerKVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary(query, key, cos, sin)
        query = query.transpose(1, 2)
        if cache is None:
            key = _repeat_key_value(key, self.num_key_value_groups).transpose(1, 2)
            value = _repeat_key_value(value, self.num_key_value_groups).transpose(1, 2)
            if seq_len > 1:
                output = functional.scaled_dot_product_attention(query, key, value, is_causal=True)
            else:
                # Vendor takes its explicit path when the fused kernel does not
                # apply; at one query the causal mask is empty, so a plain
                # fp32 softmax reproduces it.
                scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
                output = functional.softmax(scores.float(), dim=-1).type_as(query) @ value
        else:
            key, value = _append_to_cache(cache, key, value)
            key = _repeat_key_value(key, self.num_key_value_groups).transpose(1, 2)
            value = _repeat_key_value(value, self.num_key_value_groups).transpose(1, 2)
            # The fused causal path assumes column 0 is the window's first query.
            # Here the window sits at the end of the cache, so the mask is explicit.
            mask = _window_attention_mask(seq_len, key.shape[2], hidden.device)
            output = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class TalkerMLP(nn.Module):
    """SwiGLU feed-forward, matching the checkpoint's ``mlp.*`` keys."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.talker_hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.talker_hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.talker_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TalkerBlock(nn.Module):
    """Pre-norm decoder block: attention residual, then MLP residual."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.self_attn = TalkerAttention(config)
        self.input_layernorm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.mlp = TalkerMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: _LayerKVCache | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TalkerEmbedding(nn.Module):
    """Average of the eight codebook embeddings, each with its own adapter."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Embedding(num_embeddings, _ADAPTER_RANK),
                nn.GELU(),
                nn.Linear(_ADAPTER_RANK, embedding_dim, bias=False),
            )
            for _ in range(NUM_CODEBOOKS)
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Map ``[B, 8, T]`` audio-code ids to ``[B, T, embedding_dim]``."""
        base = self.base(codes)
        total = sum(
            base[:, index, :] + self.adapters[index](codes[:, index, :])
            for index in range(NUM_CODEBOOKS)
        )
        return total / NUM_CODEBOOKS


class TalkerHead(nn.Module):
    """Eight codebook heads: one shared projection plus one low-rank adapter each."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features, _ADAPTER_RANK, bias=False),
                nn.GELU(),
                nn.Linear(_ADAPTER_RANK, out_features, bias=False),
            )
            for _ in range(NUM_CODEBOOKS)
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        base = self.base(hidden)
        return [base + adapter(hidden) for adapter in self.adapters]


@dataclass(frozen=True)
class TalkerConfig:
    """Talker architecture fields read from the checkpoint's ``config.json``."""

    thinker_hidden_size: int
    talker_hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    audio_vocab_size: int
    spk_emb_size: int
    audio_spk_token: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_checkpoint(cls, model_dir: str) -> TalkerConfig:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            thinker_hidden_size=raw["hidden_size"],
            talker_hidden_size=raw["talker_hidden_size"],
            num_layers=raw["num_talker_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            intermediate_size=raw["intermediate_size"],
            audio_vocab_size=raw["audio_vocab_size"],
            spk_emb_size=raw["spk_emb_size"],
            audio_spk_token=raw["audio_spk_token"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
        )


class MiniMindTalker(nn.Module):
    """Bridge states + audio codes in, eight last-position logit tensors out."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TalkerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.thinker_hidden_size, config.thinker_hidden_size),
            nn.GELU(),
            nn.Linear(config.thinker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)
        freqs_cos, freqs_sin = _rope_tables(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        bridge: torch.Tensor,
        audio_codes: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
        cache: TalkerKVCache | None = None,
    ) -> list[torch.Tensor]:
        """Return the eight codebook logits for the last sequence position.

        With ``cache`` the inputs are only the newest window and the cache
        holds everything before it; without one they are the whole sequence.
        """
        embeds = self.embed_tokens(audio_codes)
        if spk_emb is not None:
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            mask = (audio_codes[:, 0, :] == self.config.audio_spk_token).unsqueeze(-1)
            embeds = torch.where(mask, self.spk_proj(spk_emb).unsqueeze(1), embeds)
        hidden = (
            self.embed_proj(bridge) * self.text_scale + self.codec_proj(embeds) * self.audio_scale
        )
        # The window's absolute start is the cache length, so RoPE is indexed
        # by position rather than from zero (ADR 0022).
        start = 0 if cache is None else cache.length
        seq_len = hidden.shape[1]
        cos = self.freqs_cos[start : start + seq_len]
        sin = self.freqs_sin[start : start + seq_len]
        layer_caches = cache.layers if cache is not None else [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            hidden = layer(hidden, cos, sin, layer_cache)
        # The driver samples only ``logits[0, -1, :]`` (ADR 0022: a P2-level
        # shape change, not a bitwise one).
        return self.lm_head(self.norm(hidden[:, -1:, :]))


def load_talker(
    model_dir: str, *, device: str, dtype: torch.dtype = torch.float16
) -> MiniMindTalker:
    """Load the talker weights from a MiniMind-O checkpoint directory."""
    model = MiniMindTalker(TalkerConfig.from_checkpoint(model_dir))
    state = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(
        {
            name[len(_WEIGHT_PREFIX) :]: value
            for name, value in state.items()
            if name.startswith(_WEIGHT_PREFIX)
        },
        strict=True,
    )
    return model.eval().to(device=device, dtype=dtype)


__all__ = [
    "MiniMindTalker",
    "TalkerConfig",
    "TalkerKVCache",
    "load_talker",
]
