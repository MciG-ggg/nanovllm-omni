"""MiniMind talker on MLX: the model.

The MLX twin of ``talker.py``. The staggered decode loop that drives it
lives in ``talker_mlx_runner``; the schedule, the emission gate and the
sampling recipe are the ones in ``talker_torch``; only the array ops
are MLX's.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .talker_recipe import NUM_CODEBOOKS

_ADAPTER_RANK = 256


@dataclass(frozen=True)
class MlxTalkerConfig:
    """Talker architecture fields read from the converted ``config.json``."""

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
    def from_checkpoint(cls, model_dir: str) -> MlxTalkerConfig:
        """Read the talker fields from ``<model_dir>/config.json``."""
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


class _RMSNorm(nn.Module):
    """Vendor RMSNorm: normalize in fp32, scale, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        normed = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (self.weight.astype(mx.float32) * normed).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the two halves of the head dim, the vendor's ``rotate_half``."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary(
    query: mx.array, key: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Vendor RoPE: ``cos``/``sin`` are ``[T, head_dim]``, broadcast per head."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    query = (query * cos + _rotate_half(query) * sin).astype(query.dtype)
    key = (key * cos + _rotate_half(key) * sin).astype(key.dtype)
    return query, key


def _rope_tables(head_dim: int, max_positions: int, rope_theta: float) -> tuple[mx.array, mx.array]:
    """Vendor ``precompute_freqs_cis``; no YaRN branch because this checkpoint has none."""
    freqs = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2)[: head_dim // 2] / head_dim))
    angles = mx.outer(mx.arange(max_positions), freqs)
    freqs_cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    freqs_sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    return freqs_cos, freqs_sin


class _LayerKVCache:
    """One layer's settled keys/values, ``[B, num_key_value_heads, length, head_dim]``."""

    def __init__(self) -> None:
        self.key: mx.array | None = None
        self.value: mx.array | None = None


class MlxTalkerKVCache:
    """Windowed KV cache for one request (ADR 0022)."""

    def __init__(self, num_layers: int) -> None:
        self.layers = [_LayerKVCache() for _ in range(num_layers)]

    @property
    def length(self) -> int:
        """Settled columns, which is also the absolute RoPE offset of the next window."""
        first = self.layers[0].key
        return 0 if first is None else first.shape[2]


def _append_to_cache(
    cache: _LayerKVCache, key: mx.array, value: mx.array
) -> tuple[mx.array, mx.array]:
    """Merge this step's columns into ``cache`` and keep only the settled ones."""
    if cache.key is not None:
        key = mx.concatenate([cache.key, key], axis=2)
        value = mx.concatenate([cache.value, value], axis=2)
    keep = max(0, key.shape[2] - (NUM_CODEBOOKS - 1))
    cache.key = key[:, :, :keep]
    cache.value = value[:, :, :keep]
    return key, value


class _TalkerAttention(nn.Module):
    """Grouped-query attention with per-head RMSNorm and RoPE."""

    def __init__(self, config: MlxTalkerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        hidden = config.talker_hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: _LayerKVCache | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary(query, key, cos, sin)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        key = mx.repeat(key, self.num_key_value_groups, axis=1)
        value = mx.repeat(value, self.num_key_value_groups, axis=1)
        if cache is None:
            if seq_len > 1:
                output = mx.fast.scaled_dot_product_attention(
                    query, key, value, scale=self.scale, mask="causal"
                )
            else:
                # Vendor takes its explicit path when the fused kernel does not
                # apply; at one query the causal mask is empty.
                scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
                output = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype) @ value
        else:
            key, value = _append_to_cache(cache, key, value)
            # MLX's causal mask is lower-right aligned, which is the window's
            # mask: the cached columns are visible to every window query.
            output = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale, mask="causal"
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class _TalkerMLP(nn.Module):
    """SwiGLU feed-forward, matching the checkpoint's ``mlp.*`` keys."""

    def __init__(self, config: MlxTalkerConfig) -> None:
        super().__init__()
        hidden = config.talker_hidden_size
        self.gate_proj = nn.Linear(hidden, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _TalkerBlock(nn.Module):
    """Pre-norm decoder block: attention residual, then MLP residual."""

    def __init__(self, config: MlxTalkerConfig) -> None:
        super().__init__()
        self.self_attn = _TalkerAttention(config)
        self.input_layernorm = _RMSNorm(config.talker_hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.talker_hidden_size, config.rms_norm_eps)
        self.mlp = _TalkerMLP(config)

    def __call__(
        self,
        hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache: _LayerKVCache | None = None,
    ) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _TalkerEmbedding(nn.Module):
    """Average of the eight codebook embeddings, each with its own adapter."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = [
            nn.Sequential(
                nn.Embedding(num_embeddings, _ADAPTER_RANK),
                nn.GELU(),
                nn.Linear(_ADAPTER_RANK, embedding_dim, bias=False),
            )
            for _ in range(NUM_CODEBOOKS)
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        """Map ``[B, 8, T]`` audio-code ids to ``[B, T, embedding_dim]``."""
        base = self.base(codes)
        total = base[:, 0] + self.adapters[0](codes[:, 0])
        for index in range(1, NUM_CODEBOOKS):
            total = total + base[:, index] + self.adapters[index](codes[:, index])
        return total / NUM_CODEBOOKS


class _TalkerHead(nn.Module):
    """Eight codebook heads: one shared projection plus one low-rank adapter each."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = [
            nn.Sequential(
                nn.Linear(in_features, _ADAPTER_RANK, bias=False),
                nn.GELU(),
                nn.Linear(_ADAPTER_RANK, out_features, bias=False),
            )
            for _ in range(NUM_CODEBOOKS)
        ]

    def __call__(self, hidden: mx.array) -> list[mx.array]:
        base = self.base(hidden)
        return [base + adapter(hidden) for adapter in self.adapters]


class MlxTalker(nn.Module):
    """Bridge states + audio codes in, eight last-position logit tensors out."""

    def __init__(self, config: MlxTalkerConfig, dtype: Any = mx.float16) -> None:
        super().__init__()
        self.config = config
        hidden = config.talker_hidden_size
        self.layers = [_TalkerBlock(config) for _ in range(config.num_layers)]
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.lm_head = _TalkerHead(hidden, config.audio_vocab_size)
        self.embed_tokens = _TalkerEmbedding(config.audio_vocab_size, hidden)
        self.codec_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            _RMSNorm(hidden, config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.thinker_hidden_size, config.thinker_hidden_size),
            nn.GELU(),
            nn.Linear(config.thinker_hidden_size, hidden),
            _RMSNorm(hidden, config.rms_norm_eps),
        )
        self.text_scale = mx.array(3.0)
        self.audio_scale = mx.array(1.0)
        self.spk_proj = nn.Linear(config.spk_emb_size, hidden, bias=False)
        freqs_cos, freqs_sin = _rope_tables(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.freqs_cos = freqs_cos.astype(dtype)
        self.freqs_sin = freqs_sin.astype(dtype)

    def __call__(
        self,
        bridge: mx.array,
        audio_codes: mx.array,
        spk_emb: mx.array | None = None,
        cache: MlxTalkerKVCache | None = None,
    ) -> list[mx.array]:
        """Return the eight codebook logits for the last sequence position."""
        embeds = self.embed_tokens(audio_codes)
        if spk_emb is not None:
            if spk_emb.ndim == 1:
                spk_emb = spk_emb[None]
            mask = audio_codes[:, 0, :] == self.config.audio_spk_token
            embeds = mx.where(mask[..., None], self.spk_proj(spk_emb)[:, None, :], embeds)
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
        return self.lm_head(self.norm(hidden[:, -1:, :]))


def load_mlx_talker(model_dir: str, *, dtype: str = "float16", device: str = "gpu") -> MlxTalker:
    """Load the talker weights from a converted MiniMind-O talker directory."""
    mx.set_default_device(getattr(mx, device))
    target_dtype = getattr(mx, dtype)
    model = MlxTalker(MlxTalkerConfig.from_checkpoint(model_dir), dtype=target_dtype)
    weights = mx.load(os.path.join(model_dir, "model.safetensors"))
    sequential_prefixes = ["codec_proj", "embed_proj"]
    for index in range(NUM_CODEBOOKS):
        sequential_prefixes.extend((f"embed_tokens.adapters.{index}", f"lm_head.adapters.{index}"))
    normalized_weights = {}
    for key, value in weights.items():
        for prefix in sequential_prefixes:
            if key.startswith(f"{prefix}."):
                key = key.replace(f"{prefix}.", f"{prefix}.layers.", 1)
                break
        normalized_weights[key] = value.astype(target_dtype)
    normalized_weights.setdefault("freqs_cos", model.freqs_cos)
    normalized_weights.setdefault("freqs_sin", model.freqs_sin)
    model.load_weights(list(normalized_weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


__all__ = [
    "MlxTalker",
    "MlxTalkerConfig",
    "MlxTalkerKVCache",
    "load_mlx_talker",
]
