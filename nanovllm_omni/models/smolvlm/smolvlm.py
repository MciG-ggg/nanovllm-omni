"""SmolVLM-500M-Instruct full model: SigLIP vision + SmolLM2 text decoder.

Per ``docs/dev/nanovllm-omni-smolvlm-ar-migration.md`` ADR-016 (revised,
2026-09): fork-style text decoder for paged KV + CUDA graph; vanilla
nn.Module for vision + connector (one-shot encode, no KV cache, no
paged benefit).

Submodule naming mirrors HF checkpoint keys
``model.vision_model.*`` / ``model.connector.*`` /
``model.language_model.*`` / ``lm_head.*`` so the fork ``load_model``
(with default ``prefix=""``) loads HF weights directly — no
``/tmp`` weight slicing, no prefix gymnastics.

Stages (``stage.py``) drive vision + connector on prefill and the
fork ``StageRunner`` for the AR decode loop; ``forward`` here is the
text decoder only.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

# Fork layer imports are lazy: pulling ``nanovllm.layers.attention``
# requires ``triton`` (CUDA-only); defer until first model construction
# so CPU-only hosts can import this module without a GPU toolchain.


# ---------------------------------------------------------------------------
# Text decoder: SmolLM2 (Llama-style, qkv_bias=False, no per-head RMSNorm).
# ---------------------------------------------------------------------------


class SmolLM2Attention(nn.Module):
    """Llama-style attention: GQA + RoPE + paged attention (fork Attention)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        # Lazy fork imports — see module docstring.
        from nanovllm.layers.attention import Attention
        from nanovllm.layers.linear import (
            QKVParallelLinear,
            RowParallelLinear,
        )
        from nanovllm.layers.rotary_embedding import get_rope

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
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class SmolLM2MLP(nn.Module):
    """SwiGLU MLP (Llama-style: gate_up_proj fused, down_proj separate)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        from nanovllm.layers.activation import SiluAndMul
        from nanovllm.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class SmolLM2DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        from nanovllm.layers.layernorm import RMSNorm

        self.self_attn = SmolLM2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 100000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = SmolLM2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SmolLM2Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        from nanovllm.layers.embed_head import VocabParallelEmbedding
        from nanovllm.layers.layernorm import RMSNorm

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SmolLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SmolLM2ForCausalLM(nn.Module):
    """Standalone SmolLM2 wrapper used inside ``SmolVLMModel.language_model``.

    packed_modules_mapping declares the fused-proj name rewrites so the
    fork ``load_model`` can land the per-head ``q_proj`` /
    ``k_proj`` / ``v_proj`` checkpoints into the fused
    ``qkv_proj.weight`` shards.
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
        from nanovllm.layers.embed_head import ParallelLMHead

        self.model = SmolLM2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Connector: SigLIP hidden (768) -> SmolLM2 hidden (576) via GeLU + Linear.
# Mirrors HF SmolVLMConnector.modality_projection key layout.
# ---------------------------------------------------------------------------


class SmolVLMConnector(nn.Module):
    """Vision dim → text dim projection (one-shot, no fork building blocks).

    SmolVLM-500M uses ``modality_projection = GeLU + Linear`` after a
    pixel-shuffle step that merges spatial patches; we apply the
    shuffle on the stage side (see ``stage.py``) and keep this module
    linear-only. The submodule name ``modality_projection`` (an
    ``nn.Sequential`` of GeLU + Linear) mirrors HF checkpoint key
    ``model.connector.modality_projection.0/1.*``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        scale_factor = getattr(config, "scale_factor", 2)
        v_dim = config.vision_config.hidden_size
        t_dim = config.text_config.hidden_size
        self.scale_factor = scale_factor
        self.text_dim = t_dim
        self.modality_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(v_dim * (scale_factor**2), t_dim, bias=True),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """image_features: [num_patches_after_shuffle, v_dim * scale_factor**2]"""
        return self.modality_projection(image_features)


# ---------------------------------------------------------------------------
# Top-level: SmolVLMForConditionalGeneration.
# Submodule structure mirrors HF checkpoint keys so loader with
# prefix="" lands weights directly.
# ---------------------------------------------------------------------------


def _resolve_siglip_backbone():
    """Return the HF vision backbone class for the loaded transformers version.

    transformers 5.15 renamed ``SiglipVisionTransformer`` ->
    ``SiglipVisionModel``; older versions still expose the Transformer
    name. Module-level resolution keeps the symbol patchable for tests
    that need to mock the heavy PreTrainedConfig construction path.
    """
    try:
        from transformers.models.siglip.modeling_siglip import (
            SiglipVisionModel as BackboneCls,  # noqa: N813
        )
    except ImportError:  # pragma: no cover - very old transformers
        from transformers.models.siglip.modeling_siglip import (
            SiglipVisionTransformer as BackboneCls,  # noqa: N813
        )
    return BackboneCls


class SmolVLMModel(nn.Module):
    """Holds vision_model + connector + language_model (mirror of HF keys)."""

    def __init__(self, config) -> None:
        super().__init__()
        # Lazy + patchable: construction calls ``_resolve_siglip_backbone()``
        # each time so tests can ``monkeypatch.setattr(module,
        # "_resolve_siglip_backbone", lambda: MagicMock)`` without
        # touching the real transformers import.
        self.vision_model = _resolve_siglip_backbone()(config.vision_config)
        self.connector = SmolVLMConnector(config)
        self.language_model = SmolLM2ForCausalLM(config.text_config)


class SmolVLMForConditionalGeneration(nn.Module):
    """SmolVLM full model. forward is text-decoder only; stage handles vision."""

    packed_modules_mapping = SmolLM2ForCausalLM.packed_modules_mapping

    def __init__(self, config) -> None:
        super().__init__()
        from nanovllm.layers.embed_head import ParallelLMHead

        self.model = SmolVLMModel(config)
        text_config = config.text_config
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.language_model.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.language_model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "SmolLM2Attention",
    "SmolLM2DecoderLayer",
    "SmolLM2ForCausalLM",
    "SmolLM2MLP",
    "SmolLM2Model",
    "SmolVLMConnector",
    "SmolVLMForConditionalGeneration",
    "SmolVLMModel",
    "_resolve_siglip_backbone",
]
