"""MiniMind-O talker stage.

Owns the ``TalkerAttention``, ``TalkerBlock``, ``TalkerHead``,
``TalkerEmbedding``, and ``MiniMindTalker`` model components.

Architecture matches the vendored HF ``MiniMindOmni.talker`` (see
``model_omni.py`` in the snapshot): a 4-layer pre-norm transformer
with 8 codebook heads (one base Linear + 8 LoRA-style adapters each)
that together predict the 8 audio codebook channels frame by frame.

Public symbols: ``MiniMindTalker``, ``_talker_stage``.

ponytail: matches vendor 1:1 for the prep + body + lm_head shape; the
attention layer delegates to fork's paged ``Attention`` so the
``store_kvcache`` Triton kernel handles the talker's
``D = num_kv_heads * head_dim = 192`` (not a power of two — the
kernel now pads to next-pow2 internally via ``D_PAD``).
"""

from __future__ import annotations

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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base_out = self.base(x)
        return (
            sum(base_out[:, i, :] + self.adapters[i](x[:, i, :]) for i in range(self.num_layers))
            / self.num_layers
        )


# ---------------------------------------------------------------------------
# Inner transformer block (uses fork's paged ``Attention``).
# ---------------------------------------------------------------------------


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
        # Fork paged Attention: handles GQA replication, RoPE, paged KV
        # store, and the prefill/decode kernel split. The local
        # _EagerAttention that previously wrapped stdlib SDPA is gone;
        # it skipped the paged path entirely, which forced the talker
        # off fork's KV-cache machinery and onto a per-step O(T²) loop.
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # Fork Attention returns ``[N, H_q, D]`` (matches the SDPA helper
        # and flash_attn varlen output); flatten to ``[N, H_q*D]`` for
        # RowParallelLinear, which sees only the trailing dim.
        return self.o_proj(o.flatten(1, -1))


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
    (``TalkerModule`` in ``model_omni.py``).

    The forward entry has two shapes:
      - **runner-compatible**: ``forward(input_ids=None, positions=...,
        inputs_embeds=...)`` returns hidden_states. ``compute_logits``
        applies the 8-head ``TalkerHead``. This shape is what
        ``ModelRunner.run_model`` expects when the caller has already
        done input prep (thinker-style fork usage) or pre-embedded
        the input (talker-style multimodal prep).
      - **legacy full path**: ``forward(bridge_states=..., audio_codes=...,
        positions=..., spk_emb=None)`` does prep + body + lm_head in
        one call and returns a list of 8 logits tensors. Used by
        tests and any code that wants the logits directly.

    Both paths share ``prepare_inputs_embeds`` (the ``embed_proj`` /
    ``codec_proj`` / ``speaker_proj`` work) and ``body`` (the 4
    transformer blocks + final norm).
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

    # ---- Multi-shape forward (runner + legacy) ----

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **legacy_kwargs,
    ):
        """Dispatch by kwarg shape — see class docstring."""
        if inputs_embeds is not None:
            # Runner-compatible: prep already happened; just run body
            # and return hidden_states. ``compute_logits`` is called by
            # ``ModelRunner.run_model`` next, applying ``TalkerHead``.
            return self.body(positions, inputs_embeds)
        if "bridge_states" in legacy_kwargs:
            # Legacy full path: prep + body + lm_head → list[Tensor]
            return self._forward_full(
                bridge_states=legacy_kwargs["bridge_states"],
                audio_codes=legacy_kwargs.get("audio_codes"),
                positions=positions,
                spk_emb=legacy_kwargs.get("spk_emb"),
            )
        raise ValueError(
            "MiniMindTalker.forward requires either ``inputs_embeds`` "
            "(runner path) or ``bridge_states`` in kwargs (legacy path)."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Apply the 8-head ``TalkerHead``; returns one [N, vocab] tensor per codebook."""
        return self.lm_head(hidden_states)

    def _forward_full(
        self,
        bridge_states: torch.Tensor,
        audio_codes: torch.Tensor,
        positions: torch.Tensor,
        spk_emb: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Prep + body + lm_head, returns list of 8 logits tensors."""
        embeds, positions = self.prepare_inputs_embeds(
            bridge_states, audio_codes, spk_emb, positions
        )
        hidden = self.body(positions, embeds)
        return self.compute_logits(hidden)

    # ---- Decomposed methods (mirror vllm-omni's preprocess / forward split) ----

    def prepare_inputs_embeds(
        self,
        bridge_states: torch.Tensor,
        audio_codes: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run text-side ``embed_proj`` + audio-side ``codec_proj`` + (optional) speaker prefix.

        Returns ``(hidden, positions)``. ``positions`` is shifted by +1
        when ``spk_emb`` is provided (the speaker token sits at position
        0; the original positions move to 1..T).

        Caller composes this with :meth:`body` and :meth:`compute_logits`
        for full control, or just calls :meth:`forward` for the legacy
        all-in-one path.
        """
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
        if positions is None:
            positions = torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.long)
        if spk_emb is not None:
            spk_h = self.speaker_proj(spk_emb)
            if spk_h.dim() == 2:
                spk_h = spk_h.reshape(-1, spk_h.shape[-1])
            hidden = torch.cat([spk_h, hidden], dim=0)
            positions = torch.cat([positions.new_zeros(1), positions + 1], dim=0)
        return hidden, positions

    def body(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the 4 transformer blocks + final norm, return post-norm hidden."""
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
#  Stage factory (called by pipeline.py via dotted-path resolution)
# ---------------------------------------------------------------------------


def _talker_stage(deploy, args):
    """Stage 1 factory — defers heavy model load to ``TalkerStage.__init__``.

    The MTP decode loop that consumes ``TalkerInputPayload`` and emits
    audio codes lives in ``TalkerStage.__call__`` (see ``stage.py``).
    """
    from .stage import TalkerStage

    return TalkerStage(deploy, args)
