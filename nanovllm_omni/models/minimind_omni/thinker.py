"""MiniMind-O thinker stage.

Owns the ``ThinkerAttention``, ``ThinkerBlock``, ``MiniMindThinkerBackbone``
and ``MiniMindThinker`` model components using fork layers
(``nanovllm.layers.*``).

The backbone/wrapper split mirrors the upstream ``MiniMindForCausalLM``
shape (``self.model = MiniMindModel``) so the fork's weight loader can
land the HF ``model.embed_tokens.*`` / ``model.layers.*`` / ``model.norm.*``
checkpoint keys directly. The previous flat layout (``self.embed_tokens``,
``self.layers``, ``self.norm`` at the top level) silently failed to load
on RTX 3050 / torch 2.11: every forward returned a hidden state of all
zeros because no checkpoint key matched a top-level parameter, leaving
both ``embed_tokens`` and the tied ``lm_head`` at their init values, and
the QKV / gate-up projections likewise.

Public symbols: ``MiniMindThinker``, ``_thinker_stage``.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from torch import nn


class ThinkerAttention(nn.Module):
    """Multi-head attention with paged KV cache (via fork's Attention).

    ``packed_modules_mapping`` lives here so the fork's per-key weight
    loader walks down to ``self_attn`` and resolves the q/k/v merge
    against ``qkv_proj``. A model-level mapping would short-circuit
    non-merged keys (``embed_tokens``, ``norm``, ``q_norm``, ``k_norm``,
    ``o_proj``) and skip them.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

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
        q = q.reshape(-1, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class ThinkerMLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection.

    ``packed_modules_mapping`` lives here so the loader's per-key walk
    resolves the gate/up merge against ``gate_up_proj``.
    """

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

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


class ThinkerBlock(nn.Module):
    """Pre-norm transformer block."""

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
        self.self_attn = ThinkerAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            max_position,
            head_dim,
            rms_norm_eps,
            rope_theta,
        )
        self.mlp = ThinkerMLP(hidden_size, intermediate_size)
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


class MiniMindThinkerBackbone(nn.Module):
    """Embed + transformer stack + final norm.

    Lives at ``self.model`` on ``MiniMindThinker`` so the fork's weight
    loader can land HF keys with the ``model.*`` prefix without rewriting
    the loader. Mirrors the upstream ``MiniMindForCausalLM`` /
    ``MiniMindModel`` split used by the minimind-3o checkpoint.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        head_dim: int | None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                ThinkerBlock(
                    hidden_size,
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
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)


class MiniMindThinker(nn.Module):
    """MiniMind thinker stage using fork layers.

    Produces text logits for AR decoding and exposes bridge hidden states
    for the talker stage. No vendor code, no host reads, CUDA-graph safe.

    The ``self.model`` backbone holds ``embed_tokens`` / ``layers`` /
    ``norm`` so the fork loader matches the HF ``MiniMindForCausalLM``
    checkpoint layout. Per-submodule ``packed_modules_mapping`` lives on
    ``ThinkerAttention`` / ``ThinkerMLP`` (not on this class) so the
    loader's per-key walk only fires for q/k/v / gate / up keys; keys
    for the un-merged submodules (``embed_tokens``, ``norm``, ``o_proj``,
    ``q_norm``, ``k_norm``, ``down_proj``, layernorms) fall through to
    the plain-loader branch.
    """

    def __init__(self, config) -> None:
        """HF config in, same contract as ``Qwen3ForCausalLM(hf_config)``.

        ``ModelRunner`` always calls ``model_class(hf_config)``. Keyword
        construction is gone on purpose — a second signature would drift
        from the fork.
        """
        super().__init__()
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = math.ceil(hidden_size * 8 / 3 / 256) * 256
        max_position = getattr(config, "max_position_embeddings", 4096)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        rope_theta = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim", None)
        self.bridge_layer = getattr(config, "bridge_layer", 3)
        self.num_audio_heads = getattr(config, "num_audio_heads", 8)
        self.audio_vocab_size = getattr(config, "audio_vocab_size", 2048)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = config

        # Backbone at ``self.model`` — same shape as upstream
        # ``MiniMindForCausalLM`` so the fork's weight loader can land
        # HF ``model.embed_tokens.*`` / ``model.layers.*`` / ``model.norm.*``
        # keys directly.
        self.model = MiniMindThinkerBackbone(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_position=max_position,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            head_dim=head_dim,
        )
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        # minimind-3o ships with ``tie_word_embeddings=True``; alias
        # ``lm_head`` to the backbone's ``embed_tokens`` so the tied
        # checkpoint weight lands in both places.
        self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.audio_head = nn.Linear(
            hidden_size, self.num_audio_heads * self.audio_vocab_size, bias=False
        )
        # Bridge hidden state capture that survives CUDA-graph replay.
        #
        # The old design used a plain Python attribute (``_bridge_hidden``)
        # which doesn't survive CUDA-graph replay — Python attribute
        # assignment doesn't get re-executed when the captured graph
        # re-runs its kernels. The fork's ``ModelRunner`` enables CUDA-graph
        # capture for decode steps when ``enforce_eager=False``; in that mode
        # we need a registered buffer that the captured graph can write
        # into at a fixed memory address.
        #
        # Layout: row 0 is the decode-step bridge (last token per step,
        # overwritten on every replay); rows [0:seq_len] are also written
        # during prefill in eager mode (``is_prefill`` branch in fork's
        # ``ModelRunner``). ``decode_minimind`` slices
        # ``_bridge_buffer[:1]`` after each decode replay and
        # ``_bridge_buffer[:seq_len]`` once after prefill.
        #
        # Size: ``max_position`` rows × ``hidden_size`` columns. Matches
        # the longest sequence the model can handle; fork-allocated KV
        # tensors are sized similarly.
        self.register_buffer(
            "_bridge_buffer",
            torch.zeros(max_position, hidden_size),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """ModelRunner-compatible forward: returns hidden_states."""
        hidden_states = self.model.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.model.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if i == self.bridge_layer:
                seq_len = hidden_states.size(0)
                if seq_len > 1:
                    # Prefill: eager in fork's ModelRunner
                    # (``is_prefill or self.enforce_eager`` short-circuits
                    # graph capture). Write the full hidden_states to the
                    # buffer's first ``seq_len`` rows; the decode loop
                    # reads them once via ``get_bridge_hidden``.
                    self._bridge_buffer[:seq_len].copy_(hidden_states)
                else:
                    # Decode (CUDA-graph captured): write the single
                    # last-token row at the buffer's row 0 — a fixed
                    # address that survives replay. Each replay
                    # overwrites row 0; ``decode_minimind`` reads it out
                    # between replays via ``get_bridge_hidden``.
                    self._bridge_buffer[0].copy_(hidden_states[0])
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_bridge_hidden(self) -> torch.Tensor | None:
        """Return the buffer holding the most-recent bridge rows.

        Decode consumers slice ``[:1]`` per step; the prefill consumer
        slices ``[:seq_len]`` once. Row 0 is the just-replayed decode
        row (or the first row of a prefill pass).
        """
        return self._bridge_buffer

    def get_audio_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.audio_head(hidden_states)
        return logits.view(-1, self.num_audio_heads, self.audio_vocab_size)


# ---------------------------------------------------------------------------
#  Stage factory (called by pipeline.py via tuple registration)
# ---------------------------------------------------------------------------


def _thinker_stage(deploy, args, model_class=None):
    """Stage 0 factory — fork ``ModelRunner`` wiring.

    Returns a ``ThinkerStage`` whose ``ModelRunner`` already has its KV
    tensors allocated and CUDA graphs captured. The actual decode loop
    (prefill + autoregressive text decode + bridge extraction into
    ``ThinkerStageOutput``) is Phase 4 territory; see
    ``docs/dev/nanovllm-omni-rewrite.md`` §7.
    """
    from .stage import ThinkerStage

    return ThinkerStage(deploy, args, model_class=model_class)
