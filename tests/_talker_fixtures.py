"""Shared test fixtures for the MiniMind-Omni talker.

The real ``TalkerModule`` lives in the vendored HF cache and depends on
the full ``MiniMindOmni`` checkpoint (no offline-friendly mock). The
fixtures below provide a tiny synthetic replacement with the same public
shape (``layers``, ``norm``, ``lm_head``, ``embed_tokens``, ``codec_proj``,
``embed_proj``, ``text_scale``, ``audio_scale``, ``spk_proj``,
``freqs_cos``, ``freqs_sin``) so the wrapper class can be exercised
without downloading the real model.

All sizes are tiny (hidden=8, vocab=16) so unit tests stay sub-second on
the no-torch CI variant when torch is skipped via ``importorskip``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Tiny Block + TalkerHead + TalkerEmbedding fakes
# ---------------------------------------------------------------------------


class _FakeMiniMindBlock(nn.Module):
    """One-projection stand-in for ``MiniMindBlock``.

    Vendored HF Block signature:
        ``forward(hidden_states, position_embeddings, past_key_value=None,
        use_cache=False, attention_mask=None) -> (hidden_states, present)``.
    We collapse attention + MLP into a single Linear so the wrapper's
    layer loop runs but never tries to actually attend.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)  # identity-ish so successive layers compose

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        past_key_value: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        return self.proj(hidden_states), None


class _FakeTalkerHead(nn.Module):
    """Stand-in for ``MiniMindOmniTalkerHead`` (TalkerHead in the vendored model).

    One ``base`` Linear + ``num_layers`` adapters. Each adapter is
    ``Linear -> GELU -> Linear`` (matches the vendored structure with
    bias=False on all linears). Tiny rank keeps the param count small.
    """

    def __init__(
        self, hidden_size: int, vocab_size: int, num_layers: int = 8, rank: int = 4
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(hidden_size, vocab_size, bias=False)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, rank, bias=False),
                    nn.GELU(),
                    nn.Linear(rank, vocab_size, bias=False),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class _FakeTalkerEmbedding(nn.Module):
    """Stand-in for the vendored ``TalkerEmbedding``.

    Structure: one ``base`` Embedding + ``num_layers`` adapters (each
    ``Embedding -> GELU -> Linear``). Matches the vllm-omni ``MiniMindOmniTalkerEmbedding``
    interface so ``forward(audio_ids) -> Tensor`` matches.
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int = 8, rank: int = 4
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(vocab_size, hidden_size)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(vocab_size, rank),
                    nn.GELU(),
                    nn.Linear(rank, hidden_size, bias=False),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, audio_ids: torch.Tensor) -> torch.Tensor:
        # audio_ids: [B, num_layers, T]
        base_out = self.base(audio_ids)  # [B, num_layers, T, hidden]
        return (
            sum(
                base_out[:, i, :, :] + self.adapters[i](audio_ids[:, i, :])
                for i in range(self.num_layers)
            )
            / self.num_layers
        )


# ---------------------------------------------------------------------------
# RMSNorm + RoPE buffers (mirrors the vendored ``RMSNorm`` / RoPE shape)
# ---------------------------------------------------------------------------


class _FakeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm.float()).type_as(x)


def _fake_freqs_cis(
    dim: int, end: int = 64, rope_base: float = 1e6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny RoPE precompute (matches vendored ``precompute_freqs_cis`` shape)."""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin


# ---------------------------------------------------------------------------
# Fake HF TalkerModule + bundle
# ---------------------------------------------------------------------------


def make_fake_hf_talker(
    *,
    hidden_size: int = 8,
    vocab_size: int = 16,
    text_hidden_size: int | None = None,
    num_code_layers: int = 8,
    num_hidden_layers: int = 2,
    max_position_embeddings: int = 32,
    rms_norm_eps: float = 1e-5,
    spk_emb_size: int = 4,
) -> nn.Module:
    """Build a tiny nn.Module mimicking the vendored HF ``TalkerModule``."""
    text_hidden_size = text_hidden_size or hidden_size
    head_dim = max(2, hidden_size // 2)
    cos, sin = _fake_freqs_cis(head_dim, end=max_position_embeddings)

    class _InnerRMSNorm(_FakeRMSNorm):
        pass

    inner = nn.Module()
    # Sub-modules so ``load_weights`` can find them by name.
    inner.layers = nn.ModuleList(
        [_FakeMiniMindBlock(hidden_size) for _ in range(num_hidden_layers)]
    )
    inner.norm = _InnerRMSNorm(hidden_size, eps=rms_norm_eps)
    inner.lm_head = _FakeTalkerHead(hidden_size, vocab_size, num_layers=num_code_layers)
    inner.embed_tokens = _FakeTalkerEmbedding(vocab_size, hidden_size, num_layers=num_code_layers)
    inner.codec_proj = nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size, bias=False),
        _FakeRMSNorm(hidden_size, eps=rms_norm_eps),
    )
    inner.embed_proj = nn.Sequential(
        nn.Linear(text_hidden_size, text_hidden_size, bias=False),
        nn.GELU(),
        nn.Linear(text_hidden_size, hidden_size, bias=False),
        _FakeRMSNorm(hidden_size, eps=rms_norm_eps),
    )
    inner.text_scale = nn.Parameter(torch.tensor(3.0))
    inner.audio_scale = nn.Parameter(torch.tensor(1.0))
    inner.spk_proj = nn.Linear(spk_emb_size, hidden_size, bias=False)
    inner.register_buffer("freqs_cos", cos, persistent=False)
    inner.register_buffer("freqs_sin", sin, persistent=False)
    # Some code paths duck-type ``inner.talker_config``; expose the bare config.
    inner.talker_config = SimpleNamespace(hidden_size=hidden_size, head_dim=head_dim)
    return inner


def make_fake_bundle(
    *,
    hidden_size: int = 8,
    vocab_size: int = 16,
    text_hidden_size: int | None = None,
    audio_vocab_size: int | None = None,
    audio_pad_token: int = 9,
    audio_stop_token: int = 10,
    audio_spk_token: int = 11,
    internal_stop_token_id: int = 12,
    num_code_layers: int = 8,
    num_hidden_layers: int = 2,
    max_position_embeddings: int = 32,
    rms_norm_eps: float = 1e-5,
    spk_emb_size: int = 4,
    max_steps_after_last_thinker_token: int = 4,
    use_moe: bool = False,
) -> Any:
    """Build a SimpleNamespace that mimics ``MinimindBundle`` for wrapper tests.

    ``audio_vocab_size`` defaults to ``vocab_size`` (the wrapper only reads
    it for the ``clamp(max=audio_vocab_size - 1)`` in
    ``_audio_ids_from_layer0``).
    """
    audio_vocab_size = audio_vocab_size if audio_vocab_size is not None else vocab_size
    text_hidden_size = text_hidden_size or hidden_size
    inner = make_fake_hf_talker(
        hidden_size=hidden_size,
        vocab_size=audio_vocab_size,
        text_hidden_size=text_hidden_size,
        num_code_layers=num_code_layers,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        spk_emb_size=spk_emb_size,
    )
    config = SimpleNamespace(
        hidden_size=text_hidden_size,
        talker_hidden_size=hidden_size,
        audio_vocab_size=audio_vocab_size,
        audio_pad_token=audio_pad_token,
        audio_stop_token=audio_stop_token,
        audio_spk_token=audio_spk_token,
        internal_stop_token_id=internal_stop_token_id,
        num_code_layers=num_code_layers,
        spk_emb_size=spk_emb_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        talker_max_steps_after_last_thinker_token=max_steps_after_last_thinker_token,
        use_moe=use_moe,
    )
    model = SimpleNamespace(talker=inner, config=config)
    bundle = SimpleNamespace(
        model=model,
        tokenizer=SimpleNamespace(eos_token_id=2),
        mimi=None,
        device="cpu",
        model_id="fake/minimind-3o",
        thinker=model,
        talker=None,  # wrap_talker will set it
        code2wav=None,
    )
    return bundle


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def span_bridge(hidden_size: int = 8, sequence_len: int = 4) -> torch.Tensor:
    """Build a deterministic [sequence_len, hidden_size] bridge hidden state."""
    torch.manual_seed(0)
    return torch.randn(sequence_len, hidden_size, dtype=torch.float32)


__all__ = [
    "_FakeMiniMindBlock",
    "_FakeTalkerHead",
    "_FakeTalkerEmbedding",
    "_FakeRMSNorm",
    "_fake_freqs_cis",
    "make_fake_hf_talker",
    "make_fake_bundle",
    "span_bridge",
]
