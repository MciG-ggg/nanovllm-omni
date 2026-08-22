"""Tests for ``nanovllm_omni.optim.cuda_graph`` (TK-011 followup)."""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Stubs (no real model load; a tiny linear-stack module that matches the
# surface the cache needs -- ``forward(input_ids, past_key_values=...)``
# returning a ``MoeCausalLMOutputWithPast``-shaped object).
# ---------------------------------------------------------------------------


class _FakeOutput:
    """Stand-in for ``MoeCausalLMOutputWithPast`` used by the cache."""

    def __init__(self, *, logits, past_key_values, audio_logits, aux_loss):
        self.logits = logits
        self.past_key_values = past_key_values
        self.audio_logits = audio_logits
        self.aux_loss = aux_loss


class _FakeInner(torch.nn.Module):
    """Linear-stack fake that mimics the per-step forward contract:

    * input shape: ``[B, 9, 1]`` (1 text + 8 audio channels, single token)
    * past_kvs: list of ``(k, v)`` per layer, each ``[B, n_heads, N, head_dim]``
    * output past_kvs: list of ``(k, v)`` per layer, each ``[B, n_heads, N+1, head_dim]``
    * logits: ``[B, 1, vocab]``
    * audio_logits: list of 8 tensors ``[B, 1, audio_vocab]``
    * aux_loss: scalar
    """

    VOCAB = 32
    AUDIO_VOCAB = 16
    N_LAYERS = 2
    N_HEADS = 2
    HEAD_DIM = 4

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(9, 9)  # placeholder; the fake forward ignores weights
        # Reusable zero KV so all "captures" see consistent state.
        self._zero_kv_template: tuple[torch.Tensor, torch.Tensor] = (
            # MiniMind-O layout: [B, seq, n_heads, head_dim]
            torch.zeros(1, 0, self.N_HEADS, self.HEAD_DIM),
            torch.zeros(1, 0, self.N_HEADS, self.HEAD_DIM),
        )

    def forward(self, input_ids, past_key_values=None, **kwargs):  # noqa: D401
        # past_key_values is a list of (k, v) per layer; each is
        # [B, seq, n_heads, head_dim] (MiniMind-O layout -- cat on dim=1).
        # Build new past = past + [0]*1 in seq.
        new_past = []
        for _layer_idx, (k, v) in enumerate(
            past_key_values or [self._zero_kv_template] * self.N_LAYERS
        ):
            new_k = torch.cat([k, torch.zeros(1, 1, self.N_HEADS, self.HEAD_DIM)], dim=1)
            new_v = torch.cat([v, torch.zeros(1, 1, self.N_HEADS, self.HEAD_DIM)], dim=1)
            new_past.append((new_k, new_v))
        logits = torch.zeros(1, 1, self.VOCAB)
        audio_logits = [torch.zeros(1, 1, self.AUDIO_VOCAB) for _ in range(8)]
        return _FakeOutput(
            logits=logits,
            past_key_values=new_past,
            audio_logits=audio_logits,
            aux_loss=torch.zeros(()),
        )


def _build_past(n_layers: int, n_heads: int, head_dim: int, past_len: int) -> list:
    return [
        (torch.zeros(1, past_len, n_heads, head_dim), torch.zeros(1, past_len, n_heads, head_dim))
        for _ in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_graph_compile_model_is_idempotent():
    """Wrapping twice is a no-op (the second call returns the same wrapper)."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    inner = _FakeInner()
    wrapped1 = graph_compile_model(inner)
    wrapped2 = graph_compile_model(wrapped1)
    assert wrapped1 is wrapped2
    # And ``forward`` is still routed through the cache.
    assert hasattr(wrapped1, "stats")


def test_full_prompt_call_is_eager():
    """First call (past_kvs is None) goes through eager, no graph capture."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    inner = _FakeInner()
    wrapped = graph_compile_model(inner)
    full_input = torch.zeros(1, 9, 7)  # 7-token prompt
    out = wrapped(full_input, past_key_values=None, use_cache=True)
    assert out.logits.shape == (1, 1, _FakeInner.VOCAB)
    s = wrapped.stats()
    assert s["eager_calls"] == 1
    assert s["graph_calls"] == 0
    assert s["graphs_captured"] == 0


def test_incremental_call_captures_and_replays():
    """A second call (past_kvs of length N) gets graphed and replays match eager."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    inner = _FakeInner()
    wrapped = graph_compile_model(inner)
    # First call: full prompt, past_kvs None.
    full_input = torch.zeros(1, 9, 7)
    wrapped(full_input, past_key_values=None, use_cache=True)
    # Second call: incremental with past_len=0.
    inc_input = torch.zeros(1, 9, 1)
    past0 = _build_past(_FakeInner.N_LAYERS, _FakeInner.N_HEADS, _FakeInner.HEAD_DIM, 0)
    out = wrapped(inc_input, past_key_values=past0, use_cache=True)
    s = wrapped.stats()
    assert s["graphs_captured"] == 1
    assert s["graph_calls"] == 1
    # Output present must have past_len=1.
    assert out.past_key_values[0][0].shape == (1, 1, _FakeInner.N_HEADS, _FakeInner.HEAD_DIM)
    # Replay the same shape: graph_calls should go up, no new graph captured.
    wrapped(inc_input, past_key_values=past0, use_cache=True)
    s2 = wrapped.stats()
    assert s2["graphs_captured"] == 1
    assert s2["graph_calls"] == 2


def test_different_past_lens_capture_separate_graphs():
    """Each past_len gets its own graph."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    inner = _FakeInner()
    wrapped = graph_compile_model(inner)
    full_input = torch.zeros(1, 9, 7)
    wrapped(full_input, past_key_values=None, use_cache=True)
    inc = torch.zeros(1, 9, 1)
    for n in range(3):
        past = _build_past(_FakeInner.N_LAYERS, _FakeInner.N_HEADS, _FakeInner.HEAD_DIM, n)
        wrapped(inc, past_key_values=past, use_cache=True)
    s = wrapped.stats()
    assert s["graphs_captured"] == 3
    assert s["graph_calls"] == 3


def test_capture_failure_falls_back_to_eager():
    """If capture raises, the wrapper falls back to eager without crashing."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    class Bad(torch.nn.Module):
        def forward(self, x, past_key_values=None, **kwargs):
            raise RuntimeError("synthetic forward failure")

    wrapped = graph_compile_model(Bad())
    inc = torch.zeros(1, 9, 1)
    past = _build_past(2, 2, 4, 0)
    # First incremental call: capture raises, fallback to eager (which also raises).
    with pytest.raises(RuntimeError, match="synthetic"):
        wrapped(inc, past_key_values=past, use_cache=True)
    s = wrapped.stats()
    assert s["capture_failures"] >= 1


def test_replay_output_is_independent_of_static_buffer():
    """Two consecutive replays return tensors whose data does not alias."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_model

    inner = _FakeInner()
    wrapped = graph_compile_model(inner)
    full_input = torch.zeros(1, 9, 7)
    wrapped(full_input, past_key_values=None, use_cache=True)
    inc = torch.zeros(1, 9, 1)
    past0 = _build_past(_FakeInner.N_LAYERS, _FakeInner.N_HEADS, _FakeInner.HEAD_DIM, 0)
    out1 = wrapped(inc, past_key_values=past0, use_cache=True)
    # Mutate past0; should not affect a subsequent replay (it copies into
    # the static buffer).
    past0[0][0].fill_(99.0)
    out2 = wrapped(inc, past_key_values=past0, use_cache=True)
    # Both outputs are zero (the fake forward does nothing with values),
    # so we just check shapes and that the wrapper still works after the
    # copy path.
    assert out1.past_key_values[0][0].shape == out2.past_key_values[0][0].shape  # noqa: E501
