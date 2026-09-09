"""Tests for the new MiniMind model definitions (fork-layer based).

On macOS (no CUDA/triton), we mock the fork's layers to verify the model
structure and forward logic.  Real GPU tests run on WSL/Colab.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


def _install_fork_shim():
    """Install a minimal mock of nanovllm.* so model.py can import on CPU."""
    if "nanovllm" in sys.modules:
        return
    nanovllm = types.ModuleType("nanovllm")
    sys.modules["nanovllm"] = nanovllm

    # nanovllm.layers
    layers = types.ModuleType("nanovllm.layers")
    sys.modules["nanovllm.layers"] = layers
    nanovllm.layers = layers

    # nanovllm.layers.activation
    activation = types.ModuleType("nanovllm.layers.activation")

    class MockSiluAndMul(nn.Module if "nn" in dir() else object):
        def forward(self, x):
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

    activation.SiluAndMul = MockSiluAndMul
    sys.modules["nanovllm.layers.activation"] = activation
    layers.activation = activation

    # nanovllm.layers.attention
    attention = types.ModuleType("nanovllm.layers.attention")

    class MockAttention(torch.nn.Module):
        def __init__(self, num_heads, head_dim, scale, num_kv_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.k_cache = self.v_cache = torch.tensor([])

        def forward(self, q, k, v):
            # Simple scaled dot-product (no paged KV)
            from torch.nn.functional import scaled_dot_product_attention
            q = q.transpose(1, 2)  # [seq, heads, dim] -> [seq, 1, heads, dim] for SDPA
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if q.dim() == 3:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return o.transpose(1, 2).flatten(1, -1)

    attention.Attention = MockAttention
    sys.modules["nanovllm.layers.attention"] = attention
    layers.attention = attention

    # nanovllm.layers.linear
    linear = types.ModuleType("nanovllm.layers.linear")

    class MockQKVParallelLinear(torch.nn.Module):
        def __init__(self, hidden_size, head_dim, num_heads, num_kv_heads, bias=False):
            super().__init__()
            self.q_size = num_heads * head_dim
            self.kv_size = num_kv_heads * head_dim
            self.weight = torch.nn.Parameter(torch.randn(self.q_size + 2 * self.kv_size, hidden_size))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    class MockRowParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, bias=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    class MockMergedColumnParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_sizes, bias=False):
            super().__init__()
            total = sum(output_sizes)
            self.weight = torch.nn.Parameter(torch.randn(total, input_size))
            self.output_sizes = output_sizes

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    linear.QKVParallelLinear = MockQKVParallelLinear
    linear.RowParallelLinear = MockRowParallelLinear
    linear.MergedColumnParallelLinear = MockMergedColumnParallelLinear
    linear.LinearBase = torch.nn.Module
    sys.modules["nanovllm.layers.linear"] = linear
    layers.linear = linear

    # nanovllm.layers.layernorm
    layernorm = types.ModuleType("nanovllm.layers.layernorm")

    class MockRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, residual=None):
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
            if residual is not None:
                return normed, residual + x
            return normed

    layernorm.RMSNorm = MockRMSNorm
    sys.modules["nanovllm.layers.layernorm"] = layernorm
    layers.layernorm = layernorm

    # nanovllm.layers.rotary_embedding
    rotary = types.ModuleType("nanovllm.layers.rotary_embedding")

    class MockRotaryEmbedding(torch.nn.Module):
        def forward(self, positions, q, k):
            return q, k  # skip RoPE in mock

    def mock_get_rope(head_size, rotary_dim, max_position, base):
        return MockRotaryEmbedding()

    rotary.get_rope = mock_get_rope
    rotary.RotaryEmbedding = MockRotaryEmbedding
    sys.modules["nanovllm.layers.rotary_embedding"] = rotary
    layers.rotary_embedding = rotary

    # nanovllm.layers.embed_head
    embed_head = types.ModuleType("nanovllm.layers.embed_head")

    class MockVocabParallelEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        def forward(self, input_ids):
            return self.weight[input_ids]

    class MockParallelLMHead(MockVocabParallelEmbedding):
        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    embed_head.VocabParallelEmbedding = MockVocabParallelEmbedding
    embed_head.ParallelLMHead = MockParallelLMHead
    sys.modules["nanovllm.layers.embed_head"] = embed_head
    layers.embed_head = embed_head


# Install mock before importing model
_install_fork_shim()

import torch.nn as nn
from nanovllm_omni.models.minimind_omni.model import MiniMindTalker, MiniMindThinker


class TestMiniMindThinker:
    def _make_thinker(self, **kwargs):
        defaults = dict(
            vocab_size=1000, hidden_size=128, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=256,
            max_position=512, bridge_layer=1,
            audio_vocab_size=100, num_audio_heads=8,
        )
        defaults.update(kwargs)
        return MiniMindThinker(**defaults)

    def test_forward_output_shape(self):
        model = self._make_thinker()
        ids = torch.randint(0, 1000, (4,))
        pos = torch.arange(4)
        h = model(ids, pos)
        assert h.shape == (4, 128)

    def test_compute_logits_shape(self):
        model = self._make_thinker()
        ids = torch.randint(0, 1000, (4,))
        pos = torch.arange(4)
        h = model(ids, pos)
        logits = model.compute_logits(h)
        assert logits.shape == (4, 1000)

    def test_bridge_extracted(self):
        model = self._make_thinker(bridge_layer=0)
        ids = torch.randint(0, 1000, (4,))
        pos = torch.arange(4)
        model(ids, pos)
        bridge = model.get_bridge_hidden()
        assert bridge is not None
        assert bridge.shape == (4, 128)

    def test_bridge_none_for_negative_layer(self):
        model = self._make_thinker(bridge_layer=-1)
        ids = torch.randint(0, 1000, (4,))
        pos = torch.arange(4)
        model(ids, pos)
        assert model.get_bridge_hidden() is None

    def test_audio_logits_shape(self):
        model = self._make_thinker()
        ids = torch.randint(0, 1000, (4,))
        pos = torch.arange(4)
        h = model(ids, pos)
        audio = model.get_audio_logits(h)
        assert audio.shape == (4, 8, 100)

    def test_packed_modules_mapping(self):
        model = self._make_thinker()
        assert model.packed_modules_mapping["q_proj"] == ("qkv_proj", "q")
        assert model.packed_modules_mapping["gate_proj"] == ("gate_up_proj", 0)


class TestMiniMindTalker:
    def _make_talker(self, **kwargs):
        defaults = dict(
            hidden_size=128, num_layers=2, num_heads=4,
            num_kv_heads=2, intermediate_size=256, max_position=512,
            audio_vocab_size=100, num_audio_heads=8, talker_hidden_size=128,
        )
        defaults.update(kwargs)
        return MiniMindTalker(**defaults)

    def test_forward_output_shape(self):
        model = self._make_talker()
        hidden = torch.randn(4, 128)
        codes = torch.randint(0, 100, (4,))
        pos = torch.arange(4)
        audio = model(hidden, codes, pos)
        assert audio.shape == (4, 8, 100)

    def test_packed_modules_mapping(self):
        model = self._make_talker()
        assert "q_proj" in model.packed_modules_mapping
        assert "gate_proj" in model.packed_modules_mapping
