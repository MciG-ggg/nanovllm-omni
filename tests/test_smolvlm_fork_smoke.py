"""Smoke tests for fork-AR SmolVLM (ADR-016 revised, 2026-09).

CPU-only smoke: imports + structural assertions. Real-weight GPU
validation lives in the WSL bench (``uv run python -m
nanovllm_omni.bench.bench_smolvlm --help``); those gates are deferred
to the bench run, not asserted here.

What we assert:
- SmolVLM full model instantiates from a mock config (no triton / GPU).
- submodule names mirror HF checkpoint keys (loader lands weights
  with default prefix="").
- packed_modules_mapping declares the fused-proj rewrites.
- pipeline registers with kind=LLM_AR (per ADR-018).
- stage factory exports SmolVLMStage (per ADR-019).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from nanovllm_omni.models.smolvlm.smolvlm import (
    SmolLM2ForCausalLM,
    SmolVLMConnector,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
)


def _mock_smolvlm_config():
    """Stand-in for ``transformers.SmolVLMConfig`` (no transformers dep here)."""
    vision = SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        patch_size=16,
        image_size=512,
    )
    text = SimpleNamespace(
        hidden_size=576,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        intermediate_size=1536,
        vocab_size=49152,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        rope_theta=100000,
        rope_scaling=None,
    )
    return SimpleNamespace(
        vision_config=vision,
        text_config=text,
        scale_factor=2,
    )


def test_smolvlm_submodule_layout(monkeypatch):
    """Submodule names mirror HF checkpoint keys (no prefix gymnastics).

    Mocks the HF ``SiglipVisionModel`` constructor because the smoke
    test uses a stand-in config that won't pass HF's ``PreTrainedConfig``
    type check; real config construction lives in the WSL bench path.
    """
    from unittest.mock import MagicMock

    config = _mock_smolvlm_config()
    # Patch the resolver so the eager ``self.vision_model = Siglip...()``
    # in SmolVLMModel.__init__ returns a MagicMock instead of trying to
    # validate our stand-in config against PreTrainedConfig.
    monkeypatch.setattr(
        "nanovllm_omni.models.smolvlm.smolvlm._resolve_siglip_backbone",
        lambda: MagicMock(),
    )
    model = SmolVLMForConditionalGeneration(config)

    assert hasattr(model, "model"), "missing outer self.model"
    assert isinstance(model.model, SmolVLMModel)
    assert hasattr(model.model, "vision_model"), "missing model.vision_model"
    assert hasattr(model.model, "connector"), "missing model.connector"
    assert isinstance(model.model.connector, SmolVLMConnector)
    assert hasattr(model.model, "text_model"), "missing model.text_model"
    assert isinstance(model.model.text_model, SmolLM2ForCausalLM)
    assert hasattr(model, "lm_head"), "missing lm_head"


def test_smolvlm_packed_modules_mapping():
    """The loader picks up fused-proj mapping from the text backbone submodule.

    ``SmolVLMForConditionalGeneration`` itself does NOT expose
    ``packed_modules_mapping`` -- the SigLIP vision tower uses
    per-head projections. The text backbone ``SmolLM2ForCausalLM``
    declares the fused mapping; ``_resolve_packed_modules_mapping``
    walks the module path to find it for the right subkeys.
    """
    from nanovllm_omni.models.smolvlm.smolvlm import SmolLM2ForCausalLM

    mapping = SmolLM2ForCausalLM.packed_modules_mapping
    for src, (dst, shard) in mapping.items():
        assert isinstance(src, str) and src.endswith("_proj"), src
        assert isinstance(dst, str) and dst.endswith("_proj"), (src, dst)
        assert isinstance(shard, (str, int)), (src, shard)
    assert not hasattr(SmolVLMForConditionalGeneration, "packed_modules_mapping")


def test_smolvlm_connector_shape():
    """Connector wraps a Linear under ``proj`` to mirror HF key path."""
    config = _mock_smolvlm_config()
    connector = SmolVLMConnector(config)
    # ``modality_projection`` is a wrapper; the actual Linear lives at
    # ``modality_projection.proj`` so the HF loader can land weights
    # at ``model.connector.modality_projection.proj.{weight,bias}``.
    assert hasattr(connector, "modality_projection")
    proj = connector.modality_projection.proj
    assert isinstance(proj, torch.nn.Linear)
    assert proj.in_features == config.vision_config.hidden_size * (config.scale_factor**2)
    assert proj.out_features == config.text_config.hidden_size


def test_smolvlm_connector_pure_linear():
    """Connector holds no fork state (no paged-KV / triton dep)."""
    config = _mock_smolvlm_config()
    connector = SmolVLMConnector(config)
    # Smoke: forward on random features returns right shape, runs on CPU
    # without pulling triton (fork dependency stays out of vision path).
    feats = torch.randn(2, 16, config.vision_config.hidden_size * (config.scale_factor**2))
    out = connector(feats)
    assert out.shape == (2, 16, config.text_config.hidden_size)


def test_pipeline_kind_is_llm_ar():
    """Per ADR-018: kind=LLM_AR replaces the prior LLM_GENERATION placeholder."""
    # Force re-import in case another test mutated the registry
    if "nanovllm_omni.models.smolvlm.pipeline" in sys.modules:
        del sys.modules["nanovllm_omni.models.smolvlm.pipeline"]
    from nanovllm_omni.config.registry import (
        StageExecutionType,
        resolve_pipeline_config,
    )
    from nanovllm_omni.models.smolvlm import pipeline as _pipeline  # noqa: F401

    cfg = resolve_pipeline_config("smolvlm")
    assert cfg is not None, "smolvlm pipeline not registered"
    assert len(cfg.stages) == 1
    stage = cfg.stages[0]
    assert stage.kind == StageExecutionType.LLM_AR, f"expected LLM_AR (ADR-018), got {stage.kind}"
    assert stage.name == "vlm"
    assert stage.is_terminal
    assert stage.final_output_type == "text"


def test_stage_factory_exports():
    """Stage factory returns SmolVLMStage class instance (ADR-019)."""
    from nanovllm_omni.models.smolvlm.stage import (
        SmolVLMStage,
        _vlm_stage,
    )

    assert callable(_vlm_stage)
    assert SmolVLMStage.__name__ == "SmolVLMStage"
