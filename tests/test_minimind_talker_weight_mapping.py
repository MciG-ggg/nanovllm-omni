"""Talker packed-module mappings must live on the fused submodules."""

import pytest


def test_model_level_mapping_is_absent() -> None:
    pytest.importorskip("torch")
    from nanovllm_omni.models.minimind_omni.talker import MiniMindTalker

    assert not hasattr(MiniMindTalker, "packed_modules_mapping")


def test_attention_owns_the_qkv_mapping() -> None:
    pytest.importorskip("torch")
    from nanovllm_omni.models.minimind_omni.talker import TalkerAttention

    assert set(TalkerAttention.packed_modules_mapping) == {"q_proj", "k_proj", "v_proj"}


def test_mlp_owns_the_gate_up_mapping() -> None:
    pytest.importorskip("torch")
    from nanovllm_omni.models.minimind_omni.talker import TalkerMLP

    assert set(TalkerMLP.packed_modules_mapping) == {"gate_proj", "up_proj"}
