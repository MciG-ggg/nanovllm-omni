"""MiniMind-Omni talker config-field tests (TICKET-05 phase 1).

The wrapper class reads audio / talker config fields from the loaded
HF ``OmniConfig`` (``bundle.model.config``) and exposes them as plain
attributes. These tests pin the field-parsing contract so a config
rename or default-value drift is caught at the unit level without
needing the real HF checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.talker import (  # noqa: E402
    MiniMindOmniTalkerForConditionalGeneration,
    wrap_talker,
)
from tests._talker_fixtures import make_fake_bundle  # noqa: E402

# ---------------------------------------------------------------------------
# Construction + field parsing
# ---------------------------------------------------------------------------


def test_wrap_talker_returns_wrapper_instance() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    assert isinstance(talker, MiniMindOmniTalkerForConditionalGeneration)
    # wrap_talker wires bundle.talker so callers can read it back.
    assert bundle.talker is talker


def test_wrap_talker_is_idempotent() -> None:
    bundle = make_fake_bundle()
    first = wrap_talker(bundle)
    second = wrap_talker(bundle)
    assert first is second


def test_wrapper_exposes_audio_token_constants_from_config() -> None:
    bundle = make_fake_bundle(
        audio_pad_token=9,
        audio_stop_token=10,
        audio_spk_token=11,
        internal_stop_token_id=12,
        audio_vocab_size=16,
        num_code_layers=8,
    )
    talker = wrap_talker(bundle)
    assert talker.audio_pad_token == 9
    assert talker.audio_stop_token == 10
    assert talker.audio_spk_token == 11
    assert talker.internal_stop_token_id == 12
    assert talker.audio_vocab_size == 16
    assert talker.num_code_layers == 8


def test_wrapper_exposes_watchdog_and_hidden_sizes() -> None:
    bundle = make_fake_bundle(
        hidden_size=8,
        text_hidden_size=12,  # thinker's hidden size (mismatch is the common case)
        max_steps_after_last_thinker_token=7,
    )
    talker = wrap_talker(bundle)
    assert talker.hidden_size == 8
    assert talker.text_hidden_size == 12
    assert talker.max_steps_after_last_thinker_token == 7


def test_internal_stop_token_id_falls_back_to_audio_stop_when_missing() -> None:
    """When OmniConfig has no ``internal_stop_token_id``, the wrapper defaults to
    ``audio_stop_token`` so the Phase 2 watchdog has a valid sentinel.
    """
    bundle = make_fake_bundle(audio_stop_token=10)
    # Strip the field the wrapper would otherwise read.
    delattr(bundle.model.config, "internal_stop_token_id")
    talker = wrap_talker(bundle)
    assert talker.internal_stop_token_id == 10


def test_constructor_rejects_missing_bundle_and_hf_talker() -> None:
    with pytest.raises(ValueError, match="requires a MinimindBundle"):
        MiniMindOmniTalkerForConditionalGeneration()


def test_constructor_rejects_bundle_without_model_talker() -> None:
    bundle = make_fake_bundle()
    bundle.model.talker = None
    with pytest.raises(ValueError, match="talker is missing"):
        wrap_talker(bundle)


# ---------------------------------------------------------------------------
# Inner-module sharing (key invariant: fusion in attention.py still works)
# ---------------------------------------------------------------------------


def test_layers_norm_lm_head_are_shared_with_inner_module() -> None:
    """Fusion / patching in ``attention.enable_fused_projections`` mutates the
    inner module's layers in place; the wrapper must share the same
    ``nn.ModuleList`` reference so those mutations propagate.
    """
    bundle = make_fake_bundle(num_hidden_layers=2)
    inner = bundle.model.talker
    talker = wrap_talker(bundle)
    assert talker.layers is inner.layers
    assert talker.norm is inner.norm
    assert talker.lm_head is inner.lm_head
    assert talker.embed_tokens is inner.embed_tokens
    assert talker.codec_proj is inner.codec_proj
    assert talker.embed_proj is inner.embed_proj
    assert talker.spk_proj is inner.spk_proj
    # RoPE buffers share storage with the inner module so a buffer
    # recompute on the inner module propagates.
    assert talker.freqs_cos is inner.freqs_cos
    assert talker.freqs_sin is inner.freqs_sin


def test_code_layer_masks_registered_as_buffer() -> None:
    """``_code_layer_masks`` is the [num_code_layers+1, num_code_layers]
    gate used by ``preprocess``. Pin its shape and persistent flag.
    """
    bundle = make_fake_bundle(num_code_layers=8)
    talker = wrap_talker(bundle)
    masks = talker._code_layer_masks
    assert masks.shape == (9, 8)
    # Row 0 (step=-1) is all-False; row 1 (step=0) layer 0 only; row 8 all-True.
    assert masks[0].sum().item() == 0
    assert masks[1].sum().item() == 1
    assert masks[8].sum().item() == 8


# ---------------------------------------------------------------------------
# Defaults when fields are absent (fake-free fallback)
# ---------------------------------------------------------------------------


def test_default_config_when_bundle_is_minimal() -> None:
    """``MinimindBundle`` with no ``model.config`` should fall back to
    MiniMind-O's published defaults rather than raising.
    """
    bundle = make_fake_bundle()
    bundle.model.config = None
    talker = wrap_talker(bundle)
    assert talker.audio_pad_token == 2049
    assert talker.audio_stop_token == 2050
    assert talker.audio_spk_token == 2051
    assert talker.audio_vocab_size == 2112
    assert talker.num_code_layers == 8
    assert talker.max_steps_after_last_thinker_token == 192
    assert talker.hidden_size == 768
    assert talker.text_hidden_size == 768
    assert talker.spk_emb_size == 192
