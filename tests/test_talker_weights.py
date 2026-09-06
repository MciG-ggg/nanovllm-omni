"""MiniMind-Omni talker weight-loading tests (TICKET-05 phase 1).

The wrapper's ``load_weights`` walks an iterable of ``(name, tensor)``
pairs, strips the ``model.talker.`` prefix, and copies tensors into the
shared inner-module parameters. These tests use a tiny synthetic
``TalkerModule`` (see ``tests/_talker_fixtures.py``) so the round-trip
can be exercised offline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402
from tests._talker_fixtures import make_fake_bundle  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_dict_for_inner(bundle) -> dict:
    """Build a synthetic HF-style checkpoint covering every wrapper parameter.

    Mirrors the prefix layout the real HF ``MiniMindOmni`` checkpoint
    uses (``model.talker.<param_name>``) so the prefix-stripping path is
    exercised end-to-end. Each param gets a non-zero tensor of the
    right shape so a subsequent ``copy_`` is detectable.
    """
    inner = bundle.model.talker
    state: dict = {}
    for name, param in inner.named_parameters():
        full_name = f"model.talker.{name}"
        state[full_name] = torch.full_like(param, 0.5) + torch.randn_like(param) * 0.01
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_weights_strips_model_talker_prefix() -> None:
    bundle = make_fake_bundle(num_hidden_layers=2)
    talker = wrap_talker(bundle)
    state = _make_state_dict_for_inner(bundle)
    loaded = talker.load_weights(state.items())
    # Every wrapper-local parameter should have been loaded.
    expected = {name for name, _ in talker.named_parameters()}
    assert expected.issubset(loaded), f"missing loaded params: {sorted(expected - loaded)[:5]}"


def test_load_weights_skips_thinker_audio_vision_prefixes() -> None:
    """Keys starting with ``thinker.``, ``audio_proj.``, ``vision_proj.``, or
    ``model.<other>.`` belong to other sub-modules and must be ignored.
    """
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    state: dict = {
        "model.talker.norm.weight": torch.ones(8),
        "model.thinker.norm.weight": torch.zeros(8),  # must be skipped
        "model.audio_proj.mlp.0.weight": torch.zeros(8),  # must be skipped
        "model.vision_proj.mlp.0.weight": torch.zeros(8),  # must be skipped
        "model.norm.weight": torch.zeros(8),  # bare "model." -> belongs to inner MiniMindModel
        "rotary_emb.inv_freq": torch.zeros(4),  # must be skipped
    }
    loaded = talker.load_weights(state.items())
    # Only ``norm.weight`` (after the model.talker. strip) lands.
    assert "norm.weight" in loaded
    assert len(loaded) == 1


def test_load_weights_returns_loaded_names_only() -> None:
    bundle = make_fake_bundle(num_hidden_layers=2)
    talker = wrap_talker(bundle)
    # Submit only the norm + first layer's projection.
    state = {
        "model.talker.norm.weight": torch.ones(8),
        "model.talker.layers.0.proj.weight": torch.ones(8, 8),
    }
    loaded = talker.load_weights(state.items())
    assert loaded == {"norm.weight", "layers.0.proj.weight"}


def test_load_weights_actually_copies_tensor_values() -> None:
    """Pin that the loader mutates the param data, not just registers a key."""
    bundle = make_fake_bundle(num_hidden_layers=2)
    talker = wrap_talker(bundle)
    inner = bundle.model.talker
    # Seed with all-zeros then load a known tensor.
    with torch.no_grad():
        inner.norm.weight.zero_()
    state = {"model.talker.norm.weight": torch.full((8,), 0.42)}
    talker.load_weights(state.items())
    assert torch.allclose(inner.norm.weight, torch.full((8,), 0.42))


def test_load_weights_accepts_state_dict() -> None:
    """``load_weights`` accepts any iterable of ``(name, tensor)`` pairs,
    including the output of ``inner.state_dict()`` plus a prefix.
    """
    bundle = make_fake_bundle(num_hidden_layers=2)
    inner = bundle.model.talker
    talker = wrap_talker(bundle)
    prefixed = {
        f"model.talker.{name}": tensor.detach().clone()
        for name, tensor in inner.state_dict().items()
    }
    loaded = talker.load_weights(prefixed.items())
    expected = {name for name, _ in inner.named_parameters()}
    assert expected.issubset(loaded)


def test_load_weights_unknown_key_is_silently_ignored() -> None:
    """Unknown keys (e.g. legacy parameter renames) are skipped without raising."""
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    state = {
        "model.talker.norm.weight": torch.ones(8),
        "model.talker.layers.99.ghost.weight": torch.ones(8),  # doesn't exist
        "model.talker.unrelated_attr": torch.ones(8),  # doesn't exist
    }
    loaded = talker.load_weights(state.items())
    assert loaded == {"norm.weight"}
