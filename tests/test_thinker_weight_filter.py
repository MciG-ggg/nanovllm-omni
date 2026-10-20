"""Thinker-only filter must keep HF's top-level ``lm_head.weight``."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def _fake_checkpoint(src: Path) -> None:
    state = {
        "model.embed_tokens.weight": torch.zeros(4, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "model.norm.weight": torch.zeros(2),
        "lm_head.weight": torch.ones(4, 2),
        "talker.lm_head.base.weight": torch.full((3, 2), 2.0),
        "talker.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
    }
    torch.save(state, src / "pytorch_model.bin")
    (src / "config.json").write_text("{}")


def _safetensor_keys(dst: str) -> set[str]:
    from safetensors import safe_open

    with safe_open(f"{dst}/model.safetensors", framework="pt") as f:
        return set(f.keys())


def test_thinker_only_dir_keeps_toplevel_lm_head() -> None:
    from nanovllm_omni.models.minimind_omni.stage import _thinker_only_dir

    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        _fake_checkpoint(src)
        dst = _thinker_only_dir(str(src))
        try:
            keys = _safetensor_keys(dst)
            assert "lm_head.weight" in keys
            assert "model.embed_tokens.weight" in keys
            assert "model.layers.0.self_attn.q_proj.weight" in keys
            assert "model.norm.weight" in keys
            assert "lm_head.base.weight" not in keys
            from safetensors import safe_open

            with safe_open(f"{dst}/model.safetensors", framework="pt") as f:
                assert torch.equal(f.get_tensor("lm_head.weight"), torch.ones(4, 2))
        finally:
            shutil.rmtree(dst, ignore_errors=True)


def test_talker_only_dir_drops_bare_lm_head() -> None:
    from nanovllm_omni.models.minimind_omni.stage import _talker_only_dir

    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        _fake_checkpoint(src)
        dst = _talker_only_dir(str(src))
        try:
            keys = _safetensor_keys(dst)
            assert "lm_head.weight" not in keys
            assert "lm_head.base.weight" in keys
            assert "layers.0.self_attn.q_proj.weight" in keys
        finally:
            shutil.rmtree(dst, ignore_errors=True)


def test_thinker_only_dir_keys_match_thinker_parameters() -> None:
    """Every kept key must resolve to a parameter on MiniMindThinker.

    The fork loader walks each checkpoint key through
    ``model.get_parameter(key)`` and silently skips the ones that miss.
    A prefix mismatch between the filtered file and the model therefore
    does not raise -- it runs the stage on init weights.
    """
    pytest.importorskip("triton")
    from nanovllm.utils.loader import _resolve_packed_modules_mapping

    from nanovllm_omni.models.minimind_omni.stage import _thinker_only_dir
    from nanovllm_omni.models.minimind_omni.thinker import MiniMindThinker

    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        _fake_checkpoint(src)
        dst = _thinker_only_dir(str(src))
        try:
            keys = _safetensor_keys(dst)

            class _Cfg:
                vocab_size = 4
                hidden_size = 2
                num_hidden_layers = 1
                num_attention_heads = 2
                num_key_value_heads = 2
                intermediate_size = 4
                max_position_embeddings = 16
                rms_norm_eps = 1e-6
                rope_theta = 10000
                head_dim = 1
                bridge_layer = 0
                num_audio_heads = 8
                audio_vocab_size = 3
                dtype = torch.float32

            model = MiniMindThinker(_Cfg())

            unresolved = []
            for key in keys:
                mapping = _resolve_packed_modules_mapping(model, key)
                if mapping:
                    # q/k/v merge into qkv_proj; the loader resolves the
                    # merged name and loads through its weight_loader.
                    merged = next(
                        (key.replace(k, v) for k, (v, _) in mapping.items() if k in key),
                        None,
                    )
                    if merged is None:
                        continue
                    try:
                        model.get_parameter(merged)
                    except AttributeError:
                        unresolved.append(key)
                    continue
                try:
                    model.get_parameter(key)
                except AttributeError:
                    unresolved.append(key)
            assert unresolved == []
        finally:
            shutil.rmtree(dst, ignore_errors=True)
