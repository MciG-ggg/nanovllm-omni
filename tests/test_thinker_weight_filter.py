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
            assert "embed_tokens.weight" in keys
            assert "layers.0.self_attn.q_proj.weight" in keys
            assert "norm.weight" in keys
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
