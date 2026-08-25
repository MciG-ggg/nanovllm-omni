"""Sana-0.6B diffusion pipeline wiring (no real weights / diffusers required)."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import StageExecutionType
from nanovllm_omni.models.sana_06b import stage as sana_06b_stage
from nanovllm_omni.outputs import OmniRequestOutput

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "sana_06b.yaml"


def test_pipeline_registry_resolves_sana():
    config = resolve_pipeline_config("sana_06b")
    assert config is not None
    assert [s.name for s in config.stages] == ["sana"]
    assert config.stages[0].kind == StageExecutionType.DIFFUSION
    assert config.stages[0].is_terminal is True
    assert config.stages[0].final_output_type == "image"
    assert config.default_deploy_config_name == "sana_06b.yaml"


def test_pipeline_registry_resolves_hf_handle():
    config = resolve_pipeline_config("Efficient-Large-Model/Sana_600M_1024px_diffusers")
    assert config is not None
    assert config.name == "sana_06b"


def test_from_pipeline_image_key():
    """Omni._one wraps the stage payload via from_pipeline(final_output_type)
    -- for image the payload is a PIL Image (no .audio attr), so it must land
    under ``multimodal_output["image"]`` and trip ``is_diffusion_output``."""
    img = object()  # stands in for a PIL.Image
    output = OmniRequestOutput.from_pipeline(img, final_output_type="image")
    assert output.multimodal_output is not None
    assert output.multimodal_output["image"] is img
    assert output.is_diffusion_output is True


def test_sana_stage_requires_diffusers(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers.") or name == "torch":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="No module named"):
        sana_06b_stage._sana_stage(None, OmniEngineArgs(model="sana_06b"))


def test_deploy_defaults_steps(tmp_path: Path):
    from nanovllm_omni.config.registry import load_deploy_config, merge_pipeline_deploy

    deploy = load_deploy_config(DEPLOY)
    merged = merge_pipeline_deploy(resolve_pipeline_config("sana_06b"), deploy)
    _, defaults = merged[0]
    assert defaults["num_inference_steps"] == 20


def test_try_infer_model_type_basename():
    from nanovllm_omni.entrypoints.base import try_infer_model_type

    assert try_infer_model_type("sana_06b") == "sana_06b"
    assert try_infer_model_type("/checkpoints/sana_06b_fp16") == "sana_06b"
