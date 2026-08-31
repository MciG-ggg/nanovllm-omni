"""SD-Turbo diffusion pipeline wiring (no real weights / diffusers required)."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import nanovllm_omni
from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import StageExecutionType
from nanovllm_omni.models.sd_turbo import stage as sd_turbo_stage
from nanovllm_omni.outputs import OmniRequestOutput

REPO = Path(__file__).resolve().parents[1]
DEPLOY = Path(nanovllm_omni.__file__).resolve().parent / "deploy" / "sd_turbo.yaml"


def test_pipeline_registry_resolves_sd_turbo():
    config = resolve_pipeline_config("sd_turbo")
    assert config is not None
    assert [s.name for s in config.stages] == ["sd_turbo"]
    assert config.stages[0].kind == StageExecutionType.DIFFUSION
    assert config.stages[0].is_terminal is True
    assert config.stages[0].final_output_type == "image"
    assert config.default_deploy_config_name == "sd_turbo.yaml"


def test_pipeline_registry_resolves_hf_handle():
    config = resolve_pipeline_config("stabilityai/sd-turbo")
    assert config is not None
    assert config.name == "sd_turbo"


def test_from_pipeline_image_key():
    """Omni._one wraps the stage payload via from_pipeline(final_output_type)
    -- for image the payload is a PIL Image (no .audio attr), so it must land
    under ``multimodal_output["image"]`` and trip ``is_diffusion_output``."""
    img = object()  # stands in for a PIL.Image
    output = OmniRequestOutput.from_pipeline(img, final_output_type="image")
    assert output.multimodal_output is not None
    assert output.multimodal_output["image"] is img
    assert output.is_diffusion_output is True


def test_sd_turbo_stage_requires_diffusers(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers.") or name == "torch":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="No module named"):
        sd_turbo_stage._sd_turbo_stage(None, OmniEngineArgs(model="sd_turbo"))


def test_deploy_defaults_steps(tmp_path: Path):
    from nanovllm_omni.config.registry import load_deploy_config, merge_pipeline_deploy

    deploy = load_deploy_config(DEPLOY)
    merged = merge_pipeline_deploy(resolve_pipeline_config("sd_turbo"), deploy)
    _, defaults = merged[0]
    assert defaults["num_inference_steps"] == 1
    assert defaults["guidance_scale"] == 0.0
    assert defaults["height"] == 512
    assert defaults["width"] == 512


def test_try_infer_model_type_basename():
    from nanovllm_omni.entrypoints.base import try_infer_model_type

    assert try_infer_model_type("sd_turbo") == "sd_turbo"
    assert try_infer_model_type("/checkpoints/sd_turbo_local") == "sd_turbo"
