"""SmolVLM-500M-Instruct pipeline wiring (no real weights / transformers required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import StageExecutionType
from nanovllm_omni.models.smolvlm import stage as smolvlm_stage

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "smolvlm.yaml"


def test_pipeline_registry_resolves_smolvlm():
    config = resolve_pipeline_config("smolvlm")
    assert config is not None
    assert [s.name for s in config.stages] == ["vlm"]
    assert config.stages[0].kind == StageExecutionType.LLM_GENERATION
    assert config.stages[0].is_terminal is True
    assert config.stages[0].final_output_type == "text"
    assert config.default_deploy_config_name == "smolvlm.yaml"


def test_pipeline_registry_resolves_hf_handle():
    config = resolve_pipeline_config("HuggingFaceTB/SmolVLM-500M-Instruct")
    assert config is not None
    assert config.name == "smolvlm"


def test_hf_architectures_disambiguates_smolvlm():
    config = resolve_pipeline_config("smolvlm")
    assert config.hf_architectures == ("SmolVLMForConditionalGeneration",)


def test_deploy_yaml_schema_locked():
    import yaml

    cfg = yaml.safe_load(DEPLOY.read_text())
    assert cfg["max_batch"] >= 1
    stages = cfg["stages"]
    assert len(stages) == 1
    assert stages[0]["name"] == "vlm"
    # max_new_tokens is the one knob callers actually tune.
    assert "max_new_tokens" in stages[0]["default_sampling_params"]


def test_vlm_stage_import_guards(monkeypatch):
    """stage.py must only import transformers inside _vlm_stage."""
    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError(f"transformers must stay lazy ({name})")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    # Touching the module is fine (no top-level transformers).
    import nanovllm_omni.models.smolvlm.stage as mod

    # Building the factory requires transformers -> ImportError propagates.
    # Skip on no-torch hosts: `_vlm_stage` hits `import torch` first and
    # fails with "No module named 'torch'" (also an ImportError, but one that
    # does not match the "transformers" guard). The torch CI job covers this.
    pytest.importorskip("torch")
    args = OmniEngineArgs(model="HuggingFaceTB/SmolVLM-500M-Instruct")
    with pytest.raises(ImportError, match="transformers"):
        mod._vlm_stage(deploy=None, args=args)


def test_vlm_stage_dtype_set_locked():
    """SmolVLM is BF16-native; the dtype allowlist must reject int8 quant.

    Mirrors the sd_turbo dtype guard: surface the ceiling at factory time
    instead of silently casting a quantization path into a vision encoder.
    No monkeypatching here — we just check the allowlist constant.
    """
    assert "bfloat16" in smolvlm_stage._DTYPE_ALLOWED
    assert "float16" in smolvlm_stage._DTYPE_ALLOWED
    assert "int8" not in smolvlm_stage._DTYPE_ALLOWED
    assert "int4" not in smolvlm_stage._DTYPE_ALLOWED
