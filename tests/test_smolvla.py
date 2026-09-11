"""SmolVLA pipeline wiring (no real weights / lerobot required)."""

from __future__ import annotations

from pathlib import Path

import nanovllm_omni
from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.registry import StageExecutionType
from nanovllm_omni.outputs import ActionArtifact, OmniRequestOutput

REPO = Path(__file__).resolve().parents[1]
DEPLOY = Path(nanovllm_omni.__file__).resolve().parent / "deploy" / "smolvla.yaml"


class _Arr:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def test_pipeline_registry_resolves_smolvla():
    config = resolve_pipeline_config("smolvla")
    assert config is not None
    assert [s.name for s in config.stages] == ["vlm", "action"]
    assert config.stages[0].kind == StageExecutionType.LLM_AR
    assert config.stages[0].is_terminal is False
    assert config.stages[1].kind == StageExecutionType.DIFFUSION
    assert config.stages[1].is_terminal is True
    assert config.stages[1].final_output_type == "actions"
    assert config.default_deploy_config_name == "smolvla.yaml"


def test_pipeline_registry_resolves_libero_handle():
    config = resolve_pipeline_config("HuggingFaceVLA/smolvla_libero")
    assert config is not None
    assert config.name == "smolvla"


def test_from_pipeline_actions_key():
    artifact = ActionArtifact.from_array(_Arr((50, 7), "float32"))
    output = OmniRequestOutput.from_pipeline(artifact, final_output_type="actions")
    assert output.multimodal_output is not None
    assert output.multimodal_output["actions"] is artifact


def test_local_dir_config_type_selects_smolvla(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"type": "smolvla"}', encoding="utf-8")
    from nanovllm_omni.entrypoints.base import OmniBase

    base = OmniBase(str(tmp_path), deploy_config_path=str(DEPLOY))
    assert base._resolve_pipeline().name == "smolvla"


def test_local_dir_basename_falls_back_to_registered_key(tmp_path: Path):
    """Layer-5 fallback: empty / malformed config.json + matching dir name."""
    sub = tmp_path / "smolvla_libero"
    sub.mkdir()
    # config.json absent -- layer 1, 2, 3 all skip; layer 5 catches the basename.
    from nanovllm_omni.entrypoints.base import OmniBase

    base = OmniBase(str(sub), deploy_config_path=str(DEPLOY))
    assert base._resolve_pipeline().name == "smolvla"


def test_try_infer_model_type_basename_substring():
    """Longest-match wins (mirrors vllm-omni _name_match_candidate)."""
    from nanovllm_omni.entrypoints.base import try_infer_model_type

    # Direct basename matches.
    assert try_infer_model_type("smolvla") == "smolvla"
    # Substring inside a longer path.
    assert try_infer_model_type("/checkpoints/smolvla_libero") == "smolvla"
    # No match.
    assert try_infer_model_type("totally_unknown_xyz") is None


def test_try_infer_model_type_config_json_architecture_singular(tmp_path: Path):
    """Layer-3 fallback: ``architecture`` (singular) is a VoxCPM2-style field."""
    sub = tmp_path / "anymodel"
    sub.mkdir()
    (sub / "config.json").write_text('{"architecture": "SmolVLAPolicy"}', encoding="utf-8")
    from nanovllm_omni.entrypoints.base import try_infer_model_type

    assert try_infer_model_type(str(sub)) == "SmolVLAPolicy"


def test_compute_final_stage_id_prefers_terminal_with_matching_type():
    """Layer-6: ``hf_architectures`` routes correctly; final_stage_id picks the
    stage whose ``final_output_type`` is in the requested modalities (or
    defaults to the last terminal stage)."""
    from nanovllm_omni.config.registry import (
        PipelineConfig,
        StageConfig,
        StageExecutionType,
    )
    from nanovllm_omni.entrypoints.base import OmniBase

    pipeline = PipelineConfig(
        name="multi_modal",
        stages=(
            StageConfig(
                0, "thinker", StageExecutionType.LLM_AR, "tests._stage_factories:thinker_simple"
            ),
            StageConfig(
                1,
                "actions",
                StageExecutionType.LLM_GENERATION,
                "tests._stage_factories:thinker_simple",
                is_terminal=True,
                final_output_type="actions",
            ),
            StageConfig(
                2,
                "audio",
                StageExecutionType.CODEC,
                "tests._stage_factories:thinker_simple",
                is_terminal=True,
                final_output_type="audio",
            ),
        ),
        default_deploy_config_name="multi_modal.yaml",
    )
    base = OmniBase.__new__(OmniBase)
    base._pipeline = pipeline
    base.engine_args = None  # noqa: SLF001
    base._deploy = None  # noqa: SLF001
    base._executor = None  # noqa: SLF001
    base._bundle = None  # noqa: SLF001
    # No modality filter: last terminal wins.
    assert base._compute_final_stage_id() == 2
    assert base._compute_final_stage_id(["actions"]) == 1
    assert base._compute_final_stage_id(["audio"]) == 2
    # Unknown modality falls back to last stage.
    assert base._compute_final_stage_id(["video"]) == 2
