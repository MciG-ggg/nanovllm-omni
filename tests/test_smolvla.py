"""SmolVLA pipeline wiring (no real weights / lerobot required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import StageExecutionType
from nanovllm_omni.models.smolvla import stage as smolvla_stage
from nanovllm_omni.outputs import ActionArtifact, OmniRequestOutput

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "smolvla.yaml"


class _Arr:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def test_pipeline_registry_resolves_smolvla():
    config = resolve_pipeline_config("smolvla")
    assert config is not None
    assert [s.name for s in config.stages] == ["vla"]
    assert config.stages[0].kind == StageExecutionType.LLM_GENERATION
    assert config.stages[0].is_terminal is True
    assert config.stages[0].final_output_type == "actions"
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


def test_vla_stage_requires_lerobot(monkeypatch):
    def boom() -> None:
        raise ImportError("lerobot is required for SmolVLA")

    monkeypatch.setattr(smolvla_stage, "_import_smolvla_policy", boom)
    with pytest.raises(ImportError, match="lerobot"):
        smolvla_stage._vla_stage(None, OmniEngineArgs(model="smolvla"))


def test_vla_stage_requires_image(monkeypatch):
    class _Policy:
        def predict_action_chunk(self, batch):
            raise AssertionError("should not run")

    monkeypatch.setattr(smolvla_stage, "_load_policy", lambda args: _Policy())
    forward = smolvla_stage._vla_stage(None, OmniEngineArgs(model="smolvla", device="cpu"))
    with pytest.raises(ValueError, match=r"extra\['image'\]"):
        forward("pick up the mug", SamplingParams())


def test_omni_generate_returns_actions(monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    import numpy as np

    class _Policy:
        def predict_action_chunk(self, batch):
            assert "observation.images.image" in batch
            assert batch["task"] == ["pick up the mug"]
            return np.zeros((1, 50, 7), dtype=np.float32)

        preprocessor = staticmethod(lambda b: b)
        postprocessor = staticmethod(lambda b: b)

    monkeypatch.setattr(smolvla_stage, "_load_policy", lambda args: _Policy())

    from nanovllm_omni import Omni

    omni = Omni("smolvla", device="cpu", deploy_config_path=str(DEPLOY))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    out = omni.generate(
        ["pick up the mug"],
        SamplingParams(extra={"image": image, "wrist_image": image, "state": state}),
    )[0]
    action = out.multimodal_output["actions"]
    assert isinstance(action, ActionArtifact)
    assert action.chunk_size == 50
    assert action.action_dim == 7


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


def test_as_nchw_accepts_image_bytes(monkeypatch) -> None:
    """TK-017: base64-decoded image bytes decode to a batch-NCHW tensor."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8)).save(buf, format="PNG")
    tensor = smolvla_stage._as_nchw(buf.getvalue(), None)
    assert tuple(tensor.shape) == (1, 3, 16, 24)
    assert tensor.dtype != "object"
