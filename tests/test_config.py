from pathlib import Path

import pytest

from nanovllm_omni.config import (
    DeployConfig,
    DeployStageConfig,
    OmniEngineArgs,
    PipelineConfig,
    SamplingParams,
    StageConfig,
    load_deploy_config,
    merge_pipeline_deploy,
    resolve_pipeline_config,
)
from nanovllm_omni.config.registry import (
    StageExecutionType,
    resolve_stage_factory,
)
from tests import _stage_factories as fac


def test_engine_args_accepts_deploy_kwargs():
    args = OmniEngineArgs(gpu_memory_utilization=0.8, max_num_seqs=4)
    assert args.gpu_memory_utilization == 0.8
    assert args.max_num_seqs == 4


def test_engine_args_swallows_unknown_kwargs():
    args = OmniEngineArgs(unknown_flag=1, quantum_entanglement=True)
    assert args.extra["unknown_flag"] == 1
    assert args.extra["quantum_entanglement"] is True


def test_sampling_params_matches_aligned_surface():
    params = SamplingParams(temperature=0.7, max_tokens=8, top_p=0.9, top_k=50, seed=42)
    assert params.max_tokens == 8
    assert params.top_k == 50
    assert params.seed == 42


def test_pipeline_registry_resolves_minimind():
    config = resolve_pipeline_config("minimind_o")
    assert isinstance(config, PipelineConfig)
    assert [s.name for s in config.stages] == ["thinker", "talker", "code2wav"]
    assert config.default_deploy_config_name == "minimind_omni.yaml"
    assert config.registration_handles == ("minimind_o", "jingyaogong/minimind-3o")


def test_pipeline_registry_resolves_by_hf_handle():
    config = resolve_pipeline_config("jingyaogong/minimind-3o")
    assert isinstance(config, PipelineConfig)
    assert config.name == "minimind_o"


def test_pipeline_registry_returns_none_for_unknown():
    assert resolve_pipeline_config("nonexistent_model") is None


def test_stage_config_is_frozen():
    cfg = StageConfig(
        stage_id=0,
        name="thinker",
        kind=StageExecutionType.LLM_AR,
        factory="tests._stage_factories:thinker_simple",
    )
    try:
        cfg.name = "talker"
    except Exception:
        pass
    else:
        raise AssertionError("StageConfig must be frozen")


def test_deploy_config_parses_stage_resources_and_unknown_sampling_keys(tmp_path: Path):
    path = tmp_path / "resources.yaml"
    path.write_text(
        "stages:\n"
        "  - name: thinker\n"
        "    max_num_batched_tokens: '512'\n"
        "    max_num_seqs: '2'\n"
        "    gpu_memory_utilization: '0.6'\n"
        "    device: cpu\n"
        "    devices: [cpu, cpu]\n"
        "    default_sampling_params: {temperature: 0.7, custom_key: keep}\n",
        encoding="utf-8",
    )
    stage = load_deploy_config(path).stages[0]
    assert stage.max_num_batched_tokens == 512
    assert stage.max_num_seqs == 2
    assert stage.gpu_memory_utilization == 0.6
    assert stage.device == "cpu"
    assert stage.devices == ("cpu", "cpu")
    assert stage.default_sampling_params["custom_key"] == "keep"


def test_deploy_stage_config_validates_resource_values():
    with pytest.raises(ValueError, match="max_num_seqs"):
        DeployStageConfig(name="x", max_num_seqs=0)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        DeployStageConfig(name="x", gpu_memory_utilization=1.1)


def test_deploy_config_merges_stage_defaults(tmp_path: Path):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "stages:\n"
        "  - name: thinker\n"
        "    default_sampling_params:\n"
        "      temperature: 0.7\n"
        "      max_tokens: 512\n"
        "  - name: talker\n"
        "    default_sampling_params:\n"
        "      temperature: 0.2\n"
        "      watchdog_limit: 192\n"
        "  - name: code2wav\n"
        "    default_sampling_params: {}\n",
        encoding="utf-8",
    )
    deploy = load_deploy_config(path)
    assert isinstance(deploy, DeployConfig)
    assert deploy.post_eos_padding_count == 128
    assert deploy.internal_stop_token_id == 17
    assert deploy.talker_max_steps_after_last_thinker_token == 192
    assert {s.name for s in deploy.stages} == {"thinker", "talker", "code2wav"}

    pipeline = resolve_pipeline_config("minimind_o")
    assert pipeline is not None
    merged = merge_pipeline_deploy(pipeline, deploy)
    by_name = {stage.name: defaults for stage, defaults in merged}
    assert by_name["thinker"]["temperature"] == 0.7
    assert by_name["thinker"]["max_tokens"] == 512
    assert by_name["talker"]["temperature"] == 0.2
    assert by_name["talker"]["watchdog_limit"] == 192
    assert by_name["code2wav"] == {}
    thinker_deploy = next(stage for stage in deploy.stages if stage.name == "thinker")
    assert thinker_deploy.max_num_batched_tokens is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("post_eos_padding_count", -1),
        ("post_eos_padding_count", "many"),
        ("internal_stop_token_id", "stop"),
        ("talker_max_steps_after_last_thinker_token", 1.5),
    ],
)
def test_deploy_config_rejects_invalid_phase_2_values(tmp_path: Path, key: str, value: object):
    path = tmp_path / "invalid-deploy.yaml"
    path.write_text(f"{key}: {value!r}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_deploy_config(path)


def test_deploy_config_accepts_disabled_talker_watchdog(tmp_path: Path):
    path = tmp_path / "disabled-watchdog.yaml"
    path.write_text(
        "post_eos_padding_count: 0\n"
        "internal_stop_token_id: 17\n"
        "talker_max_steps_after_last_thinker_token: -1\n",
        encoding="utf-8",
    )
    deploy = load_deploy_config(path)
    assert deploy.post_eos_padding_count == 0
    assert deploy.internal_stop_token_id == 17
    assert deploy.talker_max_steps_after_last_thinker_token == -1


def test_deploy_config_dataclass_validates_phase_2_values():
    with pytest.raises(ValueError, match="post_eos_padding_count"):
        DeployConfig(post_eos_padding_count=-1)
    with pytest.raises(ValueError, match="internal_stop_token_id"):
        DeployConfig(internal_stop_token_id="stop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="talker_max_steps"):
        DeployConfig(talker_max_steps_after_last_thinker_token=1.5)  # type: ignore[arg-type]


def test_register_pipeline_adds_entry():
    from nanovllm_omni.config.registry import (
        OMNI_PIPELINES,
        register_pipeline,
    )

    fake = PipelineConfig(
        name="fake_o",
        stages=(
            StageConfig(
                stage_id=0,
                name="only",
                kind=StageExecutionType.LLM_AR,
                factory="tests._stage_factories:thinker_simple",
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="fake_o.yaml",
    )
    register_pipeline(fake)
    try:
        assert resolve_pipeline_config("fake_o") is fake
    finally:
        OMNI_PIPELINES.pop("fake_o", None)


# ---------------------------------------------------------------------------
# PipelineConfig extra fields (TK-010-rev alignment with vllm-omni):
# ``hf_architectures`` + ``hf_config_predicate`` drive the layer-6
# disambiguator in ``OmniBase.try_infer_model_type``.
# ---------------------------------------------------------------------------


def test_pipeline_config_accepts_hf_architectures_and_predicate():
    pred = lambda c: getattr(c, "version", "") == "4.5"  # noqa: E731
    cfg = PipelineConfig(
        name="fake_o",
        stages=(
            StageConfig(
                stage_id=0,
                name="only",
                kind=StageExecutionType.LLM_AR,
                factory="tests._stage_factories:thinker_simple",
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="fake_o.yaml",
        hf_architectures=("FakeArch",),
        hf_config_predicate=pred,
    )
    assert cfg.hf_architectures == ("FakeArch",)
    assert cfg.hf_config_predicate is pred


def test_pipeline_config_hf_fields_default_to_empty():
    cfg = PipelineConfig(
        name="minimal",
        stages=(
            StageConfig(
                stage_id=0,
                name="only",
                kind=StageExecutionType.LLM_AR,
                factory="tests._stage_factories:thinker_simple",
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="minimal.yaml",
    )
    assert cfg.hf_architectures == ()
    assert cfg.hf_config_predicate is None


def test_smolvla_pipeline_declares_hf_architectures():
    """Lock the SmolVLA claim on the LeRobot policy class name."""
    cfg = resolve_pipeline_config("smolvla")
    assert cfg is not None
    assert cfg.hf_architectures == ("SmolVLAPolicy",)


# ---------------------------------------------------------------------------
# Phase 2 (TK-016) contract tests: StageExecutionType enum + string-path
# factory resolution.
# ---------------------------------------------------------------------------


def test_stage_execution_type_has_vllm_omni_taxonomy():
    # vllm-omni's StageExecutionType taxonomy: LLM_AR / LLM_GENERATION /
    # DIFFUSION / CODEC. StrEnum so members compare equal to legacy strings.
    assert {e.name for e in StageExecutionType} == {
        "LLM_AR",
        "LLM_GENERATION",
        "DIFFUSION",
        "CODEC",
    }
    assert StageExecutionType.LLM_AR == "ar"
    assert StageExecutionType.LLM_GENERATION == "generation"
    assert StageExecutionType.DIFFUSION == "diffusion"
    assert StageExecutionType.CODEC == "codec"


def test_stage_config_kind_rejects_string_literal():
    with pytest.raises(TypeError, match="StageExecutionType"):
        StageConfig(
            stage_id=0,
            name="x",
            kind="ar",  # type: ignore[arg-type]
            factory="tests._stage_factories:thinker_simple",
        )


def test_stage_config_factory_must_be_string_path():
    with pytest.raises(TypeError, match="dotted-path string"):
        StageConfig(
            stage_id=0,
            name="x",
            kind=StageExecutionType.LLM_AR,
            factory=fac.thinker_simple,  # type: ignore[arg-type]
        )


def test_stage_config_bad_factory_form_raises_value_error():
    with pytest.raises(ValueError, match="package.module:attr"):
        StageConfig(
            stage_id=0,
            name="x",
            kind=StageExecutionType.LLM_AR,
            factory="no_colon_separator",
        )


def test_stage_config_missing_attribute_raises_at_construction():
    with pytest.raises(AttributeError, match="has no attribute"):
        StageConfig(
            stage_id=0,
            name="x",
            kind=StageExecutionType.LLM_AR,
            factory="tests._stage_factories:does_not_exist",
        )


def test_stage_config_unimportable_module_raises_at_construction():
    with pytest.raises(ImportError, match="cannot import module"):
        StageConfig(
            stage_id=0,
            name="x",
            kind=StageExecutionType.LLM_AR,
            factory="definitely_not_a_real_module:_x",
        )


def test_resolve_stage_factory_returns_callable():
    fn = resolve_stage_factory("nanovllm_omni.models.minimind_omni.thinker:_thinker_stage")
    assert callable(fn)


def test_resolve_stage_factory_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_stage_factory("")


def test_resolve_stage_factory_rejects_missing_colon():
    with pytest.raises(ValueError, match="package.module:attr"):
        resolve_stage_factory("nanovllm_omni.models.minimind_omni.thinker")
