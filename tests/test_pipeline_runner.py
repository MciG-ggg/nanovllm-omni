"""PipelineRunner tests with stub factories (no model weights required)."""

from __future__ import annotations

import dataclasses

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import (
    DeployConfig,
    DeployStageConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
)
from nanovllm_omni.engine.runner import PipelineRunner
from tests import _stage_factories as fac


def _make_pipeline(stages: list[StageConfig]) -> PipelineConfig:
    return PipelineConfig(
        name="test_pipeline",
        stages=tuple(stages),
        default_deploy_config_name="test.yaml",
    )


def _make_deploy() -> DeployConfig:
    return DeployConfig(
        stages=(
            DeployStageConfig(
                name="thinker",
                default_sampling_params={"temperature": 0.7, "max_tokens": 256},
            ),
            DeployStageConfig(
                name="talker",
                default_sampling_params={"temperature": 0.2},
            ),
        )
    )


def _make_args() -> OmniEngineArgs:
    return OmniEngineArgs(model="test_model", device="cpu")


def test_runner_visits_stages_in_order():
    fac.reset_log()
    pipeline = _make_pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:logged_thinker",
            ),
            StageConfig(
                1,
                "talker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:logged_talker",
                process_input="tests._stage_factories:identity_process_input",
                input_sources=(0,),
            ),
            StageConfig(
                2,
                "code2wav",
                StageExecutionType.CODEC,
                "tests._stage_factories:logged_code2wav",
                process_input="tests._stage_factories:identity_process_input",
                input_sources=(1,),
                is_terminal=True,
                final_output_type="audio",
            ),
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    out = runner.run("hello")
    assert out == "hello->thinker->talker->code2wav"
    assert [entry[0] for entry in fac.get_log()] == ["thinker", "talker", "code2wav"]


def test_runner_calls_process_input_between_stages():
    pipeline = _make_pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:thinker_simple",
            ),
            StageConfig(
                1,
                "talker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:talker_simple",
                process_input="tests._stage_factories:bridge_process_input",
                input_sources=(0,),
                is_terminal=True,
            ),
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    out = runner.run("hello")
    assert out == "tk|br|th|hello"


def test_runner_merges_deploy_defaults_with_request_sampling():
    fac.reset_captures()
    pipeline = _make_pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:capturing_simple",
                is_terminal=True,
            )
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    runner.run("hi", SamplingParams(temperature=0.1, max_tokens=8))
    assert len(fac.get_captures()) == 1
    sp = fac.get_captures()[0]
    assert sp.temperature == 0.1
    assert sp.max_tokens == 8
    assert sp.extra["max_tokens"] == 256  # from deploy default


def test_runner_uses_deploy_defaults_when_no_request_sampling():
    fac.reset_captures()
    pipeline = _make_pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:capturing_simple",
                is_terminal=True,
            )
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    runner.run("hi")
    assert len(fac.get_captures()) == 1
    sp = fac.get_captures()[0]
    assert sp.temperature == 0.7
    assert sp.extra["temperature"] == 0.7
    assert sp.extra["max_tokens"] == 256


def test_single_stage_pipeline_supported():
    fac.reset_diffusion_log()
    pipeline = PipelineConfig(
        name="wan2_2_ti2v",
        stages=(
            StageConfig(
                0,
                "dit",
                StageExecutionType.DIFFUSION,
                "tests._stage_factories:logged_diffusion",
                is_terminal=True,
                final_output_type="video",
                diffusers_class_name="WanPipeline",
            ),
        ),
        default_deploy_config_name="wan2_2_ti2v.yaml",
    )
    runner = PipelineRunner(pipeline, DeployConfig(), _make_args())
    out = runner.run("a cat")
    assert out == "video"
    log = fac.get_diffusion_log()
    assert len(log) == 1
    assert log[0][0] == "dit"


def test_runner_passes_mode_and_stage_resources_to_factory():
    fac.reset_factory_observations()
    pipeline = _make_pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                "tests._stage_factories:observing_factory",
                is_terminal=True,
            )
        ],
    )
    pipeline = dataclasses.replace(pipeline, supported_pipeline_kinds=("collapsed", "full"))
    deploy = DeployConfig(
        stages=(
            DeployStageConfig(
                name="thinker",
                max_num_batched_tokens=512,
                max_num_seqs=2,
                gpu_memory_utilization=0.6,
                enforce_eager=True,
                device="cpu",
                devices=("cpu",),
            ),
        ),
    )
    PipelineRunner(pipeline, deploy, _make_args()).run("hello")
    observed_deploy, observed_args = fac.get_factory_observations()[0]
    assert observed_args.max_num_batched_tokens == 512
    assert observed_args.max_num_seqs == 2
    assert observed_args.gpu_memory_utilization == 0.6
    assert observed_args.enforce_eager is True
    assert observed_args.device == "cpu"
    assert observed_args.extra["devices"] == ("cpu",)


def test_minimind_full_mode_is_only_supported_kind():
    """Full is the only executable MiniMind mode (collapsed retired)."""
    from nanovllm_omni.models.minimind_omni.pipeline import MINIMIND_OMNI_PIPELINE

    assert MINIMIND_OMNI_PIPELINE.supported_pipeline_kinds == ("full",)
