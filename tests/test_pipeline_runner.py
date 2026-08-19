"""PipelineRunner tests with stub factories (no model weights required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.config_registry import (
    DeployConfig,
    DeployStageConfig,
    PipelineConfig,
    StageConfig,
)
from nanovllm_omni.engine.runner import PipelineRunner
from nanovllm_omni.engine_args import OmniEngineArgs, SamplingParams


@dataclass
class _LoggedCall:
    stage: str
    payload: Any
    sampling: SamplingParams


def _make_stage_factory(name: str, log: list[_LoggedCall]):
    def factory(deploy: Any, args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            log.append(_LoggedCall(stage=name, payload=payload, sampling=sampling))
            return f"{payload}->{name}"

        return forward

    return factory


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
    log: list[_LoggedCall] = []
    pipeline = _make_pipeline(
        [
            StageConfig(0, "thinker", "ar", _make_stage_factory("thinker", log)),
            StageConfig(
                1,
                "talker",
                "ar",
                _make_stage_factory("talker", log),
                input_sources=(0,),
            ),
            StageConfig(
                2,
                "code2wav",
                "codec",
                _make_stage_factory("code2wav", log),
                input_sources=(1,),
                is_terminal=True,
                final_output_type="audio",
            ),
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    out = runner.run("hello")
    assert out == "hello->thinker->talker->code2wav"
    assert [c.stage for c in log] == ["thinker", "talker", "code2wav"]


def test_runner_calls_process_input_between_stages():
    talker_seen: list[Any] = []

    def thinker_factory(_deploy: Any, _args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            return f"th|{payload}"

        return forward

    def talker_factory(_deploy: Any, _args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            talker_seen.append(payload)
            return f"tk|{payload}"

        return forward

    pipeline = _make_pipeline(
        [
            StageConfig(0, "thinker", "ar", thinker_factory),
            StageConfig(
                1,
                "talker",
                "ar",
                talker_factory,
                process_input=lambda payload, prompt: f"br|{payload}",
                input_sources=(0,),
                is_terminal=True,
            ),
        ]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    out = runner.run("hello")
    assert talker_seen == ["br|th|hello"]
    assert out == "tk|br|th|hello"


def test_runner_merges_deploy_defaults_with_request_sampling():
    seen: list[SamplingParams] = []

    def capturing_factory(deploy: Any, args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            seen.append(sampling)
            return payload

        return forward

    pipeline = _make_pipeline(
        [StageConfig(0, "thinker", "ar", capturing_factory, is_terminal=True)]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    runner.run("hi", SamplingParams(temperature=0.1, max_tokens=8))
    assert len(seen) == 1
    sp = seen[0]
    assert sp.temperature == 0.1
    assert sp.max_tokens == 8
    assert sp.extra["max_tokens"] == 256  # from deploy default


def test_runner_uses_deploy_defaults_when_no_request_sampling():
    seen: list[SamplingParams] = []

    def capturing_factory(deploy: Any, args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            seen.append(sampling)
            return payload

        return forward

    pipeline = _make_pipeline(
        [StageConfig(0, "thinker", "ar", capturing_factory, is_terminal=True)]
    )
    runner = PipelineRunner(pipeline, _make_deploy(), _make_args())
    runner.run("hi")
    assert len(seen) == 1
    sp = seen[0]
    assert sp.temperature == 0.7
    assert sp.extra["temperature"] == 0.7
    assert sp.extra["max_tokens"] == 256


def test_single_stage_pipeline_supported():
    log: list[_LoggedCall] = []

    def diffusion_like(deploy: Any, args: Any) -> Any:
        def forward(payload: Any, sampling: SamplingParams) -> Any:
            log.append(_LoggedCall(stage="dit", payload=payload, sampling=sampling))
            return "video"

        return forward

    pipeline = PipelineConfig(
        name="wan2_2_ti2v",
        stages=(
            StageConfig(
                0,
                "dit",
                "diffusion",
                diffusion_like,
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
    assert len(log) == 1
    assert log[0].stage == "dit"
