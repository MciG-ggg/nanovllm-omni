"""``generate`` batches across prompts: every prompt is in flight before the first step."""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
)
from nanovllm_omni.engine.executor import PipelineExecutor
from nanovllm_omni.entrypoints.omni import _make_output
from tests import _stage_factories as fac


def _batching_pipeline() -> PipelineConfig:
    return PipelineConfig(
        name="batching",
        stages=(
            StageConfig(
                0,
                "batched",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "batching_simple"),
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="batching.yaml",
    )


def _make_executor() -> PipelineExecutor:
    from nanovllm_omni.config.params import OmniEngineArgs

    return PipelineExecutor(
        pipeline=_batching_pipeline(),
        deploy=DeployConfig(),
        args=OmniEngineArgs(model="m", device="cpu"),
    )


def test_run_many_hands_one_stage_every_request_in_a_single_batch():
    fac.reset_batch_sizes()
    executor = _make_executor()
    try:
        results = executor.run_many(["a", "b", "c"], [None, None, None])
    finally:
        executor.close()

    assert fac.get_batch_sizes() == [3]
    assert [payload for payload, _ in results] == ["b|a", "b|b", "b|c"]


def test_run_many_returns_one_output_per_request_in_submission_order():
    fac.reset_batch_sizes()
    executor = _make_executor()
    try:
        results = executor.run_many(["a", "b"], [None, None])
    finally:
        executor.close()

    outputs = [_make_output(payload, stage_ms, "audio") for payload, stage_ms in results]

    assert len(outputs) == 2
    assert [set(out.custom_output["stage_ms"]) for out in outputs] == [
        {"batched"},
        {"batched"},
    ]
