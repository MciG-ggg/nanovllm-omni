"""PipelineExecutor async tests with stub runners."""

from __future__ import annotations

import asyncio

from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
)
from nanovllm_omni.engine.executor import PipelineExecutor


def _make_executor(max_concurrent: int = 1) -> PipelineExecutor:
    pipeline = PipelineConfig(
        name="t",
        stages=(
            StageConfig(
                0,
                "s",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "executor_simple"),
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="t.yaml",
    )
    args = OmniEngineArgs(model="m", device="cpu")
    return PipelineExecutor(
        pipeline=pipeline, deploy=DeployConfig(), args=args, max_concurrent=max_concurrent
    )


def test_executor_submit_is_async():
    async def go() -> str:
        executor = _make_executor()
        try:
            payload, _ = await executor.submit("hello")
            return payload
        finally:
            executor.close()

    assert asyncio.run(go()) == "out(hello)"


def test_executor_submit_sequential():
    async def go() -> list[str]:
        executor = _make_executor()
        try:
            return [(await executor.submit(f"p{i}"))[0] for i in range(3)]
        finally:
            executor.close()

    assert asyncio.run(go()) == ["out(p0)", "out(p1)", "out(p2)"]


def test_executor_run_is_sync_and_returns_stage_ms():
    executor = _make_executor()
    try:
        payload, stage_ms = executor.run("hello")
    finally:
        executor.close()

    assert payload == "out(hello)"
    assert list(stage_ms) == ["s"]


def test_executor_max_concurrent_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        _make_executor(max_concurrent=0)
