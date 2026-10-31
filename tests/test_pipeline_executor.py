"""PipelineExecutor async tests with stub runners."""

from __future__ import annotations

import asyncio
import threading

import pytest

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


def _make_overlap_executor() -> PipelineExecutor:
    pipeline = PipelineConfig(
        name="t",
        stages=(
            StageConfig(
                0,
                "s",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "overlap_stage"),
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="t.yaml",
    )
    args = OmniEngineArgs(model="m", device="cpu")
    return PipelineExecutor(pipeline=pipeline, deploy=DeployConfig(), args=args, max_concurrent=4)


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


def test_run_many_retires_earlier_requests_when_one_fails_to_enter(monkeypatch):
    """A mid-batch ``add_request`` failure must not leave the rest queued.

    They would otherwise be drained — and their results discarded — by
    whichever caller happens to come next.
    """
    executor = _make_executor()
    calls = {"n": 0}
    original = executor._runner.add_request

    def _fail_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second request failed")
        return original(*args, **kwargs)

    try:
        monkeypatch.setattr(executor._runner, "add_request", _fail_on_second)
        with pytest.raises(OSError, match="second request failed"):
            executor.run_many(["a", "b", "c"], [None, None, None])
        assert not executor._runner.has_unfinished()
    finally:
        executor.close()


def test_executor_max_concurrent_must_be_positive():
    with pytest.raises(ValueError):
        _make_executor(max_concurrent=0)


def test_executor_serialises_concurrent_sync_callers():
    """The runner keeps in-flight state, so concurrent callers must not overlap.

    Two threads inside ``step`` at once would both drive ``_in_flight`` and
    one would consume the other's finished request. ``overlap_stage``
    records the deepest overlap it saw, which makes the race very likely to
    show up rather than certain to; without the lock it hangs, so it does
    discriminate in practice.
    """
    executor = _make_overlap_executor()
    # White-box on purpose: the stage is where the overlap is observable.
    stage = executor._runner._ensure_stages()[0]
    results: list[str] = []
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for _ in range(5):
                results.append(executor.run(f"p{index}")[0])
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        executor.close()

    assert errors == []
    assert len(results) == 20
    assert stage.max_depth == 1
