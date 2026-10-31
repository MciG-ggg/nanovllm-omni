"""PipelineRunner request lifecycle: add_request / step.

The runner is a synchronous engine loop in the shape of
``vllm.LLMEngine``: ``add_request`` puts a request in the queue of the
first stage, and ``step`` lets one stage serve what it is holding and
pushes the result into the next stage's queue.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
)
from nanovllm_omni.engine.runner import PipelineRunner
from tests import _stage_factories as fac


def _make_args() -> OmniEngineArgs:
    return OmniEngineArgs(model="test_model", device="cpu")


def _pipeline(stages: list[StageConfig]) -> PipelineConfig:
    return PipelineConfig(
        name="lifecycle",
        stages=tuple(stages),
        default_deploy_config_name="lifecycle.yaml",
    )


def _single_stage(factory: str) -> PipelineConfig:
    return _pipeline(
        [
            StageConfig(
                0,
                "batched",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", factory),
                is_terminal=True,
            )
        ]
    )


def _two_stage() -> PipelineConfig:
    return _pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "logged_thinker"),
            ),
            StageConfig(
                1,
                "talker",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "logged_talker"),
                process_input=("tests._stage_factories", "identity_process_input"),
                input_sources=(0,),
                is_terminal=True,
            ),
        ]
    )


def test_step_advances_one_stage_and_returns_nothing_until_terminal():
    fac.reset_log()
    runner = PipelineRunner(_two_stage(), DeployConfig(), _make_args())
    runner.add_request("a")

    assert runner.step() == []
    assert [entry[0] for entry in fac.get_log()] == ["thinker"]

    finished = runner.step()
    assert [out.payload for out in finished] == ["a->thinker->talker"]
    assert [entry[0] for entry in fac.get_log()] == ["thinker", "talker"]
    assert not runner.has_unfinished()


def test_step_serves_every_request_queued_at_one_stage():
    fac.reset_batch_sizes()
    runner = PipelineRunner(_single_stage("batching_simple"), DeployConfig(), _make_args())
    runner.add_request("a")
    runner.add_request("b")

    finished = runner.step()

    assert fac.get_batch_sizes() == [2]
    assert [out.payload for out in finished] == ["b|a", "b|b"]


def test_step_serves_the_next_stage_in_the_rotation():
    fac.reset_log()
    runner = PipelineRunner(_two_stage(), DeployConfig(), _make_args())
    runner.add_request("a")
    runner.step()
    runner.add_request("b")
    fac.reset_log()

    runner.step()
    runner.step()

    assert [entry[0] for entry in fac.get_log()] == ["talker", "thinker"]


def test_add_request_returns_a_distinct_request_id():
    runner = PipelineRunner(_single_stage("batching_simple"), DeployConfig(), _make_args())

    first = runner.add_request("a")
    second = runner.add_request("b", request_id="mine")

    assert first != second
    assert second == "mine"


def test_add_request_rejects_a_duplicate_request_id():
    runner = PipelineRunner(_single_stage("batching_simple"), DeployConfig(), _make_args())
    runner.add_request("a", request_id="mine")

    with pytest.raises(ValueError):
        runner.add_request("b", request_id="mine")


def _batching_then_request() -> PipelineConfig:
    """Stage 0 batches across requests; stage 1 serves one at a time."""
    return _pipeline(
        [
            StageConfig(
                0,
                "batched",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "batching_simple"),
            ),
            StageConfig(
                1,
                "tail",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "logged_talker"),
                process_input=("tests._stage_factories", "identity_process_input"),
                input_sources=(0,),
                is_terminal=True,
            ),
        ]
    )


def test_request_added_while_another_is_in_flight_still_finishes():
    fac.reset_batch_sizes()
    fac.reset_log()
    runner = PipelineRunner(_batching_then_request(), DeployConfig(), _make_args())

    runner.add_request("a", request_id="a")
    assert runner.step() == []
    runner.add_request("b", request_id="b")

    finished = []
    while runner.has_unfinished():
        finished.extend(runner.step())

    assert [(out.request_id, out.payload) for out in finished] == [
        ("a", "b|a->talker"),
        ("b", "b|b->talker"),
    ]


def test_request_added_while_another_is_in_flight_does_not_join_its_batch():
    """A late arrival only batches with what is still queued, not with a
    request the stage already served."""
    fac.reset_batch_sizes()
    runner = PipelineRunner(_batching_then_request(), DeployConfig(), _make_args())

    runner.add_request("a")
    runner.step()
    runner.add_request("b")
    while runner.has_unfinished():
        runner.step()

    assert fac.get_batch_sizes() == [1, 1]


def test_a_late_arrival_does_not_preempt_a_request_already_past_it():
    """Stages are served round-robin, so a new arrival waits its turn.

    The request that already reached the talker keeps draining instead of
    being pushed behind the newcomer at stage 0.
    """
    fac.reset_log()
    runner = PipelineRunner(_two_stage(), DeployConfig(), _make_args())
    runner.add_request("a")
    runner.step()
    runner.add_request("b")
    fac.reset_log()

    runner.step()

    assert [entry[0] for entry in fac.get_log()] == ["talker"]
    assert [entry[1] for entry in fac.get_log()] == ["a->thinker"]


def test_a_stage_that_raises_does_not_leave_the_runner_stuck():
    """A failing stage must retire its requests, not strand them.

    ``run`` drains until nothing is in flight; a request the stage still
    holds after raising would make every later call spin forever.
    """
    runner = PipelineRunner(_single_stage("raising_simple"), DeployConfig(), _make_args())
    runner.add_request("a")

    with pytest.raises(RuntimeError, match="stage exploded"):
        runner.step()

    assert not runner.has_unfinished()
    assert runner.step() == []


def test_a_process_input_that_raises_retires_the_request():
    """The bridge runs inside the guard, so a failure cannot strand the batch.

    ``_advance`` calls ``process_input`` after the stage returned, so this
    failure happens one hop later than a stage that raises.
    """
    pipeline = _pipeline(
        [
            StageConfig(
                0,
                "thinker",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "thinker_simple"),
            ),
            StageConfig(
                1,
                "talker",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "talker_simple"),
                process_input=("tests._stage_factories", "raising_process_input"),
                input_sources=(0,),
                is_terminal=True,
            ),
        ]
    )
    runner = PipelineRunner(pipeline, DeployConfig(), _make_args())
    runner.add_request("a")

    with pytest.raises(ValueError, match="bridge exploded"):
        runner.step()

    assert not runner.has_unfinished()
    assert runner.step() == []


def test_a_first_stage_that_fails_to_build_retires_the_request(monkeypatch):
    """Loading the first stage happens inside ``add_request``."""
    from nanovllm_omni.config.registry import resolve_stage_factory as real_resolve

    def _boom(name):
        raise OSError("no weights")

    pipeline = _single_stage("batching_simple")
    runner = PipelineRunner(pipeline, DeployConfig(), _make_args())
    monkeypatch.setattr("nanovllm_omni.engine.runner.resolve_stage_factory", _boom)

    with pytest.raises(OSError, match="no weights"):
        runner.add_request("a")

    monkeypatch.setattr("nanovllm_omni.engine.runner.resolve_stage_factory", real_resolve)
    assert not runner.has_unfinished()


def test_stage_ms_carries_the_batch_wall_to_every_request_in_it():
    runner = PipelineRunner(_two_stage(), DeployConfig(), _make_args())
    runner.add_request("a")
    runner.add_request("b")

    finished = []
    while runner.has_unfinished():
        finished.extend(runner.step())

    assert [set(out.stage_ms) for out in finished] == [
        {"thinker", "talker"},
        {"thinker", "talker"},
    ]
