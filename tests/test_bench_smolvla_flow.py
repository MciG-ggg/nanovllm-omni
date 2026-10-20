"""Tests for nanovllm_omni.bench.bench_smolvla_flow (the ncu flow target).

The flow bench drives the action stage only, so the ncu launch window
lands on the flow-matching kernels instead of the VLM GEMMs. The loop
is checked here with a fake action stage; the GPU path needs lerobot.
"""

from __future__ import annotations

from types import SimpleNamespace

from nanovllm_omni.bench import bench_smolvla_flow


class _FakeAction:
    """Records the calls the flow loop makes."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def prepare(self, payload, sampling=None):
        self.calls.append(("prepare",))
        return SimpleNamespace(metadata={}, latents=None)

    def denoise_step(self, state, *, step, num_steps):
        self.calls.append(("denoise", step, num_steps))
        return 0.1

    def step_scheduler(self, state, noise_pred):
        self.calls.append(("schedule", noise_pred))


def test_flow_loop_prepares_once_and_runs_every_step() -> None:
    action = _FakeAction()

    wall = bench_smolvla_flow.flow_loop(action, payload=None, sampling=None, num_steps=3)

    assert wall >= 0.0
    assert action.calls[0] == ("prepare",)
    assert [call[1] for call in action.calls if call[0] == "denoise"] == [0, 1, 2]
    assert len([call for call in action.calls if call[0] == "schedule"]) == 3


def test_flow_loop_passes_the_step_count() -> None:
    action = _FakeAction()

    bench_smolvla_flow.flow_loop(action, payload=None, sampling=None, num_steps=7)

    assert {call[2] for call in action.calls if call[0] == "denoise"} == {7}
