"""Tests for the diffusion engine module.

Verifies:
- Protocol structural shape (4 methods + supports_step_execution)
- Scheduler: FIFO order, finish removes from running
- Engine: IdentityPipeline runs N steps correctly
- Engine: run_sync and step_streaming produce same output (ADR-024)
- InlineDiffusionClient: wraps a pipeline and exposes run()

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §5.1.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _IdentityPipeline:
    """Minimal DiffusionPipeline that adds 0.1 per step."""

    supports_step_execution = True

    def __init__(self, num_steps: int = 3) -> None:
        self.num_steps = num_steps

    def prepare_encode(self, request: object) -> object:
        from nanovllm_omni.diffusion.interface import StepState

        return StepState(
            request_id=getattr(request, "request_id", "t"),
            latents=0.0,
        )

    def denoise_step(self, state: object, *, step: int, num_steps: int) -> object:
        return 0.1

    def step_scheduler(self, state: object, noise_pred: object) -> None:
        state.latents = state.latents + noise_pred

    def post_decode(self, state: object) -> object:
        from nanovllm_omni.diffusion.interface import DiffusionOutput

        return DiffusionOutput(images=[state.latents], finished=True)


@dataclass
class _FakeSampling:
    extra: dict


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


def test_diffusion_pipeline_protocol_structure():
    """Protocol must expose the 4 required methods + supports_step_execution."""
    from nanovllm_omni.diffusion.interface import DiffusionPipeline

    pipe = _IdentityPipeline()
    # Runtime checkable protocol.
    assert isinstance(pipe, DiffusionPipeline)
    assert pipe.supports_step_execution is True
    for method in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        assert hasattr(pipe, method), f"missing method: {method}"


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


def test_scheduler_fifo_order():
    """Requests are processed in FIFO order."""
    from nanovllm_omni.diffusion.scheduler import RequestScheduler

    sched = RequestScheduler()
    sched.add_request("a")
    sched.add_request("b")
    sched.add_request("c")

    first = sched.schedule()
    assert first is not None
    assert first.request == "a"


def test_scheduler_finish_removes_request():
    """After finish(), running queue is empty."""
    from nanovllm_omni.diffusion.scheduler import RequestScheduler

    sched = RequestScheduler()
    sched.add_request("a")
    first = sched.schedule()
    assert first is not None
    assert sched.num_running() == 1

    sched.finish(first.request_id)
    assert sched.num_running() == 0
    assert not sched.has_requests()


def test_scheduler_step_counter_advances():
    """Each schedule() call after the first returns the next step_id."""
    from nanovllm_omni.diffusion.scheduler import RequestScheduler

    sched = RequestScheduler()
    sched.add_request("a")
    first = sched.schedule()
    assert first is not None
    assert first.step_id == 0
    sched.update_from_output(first, None)

    second = sched.schedule()
    assert second is not None
    assert second.step_id == 1


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------


def test_engine_run_sync_correct_total():
    """3 steps × 0.1 = 0.3 in state.latents."""
    from nanovllm_omni.diffusion.engine import DiffusionEngine
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=3)
    engine = DiffusionEngine(DiffusionRunner(pipe))
    req = OmniDiffusionRequest(
        request_id="t",
        prompt="hi",
        num_inference_steps=3,
    )
    outputs = engine.run_sync(req)
    assert len(outputs) == 1
    assert outputs[0].images[0] == pytest.approx(0.3)


def test_engine_run_sync_and_step_streaming_match():
    """ADR-024: run_sync and step_streaming produce the same result."""
    from nanovllm_omni.diffusion.engine import DiffusionEngine
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=3)
    engine = DiffusionEngine(DiffusionRunner(pipe))
    req = OmniDiffusionRequest(request_id="t", prompt="hi", num_inference_steps=3)

    sync_out = engine.run_sync(req)

    async def collect_streaming():
        outputs = []
        async for batch in engine.step_streaming(req):
            outputs.extend(batch)
        return outputs

    streamed = asyncio.run(collect_streaming())
    assert len(streamed) == len(sync_out)
    assert streamed[0].images[0] == pytest.approx(sync_out[0].images[0])


def test_engine_single_step():
    """Single step: 1 × 0.1 = 0.1."""
    from nanovllm_omni.diffusion.engine import DiffusionEngine
    from nanovllm_omni.diffusion.request import OmniDiffusionRequest
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=1)
    engine = DiffusionEngine(DiffusionRunner(pipe))
    req = OmniDiffusionRequest(request_id="t", prompt="hi", num_inference_steps=1)
    outputs = engine.run_sync(req)
    assert outputs[0].images[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


def test_inline_diffusion_client_runs_pipeline():
    """InlineDiffusionClient.run() returns the first image from the pipeline."""
    from nanovllm_omni.diffusion.client import InlineDiffusionClient

    pipe = _IdentityPipeline(num_steps=2)
    client = InlineDiffusionClient(pipe)

    sampling = _FakeSampling(extra={"num_inference_steps": 2})
    result = client.run("a cat", sampling)
    assert result == pytest.approx(0.2)  # 2 × 0.1


def test_inline_diffusion_client_handles_dict_payload():
    """InlineDiffusionClient accepts dict payloads ({\"prompt\": ...})."""
    from nanovllm_omni.diffusion.client import InlineDiffusionClient

    pipe = _IdentityPipeline(num_steps=1)
    client = InlineDiffusionClient(pipe)
    sampling = _FakeSampling(extra={"num_inference_steps": 1})
    result = client.run({"prompt": "a sunset"}, sampling)
    assert result == pytest.approx(0.1)


def test_inline_diffusion_client_stage_type():
    """stage_type class attribute must be 'diffusion' (mirror doc §2.3)."""
    from nanovllm_omni.diffusion.client import InlineDiffusionClient

    assert InlineDiffusionClient.stage_type == "diffusion"
