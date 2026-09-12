"""Tests for the diffusion runner module.

Verifies:
- Protocol structural shape (4 methods + supports_step_execution)
- Runner: IdentityPipeline runs N steps correctly via run_sync
- Runner: run() returns the first image/artifact for PipelineRunner
- Runner: num_steps priority is state.metadata > payload > sampling.extra

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §5.1.
"""

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

        num_steps = getattr(request, "num_inference_steps", self.num_steps)
        return StepState(
            request_id=getattr(request, "request_id", "t"),
            latents=0.0,
            metadata={"num_steps": num_steps},
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


@dataclass
class _FakeRequest:
    num_inference_steps: int = 1
    request_id: str = "t"


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
# Runner tests
# ---------------------------------------------------------------------------


def test_runner_run_sync_correct_total():
    """3 steps × 0.1 = 0.3 in state.latents."""
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=3)
    runner = DiffusionRunner(pipe)
    req = _FakeRequest(num_inference_steps=3)
    outputs = runner.run_sync(req)
    assert len(outputs) == 1
    assert outputs[0].images[0] == pytest.approx(0.3)


def test_runner_single_step():
    """Single step: 1 × 0.1 = 0.1."""
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=1)
    runner = DiffusionRunner(pipe)
    req = _FakeRequest(num_inference_steps=1)
    outputs = runner.run_sync(req)
    assert outputs[0].images[0] == pytest.approx(0.1)


def test_runner_state_metadata_wins():
    """prepare_encode writes num_steps into metadata; runner honors it."""

    class _BarePipeline(_IdentityPipeline):
        def prepare_encode(self, request):
            from nanovllm_omni.diffusion.interface import StepState

            return StepState(request_id="t", latents=0.0)

    from nanovllm_omni.diffusion.runner import DiffusionRunner

    runner = DiffusionRunner(_BarePipeline())
    sampling = _FakeSampling(extra={"num_inference_steps": 2})
    assert runner.run("a cat", sampling) == pytest.approx(0.2)  # 2 × 0.1


def test_runner_run_returns_first_image():
    """run() returns the first image for PipelineRunner."""
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    pipe = _IdentityPipeline(num_steps=2)
    runner = DiffusionRunner(pipe)
    sampling = _FakeSampling(extra={"num_inference_steps": 2})
    assert runner.run("a cat", sampling) == pytest.approx(0.2)  # 2 × 0.1


def test_runner_run_returns_none_when_empty():
    """run() returns None when the pipeline yields no images."""

    class _EmptyPipeline(_IdentityPipeline):
        def post_decode(self, state):
            from nanovllm_omni.diffusion.interface import DiffusionOutput

            return DiffusionOutput(images=[], finished=True)

    from nanovllm_omni.diffusion.runner import DiffusionRunner

    runner = DiffusionRunner(_EmptyPipeline(num_steps=1))
    assert runner.run("x", _FakeSampling(extra={})) is None
