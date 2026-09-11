"""DiffusionRunner: wraps a DiffusionPipeline and drives the denoise loop.

Owns one ``DiffusionPipeline`` instance and exposes the step-by-step
API that ``DiffusionEngine`` calls.  The runner is the bridge between
the engine's scheduling loop and the model's 4-method contract.

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` §2.3.
"""

from __future__ import annotations

from typing import Any

from .interface import DiffusionOutput, DiffusionPipeline, StepState


class DiffusionRunner:
    """Drives one DiffusionPipeline through the denoise loop."""

    def __init__(self, pipeline: DiffusionPipeline) -> None:
        self.pipeline = pipeline

    def prepare(self, payload: Any, sampling: Any = None) -> StepState:
        """Encode conditions → initial StepState.

        Accepts the raw payload directly (no OmniDiffusionRequest conversion)
        so the pipeline receives the full payload with all metadata.
        Sampling is forwarded so pipelines can read deploy defaults
        (num_inference_steps/height/width) from ``sampling.extra``.
        """
        try:
            return self.pipeline.prepare_encode(payload, sampling)
        except TypeError:
            return self.pipeline.prepare_encode(payload)

    def denoise_step(
        self,
        state: StepState,
        *,
        step: int,
        num_steps: int,
    ) -> Any:
        """Run one denoise forward; returns noise_pred / velocity."""
        return self.pipeline.denoise_step(state, step=step, num_steps=num_steps)

    def step_scheduler(self, state: StepState, noise_pred: Any) -> None:
        """Update latents via scheduler step."""
        self.pipeline.step_scheduler(state, noise_pred)

    def post_decode(self, state: StepState) -> DiffusionOutput:
        """Decode latents → final output."""
        return self.pipeline.post_decode(state)

    def run_sync(self, payload: Any, sampling: Any = None) -> list[DiffusionOutput]:
        """Run the full denoise loop synchronously. Returns [DiffusionOutput]."""
        state = self.prepare(payload, sampling)
        # Determine num_steps from state metadata or sampling extras.
        num_steps = 1
        if sampling is not None and getattr(sampling, "extra", None):
            num_steps = int(sampling.extra.get("num_inference_steps", 1))
        if hasattr(state, "metadata") and "num_steps" in state.metadata:
            num_steps = state.metadata["num_steps"]
        if hasattr(payload, "num_inference_steps"):
            num_steps = payload.num_inference_steps
        for step in range(num_steps):
            noise_pred = self.denoise_step(state, step=step, num_steps=num_steps)
            self.step_scheduler(state, noise_pred)
        return [self.post_decode(state)]


__all__ = ["DiffusionRunner"]
