"""DiffusionRunner: owns one DiffusionPipeline and drives the denoise loop.

Entry points:
- ``run_sync(payload, sampling)`` — full loop, returns [DiffusionOutput]
- ``run(payload, sampling)`` — run_sync + first image/artifact, for PipelineRunner

``num_steps`` is resolved in one place (``_resolve_num_steps``):
state.metadata wins (the pipeline already folded payload/sampling
into it during prepare_encode), then payload attribute, then
sampling extras. One definition, no second loop elsewhere.

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` §2.3.
"""

from __future__ import annotations

from typing import Any

from .interface import DiffusionOutput, DiffusionPipeline, StepState


def _resolve_num_steps(payload: Any, sampling: Any, state: StepState) -> int:
    """Single definition of the num_steps priority.

    state.metadata wins: the pipeline's prepare_encode already folded
    payload + sampling into it. Payload attribute and sampling extras
    are fallbacks for pipelines that don't.
    """
    metadata = getattr(state, "metadata", None) or {}
    if "num_steps" in metadata:
        return int(metadata["num_steps"])
    if hasattr(payload, "num_inference_steps"):
        return int(payload.num_inference_steps)
    if sampling is not None and getattr(sampling, "extra", None):
        extra_steps = int(sampling.extra.get("num_inference_steps", 0))
        if extra_steps > 0:
            return extra_steps
    return 1


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

    def run(self, payload: Any, sampling: Any = None) -> Any:
        """Run to completion; return the first image/artifact (or None).

        The shape PipelineRunner needs. Synchronous single-process call —
        no thread pool (the old InlineDiffusionClient built one and never
        used it).
        """
        outputs = self.run_sync(payload, sampling)
        if outputs and outputs[0].images:
            return outputs[0].images[0]
        return None

    def run_sync(self, payload: Any, sampling: Any = None) -> list[DiffusionOutput]:
        """Run the full denoise loop synchronously. Returns [DiffusionOutput]."""
        state = self.prepare(payload, sampling)
        num_steps = _resolve_num_steps(payload, sampling, state)
        for step in range(num_steps):
            noise_pred = self.denoise_step(state, step=step, num_steps=num_steps)
            self.step_scheduler(state, noise_pred)
        return [self.post_decode(state)]


__all__ = ["DiffusionRunner"]
