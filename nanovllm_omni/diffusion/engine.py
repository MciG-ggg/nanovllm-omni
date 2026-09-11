"""DiffusionEngine: orchestrates diffusion inference via DiffusionRunner.

Two entry points (ADR-024):
- ``run_sync(payload, sampling)`` — synchronous, used by ``PipelineRunner.run``
- ``step_streaming(payload, sampling)`` — async generator, for future SSE / streaming

Both share a private ``_denoise_loop`` method.

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §3 ADR-024.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from .runner import DiffusionRunner
from .scheduler import RequestScheduler


class DiffusionEngine:
    """Orchestrates diffusion inference through scheduler + runner.

    STEP_BATCH only (ADR-021).  No REQUEST_BATCH, no multi-GPU.

    ponytail: step-only; add REQUEST_BATCH when max_num_seqs > 1
    becomes a real requirement.
    """

    def __init__(self, runner: DiffusionRunner) -> None:
        self._runner = runner
        self._scheduler = RequestScheduler()

    def run_sync(self, payload: Any, sampling: Any = None) -> list:
        """Run one request to completion synchronously.

        Accepts the raw payload + sampling directly (no OmniDiffusionRequest
        conversion) so the pipeline receives the full payload with all
        metadata (KV cache, masks, etc.).
        """
        return list(self._denoise_loop(payload, sampling))

    async def step_streaming(
        self,
        payload: Any,
        sampling: Any = None,
    ) -> AsyncGenerator[list, None]:
        """Async generator: yield partial results each denoise step."""
        for output in self._denoise_loop(payload, sampling):
            yield [output]

    def _denoise_loop(self, payload: Any, sampling: Any = None):
        """Core denoise loop shared by run_sync and step_streaming.

        Yields DiffusionOutput after each completed request.
        """
        state = self._runner.prepare(payload, sampling)
        self._scheduler.add_request(state)

        # Determine num_steps: state metadata (pipeline config) wins over
        # sampling extras (user override) and payload defaults.
        num_steps = 1
        if hasattr(payload, "num_inference_steps"):
            num_steps = payload.num_inference_steps
        if sampling is not None and getattr(sampling, "extra", None):
            ei = int(sampling.extra.get("num_inference_steps", 0))
            if ei > 0:
                num_steps = ei
        if hasattr(state, "metadata") and "num_steps" in state.metadata:
            num_steps = state.metadata["num_steps"]  # pipeline config wins

        while self._scheduler.has_requests():
            sched = self._scheduler.schedule()
            if sched is None:
                break
            step_id = sched.step_id
            noise_pred = self._runner.denoise_step(
                state,
                step=step_id,
                num_steps=num_steps,
            )
            self._runner.step_scheduler(state, noise_pred)

            if step_id + 1 >= num_steps:
                output = self._runner.post_decode(state)
                self._scheduler.finish(sched.request_id)
                yield output
                return

            self._scheduler.update_from_output(sched, None)


__all__ = ["DiffusionEngine"]
