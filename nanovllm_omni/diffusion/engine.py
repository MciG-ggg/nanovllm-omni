"""DiffusionEngine: orchestrates diffusion inference via DiffusionRunner.

Two entry points (ADR-024):
- ``run_sync(request)`` — synchronous, used by ``PipelineRunner.run``
- ``step_streaming(request)`` — async generator, for future SSE / streaming

Both share a private ``_denoise_loop`` method.

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §3 ADR-024.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from .request import OmniDiffusionRequest
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

    def run_sync(self, request: OmniDiffusionRequest) -> list:
        """Run one request to completion synchronously.

        Returns a list of ``OmniRequestOutput``-compatible dicts.
        This is the main path consumed by ``PipelineRunner.run``.
        """
        outputs = list(self._denoise_loop(request))
        return outputs

    async def step_streaming(
        self,
        request: OmniDiffusionRequest,
    ) -> AsyncGenerator[list, None]:
        """Async generator: yield partial results each denoise step.

        For future SSE / streaming HTTP.  Currently not wired to any
        consumer (ADR-024).
        """
        for output in self._denoise_loop(request):
            yield [output]

    def _denoise_loop(self, request: OmniDiffusionRequest):
        """Core denoise loop shared by run_sync and step_streaming.

        Yields DiffusionOutput after each completed request.
        """
        state = self._runner.prepare(request)
        self._scheduler.add_request(state)

        num_steps = request.num_inference_steps
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
