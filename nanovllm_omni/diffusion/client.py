"""InlineDiffusionClient: in-process diffusion client for PipelineRunner.

Single-process, single-GPU scope.  Wraps ``DiffusionEngine`` in a
``ThreadPoolExecutor`` so ``PipelineRunner.run`` (synchronous) can call
it without blocking the event loop.

Aligned with vllm-omni's ``InlineStageDiffusionClient``
(``vllm_omni/diffusion/inline_stage_diffusion_client.py``).
Simplified: no ZMQ, no output queue, no streaming wiring.

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §3 ADR-022.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .engine import DiffusionEngine
from .interface import DiffusionPipeline
from .request import OmniDiffusionRequest
from .runner import DiffusionRunner


class InlineDiffusionClient:
    """In-process diffusion client.

    ``stage_type`` is a class-level discriminant matching
    ``vllm_omni/diffusion/inline_stage_diffusion_client.py:38``.
    Used by ``PipelineRunner`` to dispatch to the right execution path.
    """

    stage_type: str = "diffusion"

    def __init__(self, pipeline: DiffusionPipeline) -> None:
        runner = DiffusionRunner(pipeline)
        self._engine = DiffusionEngine(runner)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="diffusion",
        )

    def run(self, payload: Any, sampling: Any) -> Any:
        """Synchronous entry point for PipelineRunner.run.

        Converts the ``(payload, sampling)`` pair into an
        ``OmniDiffusionRequest`` and runs the engine synchronously.
        """
        # Extract prompt from payload (str or {"prompt": ...}).
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict):
            prompt = str(payload.get("prompt", ""))
        else:
            prompt = str(payload)

        # Extract params from sampling.
        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )
        request = OmniDiffusionRequest(
            request_id="sync",
            prompt=prompt,
            sampling_params=sampling,
            num_inference_steps=int(extras.get("num_inference_steps", 1)),
            guidance_scale=float(extras.get("guidance_scale", 0.0)),
            height=int(extras.get("height", 512)),
            width=int(extras.get("width", 512)),
        )
        outputs = self._engine.run_sync(request)
        # Return the first output's images (PIL.Image list) for
        # OmniRequestOutput.from_diffusion compatibility.
        if outputs and outputs[0].images:
            return outputs[0].images[0]
        return None

    def close(self) -> None:
        self._executor.shutdown(wait=False)


__all__ = ["InlineDiffusionClient"]
