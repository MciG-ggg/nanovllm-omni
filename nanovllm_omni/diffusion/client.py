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

        Passes the payload directly to the pipeline (no conversion to
        OmniDiffusionRequest) so the pipeline receives the full payload
        with all metadata (KV cache, masks, etc.).
        """
        outputs = self._engine.run_sync(payload, sampling)
        if outputs and outputs[0].images:
            return outputs[0].images[0]
        return None

    def close(self) -> None:
        self._executor.shutdown(wait=False)


__all__ = ["InlineDiffusionClient"]
