"""Public synchronous submission operation for the model-free pipeline."""

from dataclasses import dataclass
from time import monotonic

from nanovllm_omni.payloads import AudioPayload
from nanovllm_omni.pipeline import Pipeline


@dataclass(frozen=True)
class PipelineResult:
    """Final output and observable stage order for one submitted request."""

    audio: AudioPayload
    stage_names: tuple[str, ...]
    elapsed_seconds: float


class Orchestrator:
    """Own request entrypoints; asynchronous scheduling is intentionally deferred."""

    def submit(self, pipeline: Pipeline, request: str) -> PipelineResult:
        """Run a request through ``pipeline`` and return its typed final output."""
        started = monotonic()
        audio, outputs = pipeline.run(request)
        return PipelineResult(
            audio=audio,
            stage_names=tuple(name for name, _ in outputs),
            elapsed_seconds=monotonic() - started,
        )
