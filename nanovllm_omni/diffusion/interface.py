"""DiffusionPipeline Protocol: the 4-method contract every diffusion model must implement.

Aligned with vllm-omni's ``SupportsStepExecution`` Protocol
(``vllm_omni/diffusion/models/interface.py:49``). Method names are
alignment-boundary (AGENTS.md); signatures are simplified for teaching.

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` §2.3 for the skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass
class StepState:
    """Mutable state carried across denoise steps for one request.

    Created by ``DiffusionPipeline.prepare_encode``, mutated by
    ``denoise_step`` / ``step_scheduler``, consumed by ``post_decode``.
    """

    request_id: str
    latents: Any | None = None
    encoder_hidden_states: Any | None = None
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionOutput:
    """Result of one diffusion request.

    ``images`` is a list of PIL.Image (or numpy arrays); ``error`` is
    ``None`` on success.  ``finished`` signals the engine to remove this
    request from the scheduler.
    """

    images: list | None = None
    error: str | None = None
    finished: bool = True


@runtime_checkable
class DiffusionPipeline(Protocol):
    """Protocol that every diffusion model family must implement.

    Method names align with vllm-omni's ``SupportsStepExecution``:
    ``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` /
    ``post_decode``.  Signatures are simplified (single-request, no
    ``InputBatch`` / ``Sequence[StepRequestState]`` batching).
    """

    supports_step_execution: ClassVar[bool]

    def prepare_encode(self, request: Any) -> StepState:
        """Text / condition encoding → initial ``StepState`` with latents."""
        ...

    def denoise_step(self, state: StepState, *, step: int, num_steps: int) -> Any:
        """Run one denoise forward (UNet / action expert).  Return noise_pred or velocity."""
        ...

    def step_scheduler(self, state: StepState, noise_pred: Any) -> None:
        """Update ``state.latents`` via scheduler step (Euler / DPM / FlowMatch)."""
        ...

    def post_decode(self, state: StepState) -> DiffusionOutput:
        """VAE / vocoder decode → final ``DiffusionOutput``."""
        ...


__all__ = ["DiffusionOutput", "DiffusionPipeline", "StepState"]
