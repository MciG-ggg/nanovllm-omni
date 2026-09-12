"""Diffusion: single-process diffusion inference over a DiffusionPipeline.

``DiffusionRunner`` owns one ``DiffusionPipeline`` (the 4-method
contract: prepare_encode / denoise_step / step_scheduler / post_decode)
and drives the denoise loop. Only STEP_BATCH mode; no IPC, no
multi-GPU, no paged KV.

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` for the full plan.
"""

from .interface import DiffusionOutput, DiffusionPipeline, StepState
from .runner import DiffusionRunner

__all__ = [
    "DiffusionOutput",
    "DiffusionPipeline",
    "DiffusionRunner",
    "StepState",
]
