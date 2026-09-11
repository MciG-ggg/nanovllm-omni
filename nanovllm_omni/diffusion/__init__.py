"""DiffusionEngine: independent diffusion inference engine.

Mirrors vllm-omni's ``vllm_omni/diffusion/`` as a teaching port.
Only STEP_BATCH mode; no IPC, no multi-GPU, no paged KV.

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` for the full plan.
"""

from .client import InlineDiffusionClient
from .engine import DiffusionEngine
from .interface import DiffusionOutput, DiffusionPipeline, StepState
from .request import OmniDiffusionRequest
from .runner import DiffusionRunner
from .scheduler import RequestScheduler

__all__ = [
    "DiffusionEngine",
    "DiffusionOutput",
    "DiffusionPipeline",
    "DiffusionRunner",
    "InlineDiffusionClient",
    "OmniDiffusionRequest",
    "RequestScheduler",
    "StepState",
]
