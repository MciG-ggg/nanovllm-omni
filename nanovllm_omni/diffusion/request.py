"""OmniDiffusionRequest: dataclass for a single diffusion inference request.

Aligned with vllm-omni's ``OmniDiffusionRequest``
(``vllm_omni/diffusion/request.py``).  Simplified: no ``kv_sender_info``
(single-process, no IPC).

See ``docs/dev/nanovllm-omni-diffusion-mirror.md`` §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OmniDiffusionRequest:
    """One diffusion request entering the engine."""

    request_id: str
    prompt: str
    sampling_params: Any = None
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    height: int = 512
    width: int = 512
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["OmniDiffusionRequest"]
