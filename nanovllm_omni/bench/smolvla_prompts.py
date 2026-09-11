"""Profile input set for smolvla (1 input x 20 runs, tiny 档 with synthetic data).

Synthetic observation (256x256 RGB + 7-dim zero state) for the Phase 1
profile -- goal is "where time goes in our two-stage pipeline", not
LIBERO data fidelity. Real LIBERO demo episode population is a
follow-up (see TODO in this file).

7-dim state is the SmolVLA action_dim per the policy config
(``action_feature.shape == [7]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SmolVLAInput:
    """One smolvla profile input (synthetic observation).

    ``image`` is a HxWx3 uint8 array (numpy); ``state`` is a 7-dim
    proprio vector; ``instruction`` is the high-level task description.
    ``num_inference_steps`` is the flow-matching step count (config
    default 10; profile can vary in Phase 2).
    """

    id: str
    image: Any
    state: tuple  # 7-dim proprio
    instruction: str
    num_inference_steps: int = 10


def _synthetic_image() -> Any:
    """256x256 RGB image: vertical gradient + mid-frequency noise.

    Synthetic enough that SigLIP/SmolVLM2 vision tower has real work
    to do (not all-zeros); simple enough to fit on a slide.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    grad = np.linspace(0, 255, 256, dtype=np.uint8)
    img = np.tile(grad, (256, 1)).T  # [256, 256] gradient
    img = np.stack([img, img // 2, 255 - img], axis=-1)  # RGB
    noise = rng.integers(0, 32, size=img.shape, dtype=np.uint8)
    return (img + noise).clip(0, 255)


SMOLVLA_INPUTS: tuple[SmolVLAInput, ...] = (
    SmolVLAInput(
        id="smolvla_01",
        image=_synthetic_image(),
        state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        instruction="pick up the red block and place it in the basket",
    ),
)
# TODO(populate): replace with 6 (state, instruction) pairs from a fixed
# LIBERO demo episode. Until then, the tuple has 1 input so the bench
# short-circuits cleanly through CPU fallback + 1 GPU row.
