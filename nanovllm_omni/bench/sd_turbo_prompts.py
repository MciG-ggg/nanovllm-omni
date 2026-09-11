"""Profile input set for sd-turbo (1 input x 5 runs, tiny).

Stage breakdown: tokenize (CLIP text encoder) / unet (1-step denoise) /
vae-decode (latent to image). Fixed seed and 1 inference step so the
profile isolates the unet + vae-decode cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SdTurboInput:
    """One sd-turbo profile input.

    ``seed`` / ``precision`` / ``num_inference_steps`` / ``guidance_scale``
    are part of the profile contract -- change them only via a new ADR.
    """

    id: str
    prompt: str
    seed: int = 42
    # ponytail: sd-turbo is fp16-only by recipe (adversarial distillation
    # needs the smaller mantissa to land in the right weight scale).
    # nanovllm_omni.models.sd_turbo.stage enforces this in _TORCH_DTYPES.
    precision: str = "float16"
    num_inference_steps: int = 1
    guidance_scale: float = 0.0


SD_TURBO_INPUTS: tuple[SdTurboInput, ...] = (
    SdTurboInput(
        id="sd_turbo_01",
        prompt="a corgi puppy sitting on a beach at sunset, oil painting style",
    ),
)
