"""SD-Turbo diffusion stage factory consumed by PipelineRunner.

Mirrors ``sana_06b._sana_stage``: the factory closes over the loaded
``StableDiffusionPipeline`` (SD-Turbo is a regular SD pipeline in
diffusers; only the weights differ) and returns ``forward(payload, sampling)``.
Heavy deps (diffusers / torch) stay inside the factory and forward so
importing the pipeline registry does not require them.

SD-Turbo specifics (verified against the Stability blog + diffusers docs):

- Adversarial diffusion distillation of SD 2.1: 1-4 inference steps is
  enough; more steps does not improve quality.
- ``guidance_scale`` must be ``0.0`` (or close to it). Any positive CFG
  breaks the distillation and degrades the image. We lock the default to
  ``0.0``; callers can override via ``SamplingParams.extra["guidance_scale"]``.
- fp16 is the official precision; bf16 is not part of Stability's recipe.
- Native resolution is 512x512; ``height`` / ``width`` are exposed for
  convenience but 512x512 is what the weights were trained for.

Call contract: ``payload`` is the prompt (str / {"prompt": ...}); the
deploy YAML defaults ``num_inference_steps`` / ``guidance_scale`` /
``height`` / ``width`` arrive in ``sampling.extra`` via
``PipelineRunner._stage_sampling`` and a caller-provided
``SamplingParams.extra`` override wins.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "stabilityai/sd-turbo"
# ``args.model`` is a registry handle unless the caller passed a real
# repo id / local snapshot dir; only the bare handle maps to the default.
_REGISTERED_HANDLES = {"sd_turbo"}
# SD-Turbo is fp16-native. Other dtypes are rejected at factory time
# rather than silently producing a worse image downstream.
_TORCH_DTYPES = {"float16"}


def _sd_turbo_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load the SD-Turbo pipeline, return a forward callable."""
    import torch
    from diffusers import StableDiffusionPipeline

    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None)
    dtype = getattr(args, "dtype", None)
    if dtype is not None and dtype not in _TORCH_DTYPES:
        raise ValueError(
            f"sd_turbo stage: dtype {dtype!r} not in supported "
            f"{sorted(_TORCH_DTYPES)}; SD-Turbo is fp16-only by recipe"
        )
    torch_dtype = getattr(torch, dtype if dtype in _TORCH_DTYPES else "float16", torch.float16)
    model = getattr(args, "model", None) or _MODEL_ID
    model_id = _MODEL_ID if model in _REGISTERED_HANDLES else model

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "variant": "fp16"}
    if not allow_hf:
        kwargs["local_files_only"] = True
    pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    if device:
        pipe = pipe.to(device)

    def sd_turbo_forward(payload: Any, sampling: Any) -> Any:
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict):
            prompt = str(payload.get("prompt", ""))
        else:
            prompt = str(payload)
        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )
        steps = int(extras.get("num_inference_steps", 1))
        # Default guidance_scale is locked at 0.0 — SD-Turbo's adversarial
        # distillation breaks with any positive CFG. Callers MAY override
        # in extras (e.g. for sanity checks), but the deploy YAML enforces 0.0.
        guidance = float(extras.get("guidance_scale", 0.0))
        height = int(extras.get("height", 512))
        width = int(extras.get("width", 512))
        with torch.inference_mode():
            return pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
            ).images[0]

    return sd_turbo_forward


__all__ = ["_sd_turbo_stage"]
