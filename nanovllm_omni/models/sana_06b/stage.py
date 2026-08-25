"""Sana-0.6B diffusion stage factory consumed by PipelineRunner.

Mirrors ``smolvla._vla_stage``: the factory closes over the loaded
``SanaPipeline`` and returns ``forward(payload, sampling)``. Heavy deps
(diffusers / torch) stay inside the factory and forward so importing the
pipeline registry does not require them.

Call contract: ``payload`` is the prompt (str / {"prompt": ...}); the
deploy YAML default ``num_inference_steps`` arrives in ``sampling.extra``
via ``PipelineRunner._stage_sampling`` and a caller-provided
``SamplingParams.extra`` override wins.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
# ``args.model`` is a registry handle unless the caller passed a real
# repo id / local snapshot dir; only the bare handle maps to the default.
_REGISTERED_HANDLES = {"sana_06b"}
# Only float dtypes are accepted; ``int8`` (used by SmolVLA's quantisation
# path) is rejected to avoid a bogus ``torch.int8`` cast for a diffusion stage.
_TORCH_DTYPES = {"float16", "bfloat16", "float32"}


def _import_sana_pipeline() -> Any:
    try:
        from diffusers import SanaPipeline

        return SanaPipeline
    except ImportError as e:
        raise ImportError(
            "diffusers is required for Sana-0.6B. Install with: "
            "pip install -U diffusers transformers accelerate safetensors"
        ) from e


def _sana_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load the SanaPipeline, return a forward callable.

    The snapshot must include ``transformer/diffusion_pytorch_model.fp16.safetensors``
    (and ideally the matching ``vae`` / ``text_encoder`` fp16 variants). The
    HF repo ships a 32-bit ``diffusion_pytorch_model.safetensors`` for the
    DiT that is the reference precision -- casting it to bf16 at load time
    produces denoised noise rather than a real image (verified empirically).
    ``variant="fp16"`` selects the inference weights; diffusers falls back
    per-component to the non-variant file when its fp16 sibling is absent
    (text encoder / VAE are tolerant; the DiT is not).
    """
    import torch
    from diffusers import SanaPipeline

    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None)
    dtype = getattr(args, "dtype", None)
    torch_dtype = getattr(torch, dtype if dtype in _TORCH_DTYPES else "bfloat16", torch.bfloat16)
    model = getattr(args, "model", None) or _MODEL_ID
    model_id = _MODEL_ID if model in _REGISTERED_HANDLES else model

    kwargs = {"torch_dtype": torch_dtype, "variant": "fp16"}
    if not allow_hf:
        kwargs["local_files_only"] = True
    pipe = SanaPipeline.from_pretrained(model_id, **kwargs)
    if device:
        pipe = pipe.to(device)

    def sana_forward(payload: Any, sampling: Any) -> Any:
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
        steps = int(extras.get("num_inference_steps", 20))
        with torch.inference_mode():
            return pipe(prompt, num_inference_steps=steps).images[0]

    return sana_forward


__all__ = ["_sana_stage"]
