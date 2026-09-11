"""SD-Turbo diffusion stage factory consumed by PipelineRunner.

Loads individual SD-Turbo components (tokenizer, text_encoder, UNet,
VAE, scheduler) and returns a ``SdTurboPipeline`` object that implements
the ``DiffusionPipeline`` 4-method protocol (prepare_encode / denoise_step
/ step_scheduler / post_decode).

``DiffusionEngine`` drives the denoise loop through these 4 methods;
the stage factory no longer calls ``pipe(...)`` directly.

SD-Turbo specifics (verified against the Stability blog + diffusers docs):

- Adversarial diffusion distillation of SD 2.1: 1-4 inference steps is
  enough; more steps does not improve quality.
- ``guidance_scale`` must be ``0.0`` (or close to it). Any positive CFG
  breaks the distillation and degrades the image.
- fp16 is the official precision; bf16 is not part of Stability's recipe.
- Native resolution is 512x512.

Call contract: ``payload`` is the prompt (str / {"prompt": ...}); the
deploy YAML defaults arrive in ``sampling.extra``.

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` for the
full migration plan.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "stabilityai/sd-turbo"
_REGISTERED_HANDLES = {"sd_turbo"}
_TORCH_DTYPES = {"float16"}


class SdTurboPipeline:
    """SD-Turbo pipeline implementing the DiffusionPipeline 4-method contract.

    ponytail: guidance_scale=0.0 means no CFG — no uncond embedding,
    no latent duplication.  Simpler and faster than standard SD.
    """

    supports_step_execution = True

    def __init__(
        self,
        tokenizer: Any,
        text_encoder: Any,
        unet: Any,
        vae: Any,
        scheduler: Any,
        target_device: str,
        torch_dtype: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.target_device = target_device
        self.torch_dtype = torch_dtype
        self.vae_scaling = 0.18215  # SD v1/v2 standard

    def prepare_encode(self, request: Any, sampling: Any = None) -> Any:
        """Text → text embeddings + initial latents.

        ``request`` is the payload (prompt str or object with prompt/height/
        width/num_inference_steps attrs); ``sampling.extra`` is the fallback
        for those fields when the payload is a bare prompt string (the
        PipelineRunner path after OmniDiffusionRequest was removed).
        """
        import torch

        from nanovllm_omni.diffusion.interface import StepState

        extra = dict(getattr(sampling, "extra", None) or {})

        prompt = getattr(request, "prompt", None)
        if prompt is None:
            prompt = extra.get("prompt", str(request))

        # Tokenize + encode text.
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.target_device))[0]

        # Initial random latents.
        height = int(getattr(request, "height", extra.get("height", 512)))
        width = int(getattr(request, "width", extra.get("width", 512)))
        latent_shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(
            latent_shape,
            device=self.target_device,
            dtype=self.torch_dtype,
        )

        # Scheduler setup.
        num_steps = int(
            getattr(request, "num_inference_steps", extra.get("num_inference_steps", 1))
        )
        self.scheduler.set_timesteps(num_steps)
        latents = latents * self.scheduler.init_noise_sigma

        return StepState(
            request_id=getattr(request, "request_id", "sync"),
            latents=latents,
            encoder_hidden_states=text_embeddings,
            metadata={"num_steps": num_steps},
        )

    def denoise_step(self, state: Any, *, step: int, num_steps: int) -> Any:
        """Run UNet noise prediction for one step."""
        import torch

        with torch.inference_mode():
            latent_input = self.scheduler.scale_model_input(
                state.latents,
                self.scheduler.timesteps[step],
            )
            noise_pred = self.unet(
                latent_input,
                self.scheduler.timesteps[step],
                encoder_hidden_states=state.encoder_hidden_states,
            ).sample
        return noise_pred

    def step_scheduler(self, state: Any, noise_pred: Any) -> None:
        """Euler scheduler step: update latents."""
        state.latents = self.scheduler.step(
            noise_pred,
            self.scheduler.timesteps[state.step_index],
            state.latents,
        ).prev_sample
        state.step_index += 1

    def post_decode(self, state: Any) -> Any:
        """VAE decode → PIL Image."""
        import torch
        from PIL import Image

        from nanovllm_omni.diffusion.interface import DiffusionOutput

        with torch.inference_mode():
            latents = state.latents / self.vae_scaling
            image = self.vae.decode(latents).sample

        # To PIL.
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)
        return DiffusionOutput(images=[pil_image], finished=True)


def _sd_turbo_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SD-Turbo components, return SdTurboPipeline."""
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

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

    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if not allow_hf:
        load_kwargs["local_files_only"] = True
    variant = "fp16" if not allow_hf else None

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **load_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        variant=variant,
        **load_kwargs,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        variant=variant,
        **load_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        variant=variant,
        **load_kwargs,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    target_device = device or "cuda"
    text_encoder = text_encoder.to(target_device)
    unet = unet.to(target_device)
    vae = vae.to(target_device)

    return SdTurboPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        target_device=target_device,
        torch_dtype=torch_dtype,
    )


__all__ = ["SdTurboPipeline", "_sd_turbo_stage"]
