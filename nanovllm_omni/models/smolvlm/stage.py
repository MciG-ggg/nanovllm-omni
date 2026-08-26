"""SmolVLM-500M-Instruct VLM stage factory consumed by PipelineRunner.

Mirrors ``sd_turbo._sd_turbo_stage``: the factory closes over the loaded
``AutoModelForVision2Seq`` + ``AutoProcessor`` and returns
``forward(payload, sampling)``. Heavy deps (transformers / torch) stay
inside the factory and forward so the registry imports cleanly without
them.

SmolVLM specifics (per HuggingFaceTB model card + transformers docs):

- 500M params; BF16 weights (~1 GB). Fits comfortably in 4 GB VRAM.
- Input is one or more PIL images plus a text prompt; the chat template
  accepts both via ``processor.apply_chat_template``.
- Generation is text-only via ``model.generate(...)``; we slice off the
  prompt tokens and decode the new ones (``skip_special_tokens=True``).
- Default ``local_files_only=True``; opt-in Hub download via
  ``OmniEngineArgs.extra["allow_hf_download"]``.

Call contract: ``payload`` is the text prompt (str / {"prompt": ...});
images are passed via ``SamplingParams.extra["images"]`` as a list of
``PIL.Image.Image`` (the registry contract is text-prompt-shaped; the
image list rides in ``sampling.extra`` so the runner shape stays the
same as every other family). The example ``run.py`` opens the image on
the caller side and stuffs it in extras.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
_REGISTERED_HANDLES = {"smolvlm"}
# SmolVLM was released in BF16; the recipe is BF16. fp16 is acceptable
# but loses numerical headroom on the SigLIP vision encoder.
_DTYPE_ALLOWED = ("bfloat16", "float16", "float32")


def _vlm_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SmolVLM-500M-Instruct, return forward callable."""
    import torch

    # AutoModelForVision2Seq was renamed to AutoModelForImageTextToText in
    # transformers 4.56+. We try the new name first (canonical) and fall
    # back to the legacy one for older installs; both resolve to
    # ``SmolVLMForConditionalGeneration`` for the SmolVLM model_type.
    try:
        from transformers import AutoModelForImageTextToText as _AutoModelCls
    except ImportError:  # pragma: no cover - very old transformers
        from transformers import AutoModelForVision2Seq as _AutoModelCls  # type: ignore[no-redef]
    from transformers import AutoProcessor

    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None)
    dtype = getattr(args, "dtype", None) or "bfloat16"
    if dtype not in _DTYPE_ALLOWED:
        raise ValueError(
            f"smolvlm stage: dtype {dtype!r} not in supported "
            f"{list(_DTYPE_ALLOWED)}; SmolVLM-500M-Instruct ships BF16"
        )
    torch_dtype = getattr(torch, dtype)
    model = getattr(args, "model", None) or _MODEL_ID
    model_id = _MODEL_ID if model in _REGISTERED_HANDLES else model

    kwargs: dict[str, Any] = {"dtype": torch_dtype}
    if not allow_hf:
        kwargs["local_files_only"] = True
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    model = _AutoModelCls.from_pretrained(model_id, **kwargs)
    # SmolVLM's tokenizer pad id (128002) sits outside the LM vocab range
    # and emits a warning on first generate; resetting to None is harmless.
    model.config.pad_token_id = None
    if device:
        model = model.to(device)

    def vlm_forward(payload: Any, sampling: Any) -> Any:
        # prompt: str or {"prompt": ...} per the OmniPromptType contract
        prompt_text = (
            payload.get("prompt", "")
            if isinstance(payload, dict)
            else (payload if isinstance(payload, str) else str(payload))
        )
        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )
        # Images are a list of PIL.Image; empty list is allowed (text-only VQA).
        images = list(extras.get("images") or [])
        max_new_tokens = int(extras.get("max_new_tokens", 512))

        # Build the chat-template input. SmolVLM expects image blocks before
        # the text block; an empty list means text-only.
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat, images=images or None, return_tensors="pt")
        target_device = device if device else next(model.parameters()).device
        inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return vlm_forward


__all__ = ["_vlm_stage"]
