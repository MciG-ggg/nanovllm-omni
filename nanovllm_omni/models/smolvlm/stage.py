"""SmolVLM stage class: vision prefill + fork StageRunner AR decode.

Per ``docs/dev/nanovllm-omni-smolvlm-ar-migration.md`` ADR-016 (revised,
2026-09) + ADR-017 + ADR-019:

- Vision tower (HF SiglipVisionModel) + connector (GeLU + Linear) run
  once on prefill to build merged text embeddings; image-token
  positions get vision features, the rest get text embeddings.
- Text decoder + sampler ride the fork ``StageRunner`` (paged KV +
  CUDA graph on the decode loop); mirrors minimind's
  ``decode_minimind`` scheduler pattern.
- Stage is a class instance returned by the factory
  ``_vlm_stage(deploy, args)``; the factory's dotted path stays the
  same so ``pipeline.py`` and deploy yaml need no edits.

Call contract: ``payload`` is text prompt (str / ``{"prompt": ...}``);
images are passed via ``SamplingParams.extra["images"]`` as a list of
``PIL.Image.Image``.
"""

from __future__ import annotations

from typing import Any

import torch

from .smolvlm import SmolVLMForConditionalGeneration

_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
_DTYPE_ALLOWED = ("bfloat16", "float16", "float32")


def _vlm_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: build + return ``SmolVLMStage`` instance."""
    return SmolVLMStage(deploy, args)


class SmolVLMStage:
    """Vision prefill + fork AR decode loop for SmolVLM-500M-Instruct."""

    def __init__(self, deploy: Any, args: Any) -> None:
        from transformers import AutoConfig, AutoProcessor

        from nanovllm_omni.engine.stage_runner import (
            StageRunner,
            _ensure_dist,
            get_stage_config,
            stage_kwargs_from_args,
        )

        # Fork layers (VocabParallelEmbedding, QKVParallelLinear, etc.)
        # read ``dist.get_rank()`` at construction time. The fork's
        # ModelRunner calls _ensure_dist() internally; SmolVLMModel is
        # built BEFORE ModelRunner so we must do the same init up front
        # (no-op after ModelRunner init: the monkey-patched init skips
        # duplicate calls).
        _ensure_dist()

        extra = dict(getattr(args, "extra", None) or {})
        allow_hf = bool(extra.get("allow_hf_download", False))
        self.device = getattr(args, "device", None) or "cuda"
        dtype_str = getattr(args, "dtype", None) or "bfloat16"
        if dtype_str not in _DTYPE_ALLOWED:
            raise ValueError(
                f"smolvlm: dtype {dtype_str!r} not in supported {list(_DTYPE_ALLOWED)}"
            )
        self.dtype = getattr(torch, dtype_str)
        model_id = getattr(args, "model", None) or _MODEL_ID

        kwargs: dict[str, Any] = {"dtype": self.dtype}
        if not allow_hf:
            kwargs["local_files_only"] = True

        # 1) Build our model from HF config
        hf_config = AutoConfig.from_pretrained(model_id, **kwargs)
        self.model = SmolVLMForConditionalGeneration(hf_config)

        # 2) Load HF weights via fork loader (default prefix=""; submodule
        # names mirror HF checkpoint keys, so weights land directly).
        from nanovllm.utils.loader import load_model

        try:
            load_model(self.model, model_id)
        except (FileNotFoundError, OSError):
            # Hub snapshot fallback when offline cache misses
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_id, local_files_only=not allow_hf)
            load_model(self.model, local_dir)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # 3) Build fork StageRunner wrapping the language_model-only forward
        stage_kwargs = stage_kwargs_from_args(args)
        # Fork Config.__post_init__ asserts os.path.isdir(self.model); resolve
        # Hub ids (HuggingFaceTB/SmolVLM-500M-Instruct) to local snapshots.
        from huggingface_hub import snapshot_download

        try:
            model_dir = snapshot_download(model_id, local_files_only=not allow_hf)
        except Exception:  # pragma: no cover - rely on load_model to fail loud
            model_dir = model_id
        config = get_stage_config("smolvlm_vlm", model_dir, **stage_kwargs)
        # fork Config is a frozen-ish dataclass; dtype is a process-wide
        # torch default, not a per-stage attribute. Set it for downstream
        # model construction (smolvlm.py weights load as bf16).
        self.stage_runner = StageRunner(
            model_class=SmolVLMForConditionalGeneration,
            config=config,
            rank=0,
        )

        # 4) Set process-wide torch default dtype before ModelRunner init
        torch.set_default_dtype(self.dtype)

        # 5) Processor + image-token metadata
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        image_token_id = getattr(hf_config, "image_token_id", None)
        if image_token_id is None:
            tok = self.processor.tokenizer
            try:
                image_token_id = tok.convert_tokens_to_ids("<image>")
            except KeyError:
                image_token_id = None
        self.image_token_id = image_token_id
        eos = getattr(hf_config, "eos_token_id", None) or self.processor.tokenizer.eos_token_id
        self.eos_token_id = eos if isinstance(eos, int) else int(eos[0])

    def _build_merged_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return [T, hidden] merged embeddings, or None for text-only."""
        if pixel_values is None:
            return None
        if self.image_token_id is None:
            raise RuntimeError(
                "smolvlm: image_token_id not resolved; "
                "SmolVLMConfig.image_token_id or tokenizer <image> token missing"
            )

        with torch.inference_mode():
            vision_out = self.model.model.vision_model(pixel_values=pixel_values)
            image_features = vision_out.last_hidden_state  # [B, P, v_dim]
            # Connector is single Linear; HF GeLU is folded here per
            # the connector's documented contract.
            image_embeds = torch.nn.functional.gelu(
                self.model.model.connector(image_features)
            )  # [B, P, t_dim]
            text_embeds = self.model.model.text_model.model.embed_tokens(input_ids)  # [1, T, t_dim]
            image_mask = input_ids == self.image_token_id  # [1, T]
            n_image_tokens = int(image_mask.sum().item())
            n_image_patches = int(image_embeds.shape[1])
            if n_image_tokens != n_image_patches:
                raise RuntimeError(
                    f"smolvlm: image-token positions {n_image_tokens} "
                    f"!= vision patches {n_image_patches}; check processor expansion"
                )
            merged = text_embeds.clone()
            merged[image_mask] = image_embeds[0]
        return merged.flatten(0, 1)  # [T, t_dim]

    def __call__(self, payload: Any, sampling: Any) -> str:
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

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
        images = list(extras.get("images") or [])
        max_new_tokens = int(extras.get("max_new_tokens", 512))
        temperature = float(getattr(sampling, "temperature", 0.0) or 0.0)

        # Build chat-template input; processor expands <image> placeholders
        # into the right number of patch tokens per image.
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=images or None, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        # Prefill: vision + connector + merge (or None for text-only)
        merged_embeds = self._build_merged_embeddings(input_ids, pixel_values)

        # Fork scheduler path — mirrors minimind decode_minimind shape.
        runner = self.stage_runner.model_runner
        config = self.stage_runner.config
        fork_sp = ForkSamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            ignore_eos=False,
        )
        sequence = Sequence(input_ids[0].tolist(), fork_sp)
        scheduler = Scheduler(config)
        scheduler.add(sequence)

        # Prefill step (one shot)
        seqs, is_prefill = scheduler.schedule()
        prefill_ids, prefill_positions = runner.prepare_prefill(seqs)
        temperatures = runner.prepare_sample(seqs)
        if merged_embeds is not None:
            logits = runner.run_model(
                prefill_ids,
                prefill_positions,
                is_prefill,
                inputs_embeds=merged_embeds,
            )
        else:
            logits = runner.run_model(prefill_ids, prefill_positions, is_prefill)
        token_ids = runner.sampler(logits, temperatures).tolist()
        scheduler.postprocess(seqs, token_ids, is_prefill)
        self.stage_runner.reset_context()
        generated: list[int] = list(token_ids)

        # AR decode loop
        while not scheduler.is_finished():
            seqs, is_prefill = scheduler.schedule()
            decode_ids, decode_positions = runner.prepare_decode(seqs)
            temperatures = runner.prepare_sample(seqs)
            logits = runner.run_model(decode_ids, decode_positions, is_prefill)
            token_ids = runner.sampler(logits, temperatures).tolist()
            scheduler.postprocess(seqs, token_ids, is_prefill)
            self.stage_runner.reset_context()
            generated.extend(token_ids)
            if generated[-1] == self.eos_token_id:
                break
            if sequence.num_completion_tokens >= max_new_tokens:
                break

        return self.processor.batch_decode([generated], skip_special_tokens=True)[0]


__all__ = ["SmolVLMStage", "_vlm_stage"]
