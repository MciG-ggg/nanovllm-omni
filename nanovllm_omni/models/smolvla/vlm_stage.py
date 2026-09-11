"""SmolVLA vlm (AR) stage: SigLIP vision + SmolVLM language backbone.

Outputs ``VlmStageOutput`` with backbone hidden states for the downstream
action stage.  Currently uses HF SmolVLM as backbone (ADR-029 planned
fork migration deferred).

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §3 ADR-029.
"""

from __future__ import annotations

from typing import Any

from .stage_processors import VlmStageOutput


def _vlm_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SmolVLM backbone, return SmolVLAVlmStage."""
    return SmolVLAVlmStage(deploy, args)


class SmolVLAVlmStage:
    """AR stage: encode observation + instruction → backbone hidden states."""

    def __init__(self, deploy: Any, args: Any) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoModelCls
        except ImportError:
            from transformers import (
                AutoModelForVision2Seq as AutoModelCls,  # type: ignore[no-redef]
            )

        extra = dict(getattr(args, "extra", None) or {})
        allow_hf = bool(extra.get("allow_hf_download", False))
        self.device = getattr(args, "device", None) or "cuda"
        dtype = getattr(args, "dtype", None) or "bfloat16"

        model_path = getattr(args, "model", None) or "HuggingFaceTB/SmolVLM-500M-Instruct"
        # For SmolVLA, the backbone is SmolVLM; extract from the lerobot checkpoint
        # or use the standalone SmolVLM if model is the backbone path.
        torch_dtype = getattr(torch, dtype)
        load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if not allow_hf:
            load_kwargs["local_files_only"] = True

        self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
        self.model = AutoModelCls.from_pretrained(model_path, **load_kwargs)
        self.model.config.pad_token_id = None
        self.model = self.model.to(self.device)

    def __call__(self, payload: Any, sampling: Any) -> VlmStageOutput:
        import torch

        # Extract instruction and images from payload/sampling.
        if isinstance(payload, str):
            instruction = payload
        elif isinstance(payload, dict):
            instruction = str(payload.get("prompt", ""))
        else:
            instruction = str(payload)

        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )

        # Build observation batch (images + state).
        images = []
        img = extras.get("image") or extras.get("observation.images.image")
        if img is not None:
            images.append(img)
        wrist = extras.get("wrist_image") or extras.get("observation.images.image2")
        if wrist is not None:
            images.append(wrist)

        robot_state = extras.get("state")
        if robot_state is None:
            robot_state = extras.get("observation.state")

        # Build chat template input for SmolVLM.
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=images or None, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # Run SmolVLM to get hidden states (not generate, just forward).
        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True)
            # Use the last hidden layer as backbone output.
            hidden_states = out.hidden_states[-1]  # [1, T, H]

        # Process robot state.
        import numpy as np

        if robot_state is not None:
            state_tensor = torch.from_numpy(
                np.asarray(robot_state, dtype=np.float32).reshape(1, -1)
            ).to(self.device)
        else:
            state_tensor = torch.zeros(1, 7, device=self.device)  # default 7-DoF

        return VlmStageOutput(
            prefix_states=hidden_states.squeeze(0),  # [T, H]
            robot_state=state_tensor.squeeze(0),  # [state_dim]
            attention_mask=inputs.get("attention_mask"),
            request_id=extras.get("request_id"),
            metadata={
                "chunk_len": extras.get("chunk_len", 10),
                "action_dim": extras.get("action_dim", 7),
            },
        )


__all__ = ["SmolVLAVlmStage", "_vlm_stage"]
