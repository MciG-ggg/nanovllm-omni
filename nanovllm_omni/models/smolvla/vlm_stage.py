"""SmolVLA vlm (AR) stage: SigLIP vision + SmolVLM2 language backbone.

Loads the full lerobot SmolVLA policy, runs `embed_prefix` + `vlm_with_expert.forward`
to produce the VLM KV cache, and stores it in the bridge payload for the downstream
action stage. This is the "prefix phase" of flow matching.

Per ADR-029 (deferred): we use lerobot's internal `vlm_with_expert` directly
rather than weight-slicing into a fork `SmolVLMForCausalLM`. This keeps the
KV cache format identical to lerobot's, so the two-stage path and
`policy.predict_action_chunk` share the same ops; parity is checked by
`tests/test_parity_smolvla.py`.

See docs/dev/nanovllm-omni-smolvla-arflow-migration.md §3 ADR-029.
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.engine.stage_engine import StageEngine

from .stage_processors import VlmStageOutput


def _image_tensor(value: Any, device: str) -> Any:
    """PIL/numpy HWC image -> batched CHW tensor; keep uint8 for lerobot's /255."""
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.array(value)).permute(2, 0, 1).unsqueeze(0)
    if tensor.dtype != torch.uint8:
        tensor = tensor.float()
    return tensor.to(device)


def _vlm_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SmolVLA, return SmolVLAVlmStage."""
    return SmolVLAVlmStage(deploy, args)


class SmolVLAVlmStage(StageEngine):
    """AR stage: encode observation + instruction → KV cache for action expert.

    Internally loads the full lerobot SmolVLA policy (SigLIP + SmolVLM2 +
    action expert), but only runs the prefix phase. The action expert is
    owned by the downstream action stage.
    """

    def __init__(self, deploy: Any, args: Any) -> None:
        super().__init__()
        from lerobot.policies import make_pre_post_processors

        extra = dict(getattr(args, "extra", None) or {})
        self.device = getattr(args, "device", None) or "cuda"

        # Policy identity is fixed here at construction (see policy.py).
        # Same instance flows to the action stage via payload metadata.
        from .policy import get_policy, resolve_model_path

        model_path = resolve_model_path(args, extra)
        self.policy = get_policy(args, extra)

        # Cache references for speed.
        self.config = self.policy.config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=model_path,
        )

    def __call__(self, payload: Any, sampling: Any) -> VlmStageOutput:
        import torch
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        # Extract instruction from payload.
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

        # Build observation batch (one or two images + state, preprocessor normalizes).
        img = extras.get("image")
        if img is None:
            raise ValueError("SmolVLA vlm_stage requires an image in extras['image']")

        # Keep uint8 so lerobot's prepare_images applies its /255 branch; a
        # float [0,1] image is passed through unchanged.
        batch = {
            "observation.images.image": _image_tensor(img, self.device),
            "task": [instruction],
        }
        wrist = extras.get("wrist_image")
        if wrist is not None:
            # smolvla_libero declares this camera as observation.images.image2.
            batch["observation.images.image2"] = _image_tensor(wrist, self.device)
        robot_state = extras.get("state")
        if robot_state is not None:
            import numpy as np

            batch["observation.state"] = torch.tensor(
                np.asarray(robot_state, dtype=np.float32).reshape(1, -1),
            ).to(self.device)

        batch = self.preprocessor(batch)

        # === VLM prefix phase (same as lerobot's sample_actions) ===
        for k in batch:
            if k in self.policy._queues and k != "action":
                batch[k] = torch.stack(list(self.policy._queues[k]), dim=1)

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        # 1) Embed prefix (image + language + state embeddings).
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )

        # 2) VLM forward → KV cache.
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_kv = self.policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )

        return VlmStageOutput(
            robot_state=state.squeeze(0),
            attention_mask=prefix_att_masks,
            request_id=extras.get("request_id"),
            metadata={
                "past_key_values": past_kv,
                "prefix_pad_masks": prefix_pad_masks,
                "original_action_dim": self.config.action_feature.shape[0],
                "chunk_size": self.config.chunk_size,
                "max_action_dim": self.config.max_action_dim,
                "num_steps": self.config.num_steps,
                "use_cache": self.config.use_cache,
                "policy_ref": self.policy,  # action stage needs model.sample_noise + denoise_step
            },
        )


__all__ = ["SmolVLAVlmStage", "_vlm_stage"]
