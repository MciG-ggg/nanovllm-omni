"""SmolVLA vlm (AR) stage: SigLIP vision + SmolVLM2 language backbone.

Loads the full lerobot SmolVLA policy, runs `embed_prefix` + `vlm_with_expert.forward`
to produce the VLM KV cache, and stores it in the bridge payload for the downstream
action stage. This is the "prefix phase" of flow matching.

Per ADR-029 (deferred): we use lerobot's internal `vlm_with_expert` directly
rather than weight-slicing into a fork `SmolVLMForCausalLM`. This keeps the
KV cache format identical to lerobot's, which is critical for bit-exact
alignment between the two-stage path and `policy.predict_action_chunk`.

See docs/dev/nanovllm-omni-smolvla-arflow-migration.md §3 ADR-029.
"""

from __future__ import annotations

from typing import Any

from .stage_processors import VlmStageOutput


def _vlm_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SmolVLA, return SmolVLAVlmStage."""
    return SmolVLAVlmStage(deploy, args)


class SmolVLAVlmStage:
    """AR stage: encode observation + instruction → KV cache for action expert.

    Internally loads the full lerobot SmolVLA policy (SigLIP + SmolVLM2 +
    action expert), but only runs the prefix phase. The action expert is
    owned by the downstream action stage.
    """

    def __init__(self, deploy: Any, args: Any) -> None:
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        extra = dict(getattr(args, "extra", None) or {})
        allow_hf = bool(extra.get("allow_hf_download", False))
        self.device = getattr(args, "device", None) or "cuda"

        # Load the full lerobot policy (vlm_stage needs vlm_with_expert + preprocessor).
        model_path = getattr(args, "model", None) or "HuggingFaceVLA/smolvla_libero"
        # If the model path looks like a backbone (not a lerobot policy), use default.
        if "smolvla" not in model_path.lower() and "lerobot" not in model_path.lower():
            model_path = "HuggingFaceVLA/smolvla_libero"
        # Share one policy instance with the action stage (same process,
        # same checkpoint → same cache key). Avoids a second ~2GB load on
        # the 4GB card and keeps sample_noise RNG state identical.
        from .action_stage import _POLICY_CACHE

        cache_key = model_path + ("" if allow_hf else "_offline")
        self.policy = _POLICY_CACHE.get(cache_key)
        if self.policy is None:
            load_kwargs: dict[str, Any] = {"strict": False}
            if not allow_hf:
                load_kwargs["local_files_only"] = True
            self.policy = SmolVLAPolicy.from_pretrained(model_path, **load_kwargs)
            if self.device:
                self.policy = self.policy.to(self.device)
            _POLICY_CACHE[cache_key] = self.policy

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

        # Build observation batch (single image + state, preprocessor normalizes).
        images_list = []
        img = extras.get("image")
        if img is None:
            img = extras.get("observation.images.image")
        if img is not None:
            images_list.append(img)
        wrist = extras.get("wrist_image") or extras.get("observation.images.image2")
        if wrist is not None:
            images_list.append(wrist)

        robot_state = extras.get("state")
        if robot_state is None:
            robot_state = extras.get("observation.state")

        # Convert to lerobot batch format.
        import numpy as np

        if img is not None:
            arr = np.array(img)
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
        else:
            raise ValueError("SmolVLA vlm_stage requires an image in extras['image']")

        batch = {"observation.images.image": img_tensor, "task": [instruction]}
        if robot_state is not None:
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
            prefix_states=prefix_pad_masks,  # used for shape only; KV cache is in metadata
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
