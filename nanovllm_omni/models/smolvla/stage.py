"""SmolVLA stage factory consumed by PipelineRunner.

Mirrors MiniMind-O ``thinker._thinker_stage``: a factory closes over the
loaded policy and returns ``forward(payload, sampling)``. Observation
images / state ride in ``SamplingParams.extra``; ``payload`` is the
language instruction (same ``Omni.generate(prompts, sampling_params)``
seam as MiniMind-O).

Heavy deps (lerobot / torch / numpy) stay inside the factory and forward
so importing the pipeline registry does not require them.
"""

from __future__ import annotations

from typing import Any


def _import_smolvla_policy() -> Any:
    try:
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-not-found]
            SmolVLAPolicy,
        )

        return SmolVLAPolicy
    except ImportError:
        pass
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import (  # type: ignore[import-not-found]
            SmolVLAPolicy,
        )

        return SmolVLAPolicy
    except ImportError as e:
        raise ImportError(
            "lerobot is required for SmolVLA. Install with: pip install -e '.[smolvla]'"
        ) from e


def _load_policy(args: Any) -> Any:
    cls = _import_smolvla_policy()
    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None)
    dtype = getattr(args, "dtype", None)
    kwargs: dict[str, Any] = {"strict": False}
    if not allow_hf:
        kwargs["local_files_only"] = True
    try:
        policy = cls.from_pretrained(args.model, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        kwargs.pop("strict", None)
        policy = cls.from_pretrained(args.model)
    if device:
        policy = policy.to(device)
    policy = _maybe_quantize(policy, dtype, device)
    policy.preprocessor, policy.postprocessor = _make_processors(policy, args.model)
    return policy


def _make_processors(policy: Any, model_path: str) -> tuple[Any, Any]:
    from lerobot.policies import make_pre_post_processors

    return make_pre_post_processors(policy.config, pretrained_path=model_path)


def _maybe_quantize(policy: Any, dtype: str | None, device: Any) -> Any:
    """int8 for the 4GB card.

    ponytail: torch's ``quantize_dynamic`` only runs on CPU. For CUDA int8
    we rely on bitsandbytes (HfTokenizer4bit etc.); if bitsandbytes is
    missing, leave the policy untouched -- the caller still has its original
    precision.
    """
    if dtype not in {"int8", "qint8"}:
        return policy
    import torch

    device_s = "" if device is None else str(device)
    if device_s.startswith("cuda"):
        try:
            import bitsandbytes as _bnb  # noqa: F401

            return policy
        except ImportError:
            return policy
    core = getattr(policy, "model", policy)
    quantized = torch.ao.quantization.quantize_dynamic(core, {torch.nn.Linear}, dtype=torch.qint8)
    if hasattr(policy, "model"):
        policy.model = quantized
        return policy
    return quantized


def _as_nchw(image: Any, device: Any) -> Any:
    import numpy as np
    import torch

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    tensor = tensor.float().div_(255.0) if tensor.dtype == torch.uint8 else tensor.float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    return tensor


def _as_state(state: Any, device: Any) -> Any:
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(1, -1))
    if device:
        tensor = tensor.to(device)
    return tensor


def _obs_batch(prompt: Any, sampling: Any, device: Any) -> dict[str, Any]:
    extra = (
        dict(sampling.extra) if sampling is not None and getattr(sampling, "extra", None) else {}
    )
    instruction = extra.get("task") or extra.get("instruction") or prompt
    if not isinstance(instruction, str):
        instruction = "" if instruction is None else str(instruction)
    image = extra.get("image")
    if image is None:
        image = extra.get("observation.images.image")
    if image is None:
        raise ValueError(
            "SmolVLA needs an image in SamplingParams.extra['image'] "
            "(HWC uint8 RGB). Wrist camera: extra['wrist_image']; "
            "proprio: extra['state']."
        )
    batch: dict[str, Any] = {
        "observation.images.image": _as_nchw(image, device),
        "task": [instruction],
    }
    wrist = extra.get("wrist_image")
    if wrist is None:
        wrist = extra.get("observation.images.image2")
    if wrist is not None:
        batch["observation.images.image2"] = _as_nchw(wrist, device)
    state = extra.get("state")
    if state is None:
        state = extra.get("observation.state")
    if state is not None:
        batch["observation.state"] = _as_state(state, device)
    return batch


def _to_action_artifact(raw: Any) -> Any:
    import numpy as np

    from nanovllm_omni.outputs import ActionArtifact

    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"SmolVLA action must be 2-D [chunk, dim], got shape={array.shape}")
    return ActionArtifact.from_array(array)


def _vla_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load the LeRobot policy, return a forward callable."""
    policy = _load_policy(args)
    device = getattr(args, "device", None) or getattr(policy, "device", None)

    def vla_forward(payload: Any, sampling: Any) -> Any:
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict):
            prompt = str(payload.get("prompt", ""))
        else:
            prompt = str(payload)
        batch = _obs_batch(prompt, sampling, device)
        batch = policy.preprocessor(batch)
        if hasattr(policy, "predict_action_chunk"):
            raw = policy.predict_action_chunk(batch)
        else:
            raw = policy.select_action(batch)
        return _to_action_artifact(raw)

    return vla_forward


__all__ = ["_vla_stage"]
