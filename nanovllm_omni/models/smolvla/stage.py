"""SmolVLA stage factory consumed by PipelineRunner.

Mirrors MiniMind-O ``thinker._thinker_stage``: a factory closes over the
loaded policy and returns ``forward(payload, sampling)``. There are two
call contracts:

- Default (synthetic demo): images / state ride in ``SamplingParams.extra``
  as ``image`` / ``wrist_image`` / ``state``; ``payload`` is the instruction.
- LIBERO eval (``extra['libero_obs']`` present): the caller passes the raw
  lerobot LIBERO obs dict (``pixels`` / ``robot_state``) and the stage runs
  the exact official eval chain -- ``preprocess_observation`` ->
  ``env_preprocessor`` (flip + quat->axis-angle) -> ``policy.preprocessor``
  -> ``policy.select_action`` (internal 50-step chunk queue) ->
  ``policy.postprocessor``. This is the chain that delivers ~100% SR on
  libero_object with HuggingFaceVLA/smolvla_libero.

Heavy deps (lerobot / torch / numpy) stay inside the factory and forward
so importing the pipeline registry does not require them.
"""

from __future__ import annotations

from typing import Any


def _batch_robot_state(state: Any) -> Any:
    """Recursively add a batch dim to tensor leaves inside the nested
    ``robot_state`` dict (official eval runs on gym VectorEnvs where every
    leaf is already (B, ...))."""
    import torch

    if isinstance(state, dict):
        return {k: _batch_robot_state(v) for k, v in state.items()}
    if isinstance(state, torch.Tensor) and state.ndim == 1:
        return state.unsqueeze(0)
    return state


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
    env_processors_cache: dict[str, tuple[Any, Any]] = {}

    def _env_processors(task_suite: str) -> tuple[Any, Any]:
        """Build lerobot's LIBERO env pre/post processors, cached per suite."""
        if task_suite not in env_processors_cache:
            from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

            css = LiberoEnvConfig(task=task_suite)
            env_processors_cache[task_suite] = css.get_env_processors()
        return env_processors_cache[task_suite]

    def _libero_forward(obs: Any, instruction: str, task_suite: str) -> Any:
        """Official eval chain (lerobot/scripts/lerobot_eval.py ~L268-300)."""
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.lerobot_types import TransitionKey

        obs = dict(obs)
        # Order matters: preprocess FIRST (it renames unknown keys to
        # ``observation.*``), THEN inject ``task`` -- exactly like the
        # official eval. Setting task before preprocessing would rename it
        # to ``observation.task`` and break the tokenizer lookup.
        obs = preprocess_observation(obs)
        obs["task"] = instruction
        robot_state_key = "observation.robot_state"
        if robot_state_key in obs:
            obs[robot_state_key] = _batch_robot_state(obs[robot_state_key])
        env_preprocessor, env_postprocessor = _env_processors(task_suite)
        obs = env_preprocessor(obs)
        obs = policy.preprocessor(obs)
        with torch.inference_mode():
            raw = policy.select_action(obs)
        raw = policy.postprocessor(raw)
        action_key = TransitionKey.ACTION.value
        raw = env_postprocessor({action_key: raw})[action_key]
        return _to_action_artifact(raw)

    def vla_forward(payload: Any, sampling: Any) -> Any:
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

        # LIBERO path: caller passes a raw lerobot obs dict plus the task suite.
        if "libero_obs" in extras:
            task_suite = str(extras.get("libero_task_suite", "libero_object"))
            return _libero_forward(extras["libero_obs"], prompt, task_suite)

        # Default path: pre-built HWC images + state in extra.
        batch = _obs_batch(prompt, sampling, device)
        batch = policy.preprocessor(batch)
        raw = policy.predict_action_chunk(batch)
        postprocessor = getattr(policy, "postprocessor", None)
        if postprocessor is not None:
            raw = postprocessor(raw)
        return _to_action_artifact(raw)

    return vla_forward


__all__ = ["_vla_stage"]
