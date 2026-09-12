"""SmolVLA policy ownership: one loader, one cache, one fallback rule.

Both stages need the same lerobot ``SmolVLAPolicy`` instance (same
checkpoint → same weights, same ``sample_noise`` RNG path). The policy
identity is fixed at construction time by this module — never smuggled
through payload metadata at runtime.

``get_policy(args)`` loads (or reuses) the policy for the model path in
``args``; ``action_stage`` calls the same function instead of keeping
its own loader + fallback heuristic.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_MODEL = "HuggingFaceVLA/smolvla_libero"

_POLICY_CACHE: dict[str, Any] = {}


def resolve_model_path(args: Any, extra: dict[str, Any] | None = None) -> str:
    """Single fallback rule: explicit path wins, backbone-looking paths reset.

    ``extra`` (deploy YAML ``action_model`` et al.) beats ``args.model``;
    anything that doesn't look like a lerobot policy falls back to default.
    """
    extra = extra or {}
    model_path = (
        extra.get("action_model")
        or extra.get("model")
        or getattr(args, "model", None)
        or _DEFAULT_MODEL
    )
    lowered = model_path.lower()
    if "smolvla" not in lowered and "lerobot" not in lowered:
        model_path = extra.get("action_model") or _DEFAULT_MODEL
    return model_path


def cache_key(model_path: str, allow_hf: bool) -> str:
    """Cache key: offline loads pin ``local_files_only`` and must not collide."""
    return model_path + ("" if allow_hf else "_offline")


def get_policy(args: Any, extra: dict[str, Any] | None = None) -> Any:
    """Return the shared policy for this model path (loads once per key).

    Raises the loader's error when lerobot is missing or the checkpoint
    is unreadable — no fake fallback here; test doubles are built by
    the caller (see ``action_stage._FakePolicy``).
    """
    extra = dict(extra or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    model_path = resolve_model_path(args, extra)
    key = cache_key(model_path, allow_hf)
    policy = _POLICY_CACHE.get(key)
    if policy is not None:
        return policy
    from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-not-found]
        SmolVLAPolicy,
    )

    load_kwargs: dict[str, Any] = {"strict": False}
    if not allow_hf:
        load_kwargs["local_files_only"] = True
    policy = SmolVLAPolicy.from_pretrained(model_path, **load_kwargs)
    device = getattr(args, "device", None) or "cuda"
    if device:
        policy = policy.to(device)
    _POLICY_CACHE[key] = policy
    return policy


__all__ = ["_POLICY_CACHE", "cache_key", "get_policy", "resolve_model_path"]
