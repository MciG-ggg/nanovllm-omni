"""Backend registry: pick the array framework a stage runs on.

Three implementations exist -- ``torch`` (the fork's CUDA path), ``mlx``
(Apple Silicon) and ``ascend`` (Ascend NPU through ``torch_npu``). All
are imported lazily inside their own module, so importing this package
pulls in no framework.

The default is ``torch`` unless ``torch.npu`` is live, and selecting
``torch`` must leave the pipeline byte-identical to the pre-backend path.

This package holds the seam and nothing else: a backend answers "which
device, which dtype, which shapes can one forward carry". Model-family
code -- a family's own runner, its payload, its recipe -- lives in
``nanovllm_omni/models/<family>/`` and is reached from here only by
name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Backend, BackendCapabilities, StageRunner

_FACTORIES: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    """Register a backend factory under ``name``."""
    if name in _FACTORIES:
        raise ValueError(f"backend key {name!r} is already registered")
    _FACTORIES[name] = factory


def available_backends() -> list[str]:
    """Registered backend names."""
    return sorted(_FACTORIES)


def get_backend(name: str | None = None, **kwargs: Any) -> Backend:
    """Build the backend called ``name``; ``None`` means the default.

    Raises ``ValueError`` for an unknown name -- a silent fallback would
    run a Mac request on CUDA and fail much later, further from the cause.
    """
    if name is None:
        name = _default_backend_name()
    if name not in _FACTORIES:
        raise ValueError(f"unknown backend {name!r}; available: {available_backends()}")
    return _FACTORIES[name](**kwargs)


def _default_backend_name() -> str:
    """Pick ``ascend`` when ``torch.npu`` is live, ``torch`` otherwise."""
    try:
        import torch
    except ImportError:
        return "torch"
    try:
        # ``torch.npu`` exists only after torch_npu registers it.
        import torch_npu  # noqa: F401
    except ImportError:
        return "torch"
    if torch.npu.is_available():
        return "ascend"
    return "torch"


def _register_builtin_backends() -> None:
    from .ascend_backend import AscendBackend
    from .mlx_backend import MlxBackend
    from .torch_backend import TorchBackend

    register_backend("torch", TorchBackend)
    register_backend("mlx", MlxBackend)
    register_backend("mlx-vlm", _mlx_vlm_backend)
    register_backend("ascend", AscendBackend)


def _mlx_vlm_backend(**kwargs: Any) -> Backend:
    """The SmolVLM MLX track; it lives in the family, the registry only names it."""
    from ..models.smolvlm.vlm_mlx import MlxVlmBackend

    return MlxVlmBackend(**kwargs)


_register_builtin_backends()


__all__ = [
    "Backend",
    "BackendCapabilities",
    "StageRunner",
    "available_backends",
    "get_backend",
    "register_backend",
]
