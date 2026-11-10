"""Backend registry: pick the array framework a stage runs on.

Two implementations exist -- ``torch`` (the fork's CUDA path) and ``mlx``
(Apple Silicon). Both are imported lazily inside their own module, so
importing this package pulls in neither torch nor mlx.

The default is ``torch``, and selecting it must leave the pipeline
byte-identical to the pre-backend code path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Backend, BackendCapabilities, StageRunner

_FACTORIES: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    """Register a backend factory under ``name``."""
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
        name = "torch"
    if name not in _FACTORIES:
        raise ValueError(f"unknown backend {name!r}; available: {available_backends()}")
    return _FACTORIES[name](**kwargs)


def _register_builtin_backends() -> None:
    from .mlx_backend import MlxBackend
    from .mlx_vlm import MlxVlmBackend
    from .torch_backend import TorchBackend

    register_backend("torch", TorchBackend)
    register_backend("mlx", MlxBackend)
    register_backend("mlx-vlm", MlxVlmBackend)


_register_builtin_backends()


__all__ = [
    "Backend",
    "BackendCapabilities",
    "StageRunner",
    "available_backends",
    "get_backend",
    "register_backend",
]
