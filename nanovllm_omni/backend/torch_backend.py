"""The torch backend: the fork's CUDA path, unchanged.

Every method here forwards to the fork's ``ModelRunner`` / ``Sampler``.
This file decides only what the backend *says* it can do -- ``varlen``
because the fork's attention is a variable-length kernel over a paged KV
cache, ``graph`` because the fork captures CUDA Graphs.

Selecting this backend must leave the pipeline byte-identical to the
pre-backend code path: the fork's config, sampler and KV pool are the
same objects, only reached through one more layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities


@dataclass(frozen=True)
class TorchBackend:
    """Torch + CUDA (or any device the fork accepts)."""

    device: str = "cuda"
    dtype: str = "float16"
    name: str = "torch"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            varlen=True,
            graph=True,
            max_batch=None,
        )

    def make_config(self, model_path: str, **kwargs: Any) -> Any:
        """Build the fork ``Config`` for one stage."""
        from ..engine.stage_runner import get_stage_config

        # Fork Config has no ``device`` field (its kernels are CUDA-only),
        # so a caller's device cannot travel into it.
        kwargs.pop("device", None)
        return get_stage_config(model_path, **kwargs)

    def make_stage_runner(self, model_class: type, config: Any) -> Any:
        """Build the fork-backed runner for one stage."""
        from ..engine.stage_runner import StageRunner

        return StageRunner(model_class=model_class, config=config)


__all__ = ["TorchBackend"]
