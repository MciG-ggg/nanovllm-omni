"""The MLX backend: the Apple Silicon track's adapter.

This file holds the backend object only -- the device, the dtype, the
capabilities, and the two factories the stage calls. The runner it builds
lives in ``mlx_runner``; the shared seam is in ``base``.

``mlx`` is imported inside the methods, so importing this package pulls in
neither torch nor mlx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities
from .mlx_runner import MlxConfig, MlxStageRunner


@dataclass(frozen=True)
class MlxBackend:
    """MLX on Apple Silicon."""

    device: str | None = None
    """The MLX device; ``None`` means the GPU (``mx.gpu``)."""

    dtype: str | None = None
    """The MLX dtype; ``None`` means float16, the dtype the weights ship in."""

    name: str = "mlx"

    def __post_init__(self) -> None:
        if self.device is None:
            object.__setattr__(self, "device", "gpu")
        if self.dtype is None:
            object.__setattr__(self, "dtype", "float16")

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            varlen=False,
            graph=True,
            max_batch=None,
        )

    def make_config(self, model_path: str, **kwargs: Any) -> MlxConfig:
        # These shared stage kwargs belong to CUDA; MLX uses one device,
        # loads a converted model, and does not capture CUDA graphs.
        kwargs.pop("gpu_memory_utilization", None)
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("max_capture_bs", None)
        config = MlxConfig(
            model_path=model_path,
            dtype=kwargs.pop("dtype", None) or self.dtype,
            device=kwargs.pop("device", None) or self.device,
            kvcache_block_size=kwargs.pop("kvcache_block_size", None) or 256,
            num_kvcache_blocks=kwargs.pop("num_kvcache_blocks", None) or 64,
            max_num_seqs=kwargs.pop("max_num_seqs", None) or 512,
            max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", None) or 16384,
            eos=kwargs.pop("eos", None),
        )
        if kwargs:
            raise ValueError(f"unsupported MLX config kwargs: {', '.join(sorted(kwargs))}")
        return config

    def make_stage_runner(self, model_class: type, config: Any) -> MlxStageRunner:
        return MlxStageRunner(model_class=model_class, config=config)


__all__ = [
    "MlxBackend",
    "MlxConfig",
    "MlxStageRunner",
]
