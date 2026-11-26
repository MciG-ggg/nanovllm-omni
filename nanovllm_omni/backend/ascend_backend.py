"""The Ascend backend: ``torch_npu`` with contiguous-KV attention.

Torch is the same dialect on NPU, so this backend reuses the fork's
``ModelRunner`` instead of reimplementing step execution. It differs in
two places only: the runner gets ``device="npu"``, and the model class is
bound to ``AscendAttention`` through ``functools.partial`` -- no fork
attention file is modified (see ADR 0028).

Milestone 1 is one sequence, no prefix cache, no graph capture.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities


@dataclass(frozen=True)
class AscendBackend:
    """Ascend NPU through ``torch_npu``; thinker only."""

    device: str = "npu"
    dtype: str = "bfloat16"
    name: str = "ascend"

    def __post_init__(self) -> None:
        # The stage forwards its torch device ("cpu" on a CUDA-less host);
        # this backend only ever runs on the NPU.
        object.__setattr__(self, "device", "npu")

    @property
    def capabilities(self) -> BackendCapabilities:
        # npu_fusion_attention takes actual_seq_qlen/kvlen, so one forward
        # can carry ragged sequences once milestone 2 lands. Graph capture
        # is not planned yet.
        return BackendCapabilities(varlen=True, graph=False, max_batch=None)

    def make_config(self, model_path: str, **kwargs: Any) -> Any:
        """Build the fork ``Config`` for one stage."""
        from ..engine.stage_runner import get_stage_config

        kwargs.pop("device", None)
        kwargs["dtype"] = self.dtype
        config = get_stage_config(model_path, **kwargs)
        # The fork sizes paged KV from CUDA memory stats. NPU gets a
        # deterministic block count instead; AscendAttention ignores the
        # paged tensors and keeps its own contiguous cache.
        blocks_per_sequence = -(-config.max_model_len // config.kvcache_block_size)
        config.num_blocks = blocks_per_sequence * config.max_num_seqs
        return config

    def make_stage_runner(self, model_class: type, config: Any) -> Any:
        """Build the fork-backed runner with Ascend attention bound in."""
        import torch_npu  # noqa: F401

        from ..engine.stage_runner import StageRunner
        from .ascend_attention import AscendAttention

        # CUDA graph capture is CUDA-only; the stage defaults it on.
        config.enforce_eager = True
        bound = functools.partial(model_class, attention_cls=AscendAttention)
        return StageRunner(model_class=bound, config=config, device=self.device)


__all__ = ["AscendBackend"]
