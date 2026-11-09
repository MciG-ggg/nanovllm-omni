"""The MLX backend: Apple Silicon, one step at a time.

What MLX gives us that MPS does not: ``mx.compile`` (the graph-capture
equivalent CUDA Graph provides on CUDA -- measured 4.91x on a small
operator chain), and the same unified-memory pool MPS sees
(``max_recommended_working_set_size`` 11.84 GiB on a 16 GB M2 Air, the
same number ``torch.mps.recommended_max_memory()`` reports).

What MLX does not give us: a variable-length attention kernel.
``mx.fast.scaled_dot_product_attention`` takes a ``mask`` argument, so
dense masked attention works, but there is no ``cu_seqlens`` -- so a
batch of different lengths has to be padded. That is why this backend
reports ``varlen=False``. Upstream ``vllm-metal`` wrote a Metal kernel
for exactly this reason, and its own benchmark shows the padded path
costing +143% on a ragged batch.

``mlx_lm.generate.BatchGenerator`` ships an incremental batch scheduler
(``PromptProcessingBatch`` supports extending / filtering / splitting
batches). This backend does not use it: nanovllm-omni's own scheduler is
the thing being taught. It is the alternative worth comparing against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities


@dataclass
class MlxConfig:
    """Per-stage config for the MLX backend.

    The three graph knobs are CUDA-fork specific: MLX has no
    ``enforce_eager`` / graph-capture split, and ``mx.compile`` decides
    for itself. Stages write them unconditionally, so the fields have to
    exist.
    """

    model_path: str
    dtype: str = "float16"
    device: str = "gpu"
    enforce_eager: bool = True
    max_capture_bs: int = 1
    graph_max_seq_len: int = 2048


class MlxStageRunner:
    """Runs AR steps on an ``mlx_lm`` model.

    One sequence per forward: MLX has no variable-length attention, so
    batching would mean padding every member to the longest one and
    paying for the difference. The protocol allows it -- the interface
    does not change -- but the implementation does not yet.
    """

    def __init__(self, model_class: type, config: MlxConfig) -> None:
        import mlx.core as mx
        from mlx_lm.utils import load as mlx_load

        self._mx = mx
        self.config = config
        self._model, self.tokenizer = mlx_load(config.model_path)
        self._cache: Any = None
        self._is_prefill = False

    # -- batch preparation -------------------------------------------------

    def prepare_prefill(self, seqs: list[Any]) -> tuple[Any, Any]:
        """Token ids for one sequence's prompt, plus its positions."""
        if len(seqs) != 1:
            raise NotImplementedError("the MLX backend runs one sequence per forward")
        seq = seqs[0]
        ids = self._mx.array([seq.token_ids], dtype=self._mx.int32)
        positions = self._mx.arange(len(seq.token_ids), dtype=self._mx.int32)[None, :]
        return ids, positions

    def prepare_decode(self, seqs: list[Any]) -> tuple[Any, Any]:
        """The last token of one sequence, plus its position."""
        if len(seqs) != 1:
            raise NotImplementedError("the MLX backend runs one sequence per forward")
        seq = seqs[0]
        ids = self._mx.array([[seq.last_token]], dtype=self._mx.int32)
        positions = self._mx.array([[len(seq.token_ids) - 1]], dtype=self._mx.int32)
        return ids, positions

    def prepare_sample(self, seqs: list[Any]) -> Any:
        """Per-sequence temperatures."""
        return self._mx.array([seq.temperature for seq in seqs], dtype=self._mx.float32)

    # -- execution ---------------------------------------------------------

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        is_prefill: bool,
        inputs_embeds: Any | None = None,
    ) -> Any:
        """Run the model for one step; returns logits."""
        if is_prefill or self._cache is None:
            from mlx_lm.models.cache import make_prompt_cache

            self._cache = make_prompt_cache(self._model)
        out = self._model(input_ids, cache=self._cache)
        logits = out.logits if hasattr(out, "logits") else out
        return logits[:, -1, :]

    def set_context(self, is_prefill: bool, **kwargs: Any) -> None:
        """MLX keeps the per-step state in the cache object, not in globals."""
        self._is_prefill = is_prefill

    def reset_context(self) -> None:
        """MLX keeps the per-step state in the cache object, not in globals."""

    def sample(self, logits: Any, temperatures: Any, sampler: Any = None) -> Any:
        """Draw one token per sequence."""
        if sampler is not None:
            return sampler(logits, temperatures)
        scaled = logits.astype(self._mx.float32) / temperatures[:, None]
        return self._mx.argmax(scaled, axis=-1)

    def bridge_hidden(self) -> Any:
        """MiniMind-O's thinker is the only family with a bridge."""
        raise NotImplementedError("the MLX backend has no bridge hidden states")

    @property
    def model(self) -> Any:
        """The model object behind this runner."""
        return self._model


@dataclass(frozen=True)
class MlxBackend:
    """MLX on Apple Silicon."""

    device: str = "gpu"
    dtype: str = "float16"
    name: str = "mlx"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            varlen=False,
            graph=True,
            max_batch=1,
        )

    def make_config(self, model_path: str, **kwargs: Any) -> MlxConfig:
        return MlxConfig(
            model_path=model_path,
            dtype=kwargs.pop("dtype", None) or self.dtype,
            device=kwargs.pop("device", None) or self.device,
        )

    def make_stage_runner(self, model_class: type, config: Any) -> MlxStageRunner:
        return MlxStageRunner(model_class=model_class, config=config)


__all__ = ["MlxBackend", "MlxConfig", "MlxStageRunner"]
