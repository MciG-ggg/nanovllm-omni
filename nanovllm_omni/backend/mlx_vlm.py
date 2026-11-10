"""The MLX-VLM backend: SmolVLM's vision tower and text decoder on MLX.

mlx-vlm ships SmolVLM as a subclass of its Idefics3 model -- the 500M
checkpoint declares ``model_type: idefics3`` -- so this backend drives
mlx-vlm's model, processor and KV cache instead of porting the SigLIP
tower by hand. The vision tower and the connector run in MLX:
``Model.get_input_embeddings`` returns the merged text+image embedding,
which the stage hands to ``forward`` as ``inputs_embeds`` -- the same
contract the CUDA path uses.

The stage keeps one decode loop for both tracks; only the processor and
the image merge differ, and both live behind ``MlxVlmSupport``.

Nothing here imports mlx at module level, so the backend registry stays
importable on a machine without MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities


@dataclass
class MlxVlmConfig:
    """Per-stage config for the MLX VLM backend.

    The scheduler knobs (``max_num_seqs`` ... ``num_kvcache_blocks``) are
    what the fork ``Scheduler`` reads; MLX keeps one dense KV cache, so
    the block table is bookkeeping only. ``eos`` is filled in by the
    runner, which is the first thing to see the tokenizer.
    """

    model_path: str
    dtype: str = "float16"
    device: str = "gpu"
    enforce_eager: bool = True
    max_capture_bs: int = 1
    graph_max_seq_len: int = 2048
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 16384
    eos: int | None = None
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 64


class MlxVlmStageRunner:
    """Runs SmolVLM AR steps on mlx-vlm's MLX model.

    One sequence per forward: the merged image embedding is built for a
    single prompt and MLX's KV cache is dense, so a ragged batch has
    nowhere to go.
    """

    def __init__(self, model_class: type, config: MlxVlmConfig) -> None:
        import mlx.core as mx
        from mlx_vlm.utils import load as mlx_vlm_load

        self._mx = mx
        self.config = config
        mx.set_default_device(getattr(mx, config.device))
        # ``model_class`` is the torch factory contract; MLX picks the
        # architecture from ``config.json`` in the model directory.
        self._model, self.processor = mlx_vlm_load(config.model_path)
        self._model.set_dtype(getattr(mx, config.dtype))
        mx.eval(self._model.parameters())
        # The fork Scheduler stops on ``config.eos``; only the tokenizer
        # knows it, and the runner is what loads the tokenizer.
        self.config.eos = self.processor.tokenizer.eos_token_id
        self._cache: Any = None

    # -- batch preparation -------------------------------------------------

    def prepare_prefill(self, seqs: list[Any]) -> tuple[Any, Any]:
        """Prompt ids for the one sequence, plus its positions."""
        seq = _only(seqs)
        ids = self._mx.array([list(seq.token_ids)], dtype=self._mx.int32)
        positions = self._mx.arange(len(seq.token_ids), dtype=self._mx.int32)[None, :]
        return ids, positions

    def prepare_decode(self, seqs: list[Any]) -> tuple[Any, Any]:
        """The last token of the sequence, plus its position."""
        seq = _only(seqs)
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
        """Run the text decoder for one step; returns the last-position logits."""
        if is_prefill:
            from mlx_vlm.models.cache import make_prompt_cache

            self._cache = make_prompt_cache(self._model.language_model)
        elif self._cache is None:
            # A decode step without a prefill would feed one token into an
            # empty cache and return plausible-looking garbage.
            raise RuntimeError("the MLX VLM backend needs a prefill before its first decode step")
        if inputs_embeds is not None:
            embeds = (
                self._mx.expand_dims(inputs_embeds, 0) if inputs_embeds.ndim == 2 else inputs_embeds
            )
            outputs = self._model.language_model(input_ids, inputs_embeds=embeds, cache=self._cache)
        else:
            outputs = self._model.language_model(input_ids, cache=self._cache)
        # MLX's cache carries the RoPE offset, so ``positions`` is not consumed.
        return outputs.logits[:, -1, :]

    def set_context(self, is_prefill: bool, **kwargs: Any) -> None:
        """MLX keeps the per-step state in the cache object, not in globals."""

    def reset_context(self) -> None:
        """The cache spans the whole decode, so there is nothing per-step to clear."""

    def sample(self, logits: Any, temperatures: Any, sampler: Any = None) -> Any:
        """Draw one token per sequence from the global MLX stream."""
        if sampler is not None:
            return sampler(logits, temperatures)
        scaled = logits.astype(self._mx.float32) / (temperatures[:, None] + 1e-9)
        return self._mx.random.categorical(scaled, axis=-1)

    def bridge_hidden(self) -> Any:
        """SmolVLM has no second stage, so there is nothing to hand over."""
        return None

    @property
    def model(self) -> Any:
        """The model object behind this runner."""
        return self._model


class MlxVlmSupport:
    """The MLX side of a SmolVLM stage: the processor and the image merge.

    mlx-vlm's processor is not transformers': it expands the image
    placeholder into the patch tokens and returns ``mx.array`` values.
    """

    def __init__(self, runner: MlxVlmStageRunner) -> None:
        self._model = runner.model
        self.processor = runner.processor
        self.image_token_id = self._model.config.image_token_index
        self.eos_token_id = runner.config.eos

    def seed(self, value: int) -> None:
        """Seed the MLX global stream, which ``sample`` draws from."""
        import mlx.core as mx

        mx.random.seed(int(value))

    def encode(self, prompt_text: str, images: list[Any]) -> tuple[Any, Any]:
        """The processor's ``input_ids`` and ``pixel_values`` for one request."""
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        # mlx-vlm loads the chat template into the tokenizer, not into the
        # processor, so the tokenizer is the one that can render it.
        chat = self.processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=chat, images=images or None, return_tensors="mlx")
        return inputs["input_ids"], inputs.get("pixel_values")

    def merged_embeddings(self, input_ids: Any, pixel_values: Any) -> Any:
        """[T, hidden] merged embedding: vision features at the image-token slots.

        The image tiles this processor produces are all full, so the model
        builds the all-ones patch mask the processor would have passed.
        """
        if pixel_values is None:
            return None
        features = self._model.get_input_embeddings(input_ids, pixel_values).inputs_embeds
        return features.reshape(-1, features.shape[-1])


def _only(seqs: list[Any]) -> Any:
    """The one sequence this runner carries; MLX has no ragged batch here."""
    if len(seqs) != 1:
        raise ValueError(f"the MLX VLM runner runs one sequence per forward, got {len(seqs)}")
    return seqs[0]


@dataclass(frozen=True)
class MlxVlmBackend:
    """MLX-VLM on Apple Silicon."""

    device: str = "gpu"
    dtype: str = "float16"
    name: str = "mlx-vlm"

    def __post_init__(self) -> None:
        if self.device not in ("gpu", "cpu"):
            # MLX has no CUDA device, so a request for one lands on the GPU.
            object.__setattr__(self, "device", "gpu")

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            varlen=False,
            graph=True,
            max_batch=1,
        )

    def make_config(self, model_path: str, **kwargs: Any) -> MlxVlmConfig:
        # These shared stage kwargs belong to CUDA; MLX uses one device,
        # loads a converted model, and does not capture CUDA graphs.
        kwargs.pop("gpu_memory_utilization", None)
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("max_capture_bs", None)
        config = MlxVlmConfig(
            model_path=model_path,
            dtype=kwargs.pop("dtype", None) or self.dtype,
            device=kwargs.pop("device", None) or self.device,
            kvcache_block_size=kwargs.pop("kvcache_block_size", None) or 256,
            num_kvcache_blocks=kwargs.pop("num_kvcache_blocks", None) or 64,
            max_num_seqs=kwargs.pop("max_num_seqs", None) or 1,
            max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", None) or 16384,
            eos=kwargs.pop("eos", None),
        )
        if kwargs:
            raise ValueError(f"unsupported MLX VLM config kwargs: {', '.join(sorted(kwargs))}")
        return config

    def make_stage_runner(self, model_class: type, config: Any) -> MlxVlmStageRunner:
        return MlxVlmStageRunner(model_class=model_class, config=config)

    def make_talker_runner(self, model_path: str, **kwargs: Any) -> Any:
        """SmolVLM has no talker stage."""
        raise NotImplementedError("the MLX VLM backend has no talker runner")


__all__ = [
    "MlxVlmBackend",
    "MlxVlmConfig",
    "MlxVlmStageRunner",
    "MlxVlmSupport",
]
