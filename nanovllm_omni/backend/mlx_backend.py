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

Milestone 1 (ADR 0023) runs the MiniMind-O thinker alone. ``generate_mlx``
is the single-sequence AR loop: prefill, then one decode step per token,
with the same sampling recipe (temperature, repetition penalty, top-p,
seed) the CUDA thinker uses in ``models/minimind_omni/stage_runner.py``.
``generate_mlx_batch`` runs a static dense batch through the same recipe.
The model directory must be an ``mlx_lm`` model;
``scripts/convert_minimind_thinker_mlx.py`` writes one from the
minimind-3o checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities


class _BridgeRecorder:
    """A plain holder for the bridge rows; not an array framework module."""

    def __init__(self) -> None:
        self.rows: Any = None


def _bridge_layer_index(model: Any) -> int:
    """Which transformer layer the talker's bridge comes from.

    The vendor default is ``num_hidden_layers // 2 - 1`` (3 for this
    checkpoint); the fork reads ``bridge_layer`` from its config with the
    same fallback. A converted model that carries the key wins.
    """
    args = getattr(model, "args", None)
    config = getattr(model, "config", None)
    for holder in (args, config):
        if holder is None:
            continue
        value = getattr(holder, "bridge_layer", None)
        if value is None and isinstance(holder, dict):
            value = holder.get("bridge_layer")
        if value is not None:
            return int(value)
    layers = getattr(getattr(model, "model", None), "layers", None)
    count = len(layers) if layers is not None else 0
    return count // 2 - 1


@dataclass
class MlxConfig:
    """Per-stage config for the MLX backend.

    ``enforce_eager=False`` asks for ``mx.compile`` on the decode step;
    ``graph_max_seq_len`` is then the fixed KV capacity the graph is built
    over, since a reallocated cache changes shapes and re-traces.
    """

    model_path: str
    dtype: str = "float16"
    device: str = "gpu"
    enforce_eager: bool = True
    max_capture_bs: int = 1
    graph_max_seq_len: int = 2048
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 64
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    eos: int | None = None


class MlxStageRunner:
    """Runs AR steps on an ``mlx_lm`` model.

    A batch is dense: MLX has no variable-length attention, so every prompt
    is left-padded to the longest one and ``BatchKVCache`` masks the pad
    columns. This is the transitional form ADR 0015 rejects -- a short
    request still pays for the longest one, and the batch runs until every
    member finishes. A single sequence keeps the unpadded ``KVCache`` path.
    """

    def __init__(self, model_class: type, config: MlxConfig) -> None:
        import mlx.core as mx
        from mlx_lm.utils import load as mlx_load

        self._mx = mx
        self.config = config
        mx.set_default_device(getattr(mx, config.device))
        # ``model_class`` is the torch factory contract; MLX picks the
        # architecture from ``config.json`` in the model directory.
        self._model, self.tokenizer = mlx_load(config.model_path)
        if config.eos is None:
            config.eos = min(int(token) for token in self.tokenizer.eos_token_ids)
        self._model.set_dtype(getattr(mx, config.dtype))
        mx.eval(self._model.parameters())
        self._cache: Any = None
        self._rng: Any = None
        self._left_padding: list[int] = [0]
        self._bridge: Any = None
        self._bridge_layer = _bridge_layer_index(self._model)
        self._install_bridge_recorder()

    def _install_bridge_recorder(self) -> None:
        """Wrap the bridge layer so its output is recorded."""
        body = getattr(self._model, "model", None)
        layers = getattr(body, "layers", None)
        if not layers or not 0 <= self._bridge_layer < len(layers):
            return
        from mlx import nn

        recorder = _BridgeRecorder()
        inner = layers[self._bridge_layer]

        class _Recording(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = inner

            def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
                out = inner(x, *args, **kwargs)
                recorder.rows = out
                return out

        layers[self._bridge_layer] = _Recording()
        self._bridge = recorder

    # -- batch preparation -------------------------------------------------

    def prepare_prefill(self, seqs: list[Any]) -> tuple[Any, Any]:
        """Left-padded prompt ids for the batch, plus their positions."""
        lengths = [len(seq.token_ids) for seq in seqs]
        max_len = max(lengths)
        self._left_padding = [max_len - length for length in lengths]
        padded = [
            [0] * pad + list(seq.token_ids)
            for pad, seq in zip(self._left_padding, seqs, strict=True)
        ]
        ids = self._mx.array(padded, dtype=self._mx.int32)
        positions = self._mx.arange(max_len, dtype=self._mx.int32)[None, :]
        return ids, positions

    def prepare_decode(self, seqs: list[Any]) -> tuple[Any, Any]:
        """The last token of every sequence, plus its position."""
        ids = self._mx.array([[seq.last_token] for seq in seqs], dtype=self._mx.int32)
        positions = self._mx.array([[len(seq.token_ids) - 1] for seq in seqs], dtype=self._mx.int32)
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
        """Run the model for one step; returns the last-position logits."""
        if inputs_embeds is not None:
            raise NotImplementedError("the MLX backend has no inputs_embeds path")
        if is_prefill:
            from mlx_lm.models.cache import BatchKVCache

            new_cache = [BatchKVCache(self._left_padding) for _ in range(len(self._model.layers))]
            logits = self._model(input_ids, cache=new_cache)
            if self._cache is None:
                self._cache = new_cache
            else:
                for cached_layer, new_layer in zip(self._cache, new_cache, strict=True):
                    cached_layer.extend(new_layer)
            return logits[:, -1, :]
        if self._cache is None:
            # A decode step without a prefill would feed one token into an
            # empty cache and return plausible-looking garbage.
            raise RuntimeError("the MLX backend needs a prefill before its first decode step")
        # MLX's cache carries the RoPE offset, so ``positions`` is not consumed.
        logits = self._model(input_ids, cache=self._cache)
        return logits[:, -1, :]

    def filter_cache_rows(self, batch_indices: list[int]) -> None:
        """Keep the cache rows for the still-running sequences."""
        if self._cache is None:
            return
        if not batch_indices:
            self._cache = None
            return
        for cache in self._cache:
            cache.filter(batch_indices)

    def recapture_bridge(self, token_ids: list[int]) -> Any:
        """One full-sequence prefill, returning the bridge rows ``[T, hidden]``.

        The decode loop forwards one token per step against the paged cache,
        but the vendor runs full-sequence attention every step. The rows are
        not bit-equal and the talker cross-attends to them, so the bridge
        comes from this re-run instead.

        The prefill uses a fresh cache, so the running decode state in
        ``self._cache`` is untouched.
        """
        from mlx_lm.models.cache import BatchKVCache

        if not token_ids:
            return None
        cache = [BatchKVCache([0]) for _ in range(len(self._model.layers))]
        ids = self._mx.array([list(token_ids)], dtype=self._mx.int32)
        self._model(ids, cache=cache)
        rows = self.bridge_hidden()
        if rows is None:
            return None
        return rows[0]

    def set_context(self, is_prefill: bool, **kwargs: Any) -> None:
        """MLX keeps the per-step state in the cache object, not in globals."""

    def reset_context(self) -> None:
        """The cache spans the whole decode, so there is nothing per-step to clear."""

    def sample(self, logits: Any, temperatures: Any, sampler: Any = None) -> Any:
        """Draw one token per sequence."""
        if sampler is not None:
            return sampler(logits, temperatures)
        scaled = logits.astype(self._mx.float32) / (temperatures[:, None] + 1e-9)
        if self._rng is None:
            self._rng = self._mx.random.key(time.time_ns() % (2**31))
        self._rng, key = self._mx.random.split(self._rng)
        return self._mx.random.categorical(scaled, axis=-1, key=key)

    def bridge_hidden(self) -> Any:
        """The thinker's bridge rows, ``[B, T, hidden]``; ``None`` before the first forward.

        The vendor takes the hidden states after layer ``bridge_layer`` and
        before the final norm -- that is what the talker consumes, so the
        recorder wraps that one layer instead of the model's output.
        """
        return self._bridge.rows if self._bridge is not None else None

    @property
    def model(self) -> Any:
        """The model object behind this runner."""
        return self._model


@dataclass
class MlxGeneration:
    """One MLX decode: the token ids plus the two wall-clock phases."""

    token_ids: list[int]
    ttft_s: float
    decode_s: float


@dataclass
class MlxBatchGeneration:
    """A static dense batch: one token list per sequence plus the timings."""

    token_ids: list[list[int]]
    ttft_s: float
    decode_s: float
    steps: int


def _sequence(prompt_token_ids: list[int], sampling: Any) -> Any:
    """Build the fork ``Sequence`` that ``prepare_*`` reads."""
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

    # The clamp mirrors ``ThinkerStage._prepare_decode`` so both tracks feed
    # the sampler the same temperature.
    temperature = float(sampling.temperature)
    if temperature <= 1e-10:
        temperature = 1e-5
    return Sequence(
        prompt_token_ids,
        ForkSamplingParams(
            temperature=temperature,
            max_tokens=sampling.max_tokens,
            ignore_eos=True,
        ),
    )


def _make_sampler(mx: Any, sequences: list[Any], samplings: list[Any], keys: list[Any]) -> Any:
    """One sampler for one or many sequences, matching the CUDA recipe.

    ``top_k`` is unused on purpose: the vendor text path does not apply it
    (only the audio codebooks do), and ADR 0021 aligns against that path.
    Each row owns its key, so a sequence's draw does not depend on how many
    others share its batch.
    """

    def sampler(logits: Any, temperatures: Any) -> Any:
        tokens = []
        for index, (sequence, sampling) in enumerate(zip(sequences, samplings, strict=True)):
            scaled = logits[index].astype(mx.float32) / (temperatures[index] + 1e-9)
            penalty = mx.ones((scaled.shape[-1],), dtype=mx.float32)
            penalty[mx.array(sorted(set(sequence.token_ids)), dtype=mx.int32)] = float(
                (sampling.extra or {}).get("repetition_penalty", 1.05)
            )
            scaled = scaled / penalty
            if sampling.top_p < 1.0:
                order = mx.argsort(-scaled)
                ordered = mx.take_along_axis(scaled, order, axis=-1)
                cumulative = mx.cumsum(mx.softmax(ordered), axis=-1)
                removed = cumulative > sampling.top_p
                removed[1:] = removed[:-1]
                removed[0] = False
                scaled = mx.put_along_axis(
                    scaled, order, mx.where(removed, -mx.inf, ordered), axis=-1
                )
            keys[index], step_key = mx.random.split(keys[index])
            tokens.append(mx.random.categorical(scaled[None, :], axis=-1, key=step_key))
        return mx.concatenate(tokens, axis=0)

    return sampler


def _seed_key(mx: Any, sampling: Any) -> Any:
    seed = int(sampling.seed) if sampling.seed is not None else time.time_ns() % (2**31)
    return mx.random.key(seed)


def generate_mlx(
    runner: MlxStageRunner,
    prompt_token_ids: list[int],
    sampling: Any,
    eos_id: int | None = None,
) -> MlxGeneration:
    """Prefill then decode one sequence with the CUDA thinker's sampling recipe."""
    import mlx.core as mx

    sequence = _sequence(prompt_token_ids, sampling)
    sampler = _make_sampler(mx, [sequence], [sampling], [_seed_key(mx, sampling)])

    runner.set_context(is_prefill=True)
    started = time.perf_counter()
    input_ids, positions = runner.prepare_prefill([sequence])
    temperatures = runner.prepare_sample([sequence])
    logits = runner.forward(input_ids, positions, True)
    token = int(runner.sample(logits, temperatures, sampler=sampler)[0].item())
    ttft_s = time.perf_counter() - started
    sequence.append_token(token)
    generated = [token]

    decode_started = time.perf_counter()
    while len(generated) < sampling.max_tokens and (eos_id is None or generated[-1] != eos_id):
        runner.set_context(is_prefill=False)
        input_ids, positions = runner.prepare_decode([sequence])
        temperatures = runner.prepare_sample([sequence])
        logits = runner.forward(input_ids, positions, False)
        token = int(runner.sample(logits, temperatures, sampler=sampler)[0].item())
        sequence.append_token(token)
        generated.append(token)
    decode_s = time.perf_counter() - decode_started
    runner.reset_context()
    return MlxGeneration(token_ids=generated, ttft_s=ttft_s, decode_s=decode_s)


def generate_mlx_batch(
    runner: MlxStageRunner,
    prompt_token_ids_list: list[list[int]],
    samplings: list[Any],
    eos_ids: list[int | None] | None = None,
) -> MlxBatchGeneration:
    """Prefill then decode a static dense batch, each row to its own stop."""
    import mlx.core as mx

    if len(prompt_token_ids_list) != len(samplings):
        raise ValueError("one sampling config per prompt is required")
    if eos_ids is None:
        eos_ids = [None] * len(prompt_token_ids_list)
    sequences = [
        _sequence(prompt, sampling)
        for prompt, sampling in zip(prompt_token_ids_list, samplings, strict=True)
    ]
    sampler = _make_sampler(
        mx, sequences, samplings, [_seed_key(mx, sampling) for sampling in samplings]
    )

    runner.set_context(is_prefill=True)
    started = time.perf_counter()
    input_ids, positions = runner.prepare_prefill(sequences)
    temperatures = runner.prepare_sample(sequences)
    logits = runner.forward(input_ids, positions, True)
    first = runner.sample(logits, temperatures, sampler=sampler).tolist()
    ttft_s = time.perf_counter() - started
    generated = [[token] for token in first]
    finished = [eos is not None and token == eos for token, eos in zip(first, eos_ids, strict=True)]
    # The first decode step reads ``last_token``; without this the prompt's
    # last token would be fed twice and every later token would shift by one.
    for sequence, token in zip(sequences, first, strict=True):
        sequence.append_token(token)

    steps = 1
    decode_started = time.perf_counter()
    while True:
        active = {
            index
            for index in range(len(sequences))
            if not finished[index] and len(generated[index]) < samplings[index].max_tokens
        }
        if not active:
            break
        # A finished row keeps feeding its last token so the dense batch
        # stays rectangular; its cache grows but nothing reads that row.
        input_ids, positions = runner.prepare_decode(sequences)
        temperatures = runner.prepare_sample(sequences)
        logits = runner.forward(input_ids, positions, False)
        tokens = runner.sample(logits, temperatures, sampler=sampler).tolist()
        steps += 1
        for index, token in enumerate(tokens):
            if index not in active:
                continue
            sequences[index].append_token(token)
            generated[index].append(token)
            if eos_ids[index] is not None and token == eos_ids[index]:
                finished[index] = True
    decode_s = time.perf_counter() - decode_started
    runner.reset_context()
    return MlxBatchGeneration(token_ids=generated, ttft_s=ttft_s, decode_s=decode_s, steps=steps)


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
        return MlxConfig(
            model_path=model_path,
            dtype=kwargs.pop("dtype", None) or self.dtype,
            device=kwargs.pop("device", None) or self.device,
            kvcache_block_size=kwargs.pop("kvcache_block_size", None) or 256,
            num_kvcache_blocks=kwargs.pop("num_kvcache_blocks", None) or 64,
            max_num_seqs=kwargs.pop("max_num_seqs", None) or 512,
            max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", None) or 16384,
            eos=kwargs.pop("eos", None),
        )

    def make_stage_runner(self, model_class: type, config: Any) -> MlxStageRunner:
        return MlxStageRunner(model_class=model_class, config=config)

    def make_talker_runner(self, model_path: str, **kwargs: Any) -> Any:
        """Build the runner that decodes MiniMind audio codes on MLX."""
        from .mlx_talker import MlxTalkerRunner

        return MlxTalkerRunner(
            model_path,
            dtype=kwargs.pop("dtype", None) or self.dtype,
            device=kwargs.pop("device", None) or self.device,
            **kwargs,
        )


__all__ = [
    "MlxBackend",
    "MlxBatchGeneration",
    "MlxConfig",
    "MlxGeneration",
    "MlxStageRunner",
    "generate_mlx",
    "generate_mlx_batch",
]
