"""The MiniMind talker decode loop on MLX.

The schedule, the emission gate and the sampling recipe are the ones in
``talker_torch``; only the array ops are MLX's. The model it drives is
``talker_mlx``.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np

from .talker_mlx import MlxTalkerKVCache, load_mlx_talker
from .talker_recipe import (
    AUDIO_PAD_TOKEN,
    AUDIO_STOP_TOKEN,
    NUM_CODEBOOKS,
    TalkerRequest,
    can_emit_frame,
)

_TOP_K = 50


def _as_mx(value: Any, dtype: Any) -> mx.array:
    """Bring a bridge or speaker tensor into MLX, whatever framework it came from."""
    if isinstance(value, mx.array):
        return value.astype(dtype)
    return mx.array(np.asarray(value)).astype(dtype)


class MlxTalkerRunner:
    """Decodes MiniMind audio codes with MLX ops."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "gpu",
        dtype: str = "float16",
        impl: str = "rewritten",
        kv: str = "dense",
    ) -> None:
        if impl != "rewritten":
            # The vendor arm is a transformers dynamic module on torch; there
            # is no MLX copy of it, so the rewritten talker is the only arm.
            raise ValueError(
                f"the MLX backend has no {impl!r} talker arm; use talker_impl='rewritten'"
            )
        if kv not in ("dense", "windowed"):
            raise ValueError(f"unknown talker_kv {kv!r}; expected 'dense' or 'windowed'")
        self._kv = kv
        self._dtype = getattr(mx, dtype)
        self._model = load_mlx_talker(model_path, dtype=dtype, device=device)

    def generate(self, request: TalkerRequest) -> list[list[int]]:
        """Decode the audio codes and return the emitted frames."""
        model = self._model
        bridge = _as_mx(request.bridge, self._dtype)
        bridge = mx.expand_dims(bridge, 0)
        spk_emb = request.speaker_embedding
        if spk_emb is not None:
            spk_emb = _as_mx(spk_emb, self._dtype)

        start_pos = request.start_pos
        num_steps = request.num_steps
        seed = int(request.seed) if request.seed is not None else time.time_ns() % (2**31)
        key = mx.random.key(seed)

        # Vendor ``step`` is the count of generated tokens BEFORE this
        # iteration's append, so the first iteration has ``step = 0`` and
        # ``audio_step = -1`` (every codebook pads).
        audio_codes: list[list[int]] = [[] for _ in range(NUM_CODEBOOKS)]
        audio_stop_pos: list[int | None] = [None] * NUM_CODEBOOKS
        emitted_frames: list[list[int]] = []

        # Audio buffer: prompt columns stay pad; generated column
        # ``start_pos + p`` holds ``audio_codes[i][p]`` for ``i < min(p, 8)``
        # (vendor's diagonal backfill). Each step writes only the column that
        # just became final.
        buf = mx.full((1, NUM_CODEBOOKS, start_pos + num_steps), AUDIO_PAD_TOKEN, dtype=mx.int32)

        cache = MlxTalkerKVCache(len(model.layers)) if self._kv == "windowed" else None

        for step in range(num_steps):
            audio_step = step - 1
            # Vendor's forward sees ``audio_buffer`` BEFORE this iteration's
            # append, so the sequence length is ``start_pos + step``.
            seq_len = start_pos + step
            column = step - 1
            if column >= 0:
                filled = min(column, NUM_CODEBOOKS)
                if filled:
                    buf[0, :filled, start_pos + column] = mx.array(
                        [audio_codes[index][column] for index in range(filled)], dtype=mx.int32
                    )

            # The dense path recomputes the whole prefix every step. The
            # windowed path recomputes only the newest columns and reads the
            # settled prefix from the cache, whose length is also the absolute
            # RoPE offset (ADR 0022).
            lo = cache.length if cache is not None else 0
            logits_list = model(bridge[:, lo:seq_len, :], buf[:, :, lo:seq_len], spk_emb, cache)

            # Sample one code per codebook: ``logits_i = al[0,-1,:] / 0.2``,
            # then ``for prev in audio_codes[i][-3:]: logits_i[prev] /= 1.05``
            # (last-3 of THAT layer only), then ``topk(50)`` + multinomial.
            for index, logits in enumerate(logits_list):
                if audio_step < index:
                    audio_codes[index].append(AUDIO_PAD_TOKEN)
                    continue
                logits_i = logits[0, -1, :] / request.temperature
                for previous_code in audio_codes[index][-3:]:
                    logits_i[previous_code] = logits_i[previous_code] / request.repetition_penalty
                order = mx.argsort(-logits_i)[:_TOP_K]
                key, step_key = mx.random.split(key)
                probabilities = mx.softmax(logits_i[order].astype(mx.float32), axis=-1)
                sampled = mx.random.categorical(probabilities, key=step_key)
                code = int(order[sampled].item())
                audio_codes[index].append(code)
                if audio_stop_pos[index] is None and code >= 2048:
                    audio_stop_pos[index] = len(audio_codes[index]) - 1

            # Vendor break check runs BEFORE the append/emission.
            if audio_codes[7] and audio_codes[7][-1] == AUDIO_STOP_TOKEN:
                break

            # HF emission gate: ``audio_step >= 7`` with the diagonal
            # ``frame[i] = audio_codes[i][step - 7 + i]`` and the
            # active-layer filter over all 8 layers.
            if audio_step >= NUM_CODEBOOKS - 1:
                if can_emit_frame(audio_stop_pos, step):
                    idx = [step - (NUM_CODEBOOKS - 1) + index for index in range(NUM_CODEBOOKS)]
                    emitted_frames.append(
                        [audio_codes[index][idx[index]] for index in range(NUM_CODEBOOKS)]
                    )
                else:
                    # A closed gate never reopens, so every remaining step
                    # would be a full-sequence forward with no observable
                    # effect.
                    break

        return emitted_frames


__all__ = ["MlxTalkerRunner"]
