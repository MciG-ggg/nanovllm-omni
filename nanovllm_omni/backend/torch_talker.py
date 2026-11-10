"""The MiniMind talker on torch: the model plus the staggered decode loop.

The loop is the vendor's ``stream_generate`` schedule, moved out of the
stage so the stage can pick a backend the way it does for the thinker.
It lives here rather than in the family because the same schedule on MLX
needs different array ops, and a runner per device is the honest form.

The recipe itself -- eight codebooks, the emission gate, the token
constants -- lives in ``talker_recipe``, which both runners import.
"""

from __future__ import annotations

import os
from typing import Any

from .base import TalkerRequest
from .talker_recipe import (
    AUDIO_PAD_TOKEN,
    AUDIO_SPK_TOKEN,
    AUDIO_STOP_TOKEN,
    NUM_CODEBOOKS,
    can_emit_frame,
)


class TorchTalkerRunner:
    """Decodes MiniMind audio codes with torch ops."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        impl: str = "rewritten",
        kv: str = "dense",
    ) -> None:
        if impl not in ("rewritten", "vendor"):
            raise ValueError(f"unknown talker_impl {impl!r}; expected 'rewritten' or 'vendor'")
        if kv not in ("dense", "windowed"):
            raise ValueError(f"unknown talker_kv {kv!r}; expected 'dense' or 'windowed'")
        if kv == "windowed" and impl != "rewritten":
            raise ValueError("talker_kv='windowed' requires talker_impl='rewritten'")
        self._impl = impl
        self._kv = kv
        self._device = device
        import torch

        self._generator = torch.Generator(device=device)
        self._model = self._load(model_path, device, dtype)

    def _load(self, model_path: str, device: str, dtype: str) -> Any:
        """Load the talker arm selected by ``impl``."""
        import torch

        if self._impl == "rewritten":
            from ..models.minimind_omni.talker import load_talker

            return load_talker(model_path, device=device, dtype=getattr(torch, dtype))

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_cls = get_class_from_dynamic_module("model_omni.OmniConfig", model_path)
        talker_cls = get_class_from_dynamic_module("model_omni.TalkerModule", model_path)
        config = config_cls.from_pretrained(model_path)
        talker = talker_cls(config).eval()
        weights = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(weights):
            state = torch.load(weights, map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file

            state = {}
            for name in sorted(os.listdir(model_path)):
                if name.endswith(".safetensors"):
                    state.update(load_file(os.path.join(model_path, name)))
        talker.load_state_dict(
            {
                key[len("talker.") :]: value
                for key, value in state.items()
                if key.startswith("talker.")
            },
            strict=True,
        )
        return talker.to(device=device, dtype=getattr(torch, dtype))

    def _logits(self, bridge: Any, audio_codes: Any, spk_emb: Any, cache: Any) -> Any:
        """Return the talker's eight logit tensors for the given columns."""
        import torch

        if self._impl == "rewritten":
            return self._model(bridge, audio_codes, spk_emb, cache)

        talker = self._model
        embeds = talker.embed_tokens(audio_codes)
        if spk_emb is not None:
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            mask = (audio_codes[:, 0, :] == AUDIO_SPK_TOKEN).unsqueeze(-1)
            embeds = torch.where(mask, talker.spk_proj(spk_emb).unsqueeze(1), embeds)
        hidden = (
            talker.embed_proj(bridge) * talker.text_scale
            + talker.codec_proj(embeds) * talker.audio_scale
        )
        seq_len = hidden.shape[1]
        position_embeddings = (talker.freqs_cos[:seq_len], talker.freqs_sin[:seq_len])
        for layer in talker.layers:
            hidden, _ = layer(
                hidden,
                position_embeddings,
                past_key_value=None,
                use_cache=False,
            )
        hidden = talker.norm(hidden)
        return talker.lm_head(hidden)

    def generate(self, request: TalkerRequest) -> list[list[int]]:
        """Decode the audio codes and return the emitted frames."""
        import torch

        model = self._model
        device = self._device
        if request.seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(int(request.seed))

        bridge = request.bridge.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
        spk_emb = request.speaker_embedding

        start_pos = request.start_pos
        num_steps = request.num_steps

        # Vendor ``step`` is the count of generated tokens BEFORE this
        # iteration's append, so the first iteration has ``step = 0`` and
        # ``audio_step = -1`` (every codebook pads). Codebook ``i`` starts
        # sampling real codes once ``audio_step >= i``.
        audio_codes: list[list[int]] = [[] for _ in range(NUM_CODEBOOKS)]
        audio_stop_pos: list[int | None] = [None] * NUM_CODEBOOKS
        emitted_frames: list[list[int]] = []

        # Audio buffer: prompt columns stay pad; generated column
        # ``start_pos + p`` holds ``audio_codes[i][p]`` for ``i < min(p, 8)``
        # (vendor's diagonal backfill). Column ``start_pos + p`` is final as
        # soon as codebook ``p - 1`` has sampled step ``p``, and
        # ``audio_codes[i][p]`` never changes after it is appended -- so each
        # step writes only the column that just became final instead of
        # rebuilding the whole buffer. Same values either way; ADR 0007 has
        # the timings that motivated this.
        buf = torch.full(
            (1, NUM_CODEBOOKS, start_pos + num_steps),
            AUDIO_PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )

        cache = None
        if self._kv == "windowed":
            from ..models.minimind_omni.talker import TalkerKVCache

            cache = TalkerKVCache(len(model.layers))

        with torch.no_grad():
            for step in range(num_steps):
                audio_step = step - 1
                # Vendor's forward sees ``audio_buffer`` BEFORE this
                # iteration's append, so the sequence length is
                # ``start_pos + step`` -- the very first forward covers
                # exactly the prompt.
                seq_len = start_pos + step
                column = step - 1
                if column >= 0:
                    filled = min(column, NUM_CODEBOOKS)
                    if filled:
                        buf[0, :filled, start_pos + column] = torch.tensor(
                            [audio_codes[i][column] for i in range(filled)],
                            dtype=torch.long,
                            device=device,
                        )

                # The dense path recomputes the whole prefix every step
                # (``use_cache=False`` on the parity path). The windowed path
                # recomputes only the newest columns and reads the settled
                # prefix from the cache, whose length is also the absolute
                # RoPE offset (ADR 0022).
                lo = cache.length if cache is not None else 0
                logits_list = self._logits(
                    bridge[:, lo:seq_len, :], buf[:, :, lo:seq_len], spk_emb, cache
                )

                # Sample one code per codebook. HF ``model_omni.py``
                # stream_generate: ``logits_i = al[0,-1,:] / 0.2``, then
                # ``for prev in audio_codes[i][-3:]: logits_i[prev] /= 1.05``
                # (last-3 of THAT layer only), then ``topk(50)`` +
                # multinomial. The penalty is real in the HF reference --
                # it is only absent from the vLLM refactor.
                for i, logits in enumerate(logits_list):
                    if audio_step < i:
                        audio_codes[i].append(AUDIO_PAD_TOKEN)
                        continue
                    # Match vendor exactly: keep the model dtype through
                    # temperature scaling and top-k, then draw with
                    # torch.multinomial from the 50-token softmax. Using the
                    # generic Gumbel sampler consumes a different RNG path;
                    # one different code recursively changes every later
                    # audio-buffer column.
                    logits_i = logits[0, -1, :].clone() / request.temperature
                    for previous_code in audio_codes[i][-3:]:
                        logits_i[previous_code] /= request.repetition_penalty
                    top_value, top_index = logits_i.topk(50)
                    sampled = torch.multinomial(
                        torch.softmax(top_value, dim=-1), 1, generator=self._generator
                    )
                    code = top_index[sampled].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

                # Vendor break check runs BEFORE the append/emission.
                if audio_codes[7] and audio_codes[7][-1] == AUDIO_STOP_TOKEN:
                    break

                # HF emission gate: ``audio_step >= 7`` with the diagonal
                # ``frame[i] = audio_codes[i][step - 7 + i]`` and the
                # active-layer filter ``stop_pos[i] is None or
                # step - 7 + i < stop_pos[i]`` over all 8 layers.
                if audio_step >= NUM_CODEBOOKS - 1:
                    if can_emit_frame(audio_stop_pos, step):
                        idx = [step - (NUM_CODEBOOKS - 1) + i for i in range(NUM_CODEBOOKS)]
                        emitted_frames.append(
                            [audio_codes[i][idx[i]] for i in range(NUM_CODEBOOKS)]
                        )
                    else:
                        # A closed gate never reopens, so every remaining step
                        # would be a full-sequence forward with no observable
                        # effect. Stopping here keeps the frames byte-identical.
                        break

        return emitted_frames


__all__ = ["TorchTalkerRunner"]
