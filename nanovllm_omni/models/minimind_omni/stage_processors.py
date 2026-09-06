"""Pure MiniMind-O payload processors for the full three-stage pipeline.

Reshape and validate stage handoff data; never call a model. Public
symbols: ``ThinkerStageOutput``, ``TalkerInputPayload``,
``Code2WavInputPayload``, ``thinker2talker``, ``talker2code2wav``,
``AUDIO_PAD_TOKEN_ID``.

Contracts:
* ``ThinkerStageOutput.bridge_states``: ``[T, H]`` (or ``[B, T, H]``
  flattened). ``prompt_token_ids`` / ``output_token_ids`` are 1-D.
* ``TalkerInputPayload.input_ids``: ``[T_prompt]`` placeholder audio-pad
  IDs; ``text_token_ids`` retains the aligned thinker text IDs.
* ``Code2WavInputPayload.audio_codes``: frame-major ``[F, C]`` with one
  column per codec codebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .bundle import MIMI_SAMPLE_RATE

AUDIO_PAD_TOKEN_ID = 2049


@dataclass(frozen=True)
class ThinkerStageOutput:
    """Explicit full-mode output emitted by a thinker stage."""

    bridge_states: torch.Tensor
    prompt_token_ids: Sequence[int] = ()
    output_token_ids: Sequence[int] = ()
    text_token_ids: Sequence[int] = ()
    input_ids: Any | None = None
    text_state: Any = None
    speaker_embedding: torch.Tensor | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TalkerInputPayload:
    """Full-mode input for the talker stage."""

    input_ids: torch.Tensor
    bridge_states: torch.Tensor
    text_token_ids: tuple[int, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()
    output_token_ids: tuple[int, ...] = ()
    speaker_embedding: torch.Tensor | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def additional_information(self) -> dict[str, Any]:
        """Return the info shape consumed by the local talker helpers."""
        return {
            "hidden_states": {"bridge": self.bridge_states},
            "ids": {
                "prompt": list(self.prompt_token_ids),
                "output": list(self.output_token_ids),
                "all": list(self.text_token_ids),
            },
        }


@dataclass(frozen=True)
class Code2WavInputPayload:
    """Full-mode input for Code2Wav, with frame-major audio code rows."""

    audio_codes: torch.Tensor
    sample_rate: int = MIMI_SAMPLE_RATE
    request_id: str | None = None
    device: torch.device | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> dict[str, torch.Tensor]:
        return {"audio": self.audio_codes}


def _value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _as_token_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(token) for token in value.detach().cpu().reshape(-1).tolist()]
    return [int(token) for token in value]


def _metadata(payload: Any) -> dict[str, Any]:
    metadata = _value(payload, "metadata")
    result = dict(metadata) if isinstance(metadata, Mapping) else {}
    meta = _value(payload, "meta")
    if isinstance(meta, Mapping):
        result.update(meta)
    return result


def _request_id(payload: Any) -> str | None:
    request_id = _value(payload, "request_id")
    return None if request_id is None else str(request_id)


def _bridge_from(payload: Any) -> Any:
    if isinstance(payload, ThinkerStageOutput):
        return payload.bridge_states
    additional = _value(payload, "additional_information")
    hidden = _value(additional, "hidden_states")
    bridge = _value(hidden, "bridge")
    if bridge is not None:
        return bridge
    hidden = _value(payload, "hidden_states")
    bridge = _value(hidden, "bridge")
    if bridge is not None:
        return bridge
    bridge = _value(payload, "bridge_states")
    if bridge is not None:
        return bridge
    for envelope_name in ("multimodal_output", "multimodal_outputs"):
        envelope = _value(payload, envelope_name)
        hidden = _value(envelope, "hidden_states")
        bridge = _value(hidden, "bridge")
        if bridge is not None:
            return bridge
        bridge = _value(envelope, "hidden_states.bridge")
        if bridge is not None:
            return bridge
    return None


def _aligned_text_ids(payload: Any) -> tuple[list[int], list[int], list[int]]:
    if isinstance(payload, ThinkerStageOutput):
        prompt_ids = _as_token_list(payload.prompt_token_ids)
        output_ids = _as_token_list(payload.output_token_ids)
        all_ids = _as_token_list(payload.text_token_ids) or prompt_ids + output_ids
        if not all_ids:
            all_ids = _as_token_list(payload.input_ids)
    else:
        ids = _value(payload, "ids")
        prompt_ids = _as_token_list(_value(ids, "prompt")) or _as_token_list(
            _value(payload, "prompt_token_ids")
        )
        output_ids = _as_token_list(_value(ids, "output")) or _as_token_list(
            _value(payload, "output_token_ids")
        )
        all_ids = _as_token_list(_value(ids, "all")) or prompt_ids + output_ids
        if not all_ids:
            all_ids = _as_token_list(_value(payload, "input_ids"))
        if not all_ids:
            all_ids = _as_token_list(_value(payload, "text_token_ids"))
    return prompt_ids, output_ids, all_ids or [AUDIO_PAD_TOKEN_ID]


def _normalise_bridge(payload: Any, text_len: int) -> torch.Tensor:
    bridge = _bridge_from(payload)
    if not isinstance(bridge, torch.Tensor):
        raise ValueError(
            "MiniMind thinker2talker requires non-empty hidden_states.bridge "
            f"tensor in full mode; got {type(bridge).__name__}."
        )
    if bridge.ndim == 3:
        bridge = bridge.reshape(-1, bridge.shape[-1])
    if bridge.ndim != 2:
        raise ValueError(
            "MiniMind thinker bridge states must have rank 2 [tokens, hidden] "
            f"or rank 3 [batch, tokens, hidden], got {tuple(bridge.shape)}."
        )
    if bridge.shape[0] == 0 or bridge.shape[1] == 0:
        raise ValueError(
            "MiniMind thinker2talker received empty bridge hidden states in full mode."
        )
    if text_len > bridge.shape[0]:
        raise ValueError(
            "MiniMind thinker bridge states are shorter than text state: "
            f"{bridge.shape[0]} rows for {text_len} text tokens."
        )
    return bridge[-text_len:].detach().to(dtype=torch.float32)


def _speaker_embedding(payload: Any) -> torch.Tensor | None:
    if isinstance(payload, ThinkerStageOutput):
        return payload.speaker_embedding
    for key in ("speaker_embedding", "spk_emb", "speaker_emb"):
        value = _value(payload, key)
        if isinstance(value, torch.Tensor):
            return value
    metadata = _metadata(payload)
    for key in ("speaker_embedding", "spk_emb", "speaker_emb"):
        value = metadata.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def thinker2talker(payload: Any, prompt: str = "") -> Any:
    """Convert one thinker result into a talker input.

    ``prompt`` is part of the local runner hook signature. Full-mode
    payloads carry token IDs from the thinker; tokenizing here would
    violate the pure processor boundary, so the prompt is unused.
    """
    del prompt
    if isinstance(payload, TalkerInputPayload):
        return payload

    prompt_ids, output_ids, all_ids = _aligned_text_ids(payload)
    bridge = _normalise_bridge(payload, len(all_ids))
    if len(all_ids) > bridge.shape[0]:  # defensive; _normalise_bridge already checks
        all_ids = all_ids[-bridge.shape[0] :]
    if len(prompt_ids) > len(all_ids):
        prompt_ids = prompt_ids[-len(all_ids) :]
        output_ids = []
    else:
        output_ids = output_ids[: len(all_ids) - len(prompt_ids)]
    input_ids = torch.full(
        (max(1, len(prompt_ids)),),
        AUDIO_PAD_TOKEN_ID,
        dtype=torch.long,
        device=bridge.device,
    )
    metadata = _metadata(payload)
    text_state = _value(payload, "text_state")
    if text_state is not None:
        metadata["text_state"] = text_state
    return TalkerInputPayload(
        input_ids=input_ids,
        bridge_states=bridge,
        text_token_ids=tuple(all_ids),
        prompt_token_ids=tuple(prompt_ids),
        output_token_ids=tuple(output_ids),
        speaker_embedding=_speaker_embedding(payload),
        request_id=_request_id(payload),
        metadata=metadata,
    )


def _audio_codes_from(payload: Any) -> Any:
    if isinstance(payload, Code2WavInputPayload):
        return payload.audio_codes
    if hasattr(payload, "audio_codes"):
        return payload.audio_codes
    for envelope_name in ("multimodal_output", "multimodal_outputs"):
        envelope = _value(payload, envelope_name)
        codes = _value(envelope, "codes")
        audio = _value(codes, "audio")
        if audio is not None:
            return audio
    codes = _value(payload, "codes")
    return _value(codes, "audio")


def _normalise_audio_codes(payload: Any) -> torch.Tensor:
    audio_codes = _audio_codes_from(payload)
    request_id = _request_id(payload)
    if not isinstance(audio_codes, torch.Tensor):
        raise TypeError(
            "MiniMind talker2code2wav expected codes.audio tensor "
            f"for request {request_id!r}, got {type(audio_codes).__name__}."
        )
    if audio_codes.ndim != 2:
        raise ValueError(
            "MiniMind talker audio codes must have shape [frames, codebooks], "
            f"got {tuple(audio_codes.shape)} for request {request_id!r}."
        )
    if audio_codes.shape[0] == 0 or audio_codes.shape[1] == 0:
        raise ValueError(
            "MiniMind talker audio codes must be non-empty [frames, codebooks] "
            f"for request {request_id!r}."
        )
    return audio_codes.detach().to(dtype=torch.long)


def talker2code2wav(payload: Any, prompt: str = "") -> Any:
    """Convert one talker result into a Code2Wav input."""
    del prompt
    if isinstance(payload, Code2WavInputPayload):
        return payload

    audio_codes = _normalise_audio_codes(payload)
    metadata = _metadata(payload)
    sample_rate = metadata.get("sample_rate", MIMI_SAMPLE_RATE)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"MiniMind Code2Wav sample_rate must be a positive integer, got {sample_rate!r}."
        )
    return Code2WavInputPayload(
        audio_codes=audio_codes,
        sample_rate=sample_rate,
        request_id=_request_id(payload),
        device=audio_codes.device,
        metadata=metadata,
    )


__all__ = [
    "AUDIO_PAD_TOKEN_ID",
    "Code2WavInputPayload",
    "TalkerInputPayload",
    "ThinkerStageOutput",
    "talker2code2wav",
    "thinker2talker",
]
