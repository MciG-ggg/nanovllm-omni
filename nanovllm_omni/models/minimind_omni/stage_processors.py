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

from collections.abc import Mapping, Sequence  # noqa: F401 (Mapping used in dataclass fields)
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


def _as_token_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(token) for token in value.detach().cpu().reshape(-1).tolist()]
    return [int(token) for token in value]


def _bridge_from(payload: ThinkerStageOutput) -> Any:
    return payload.bridge_states


def _aligned_text_ids(payload: ThinkerStageOutput) -> tuple[list[int], list[int], list[int]]:
    prompt_ids = _as_token_list(payload.prompt_token_ids)
    output_ids = _as_token_list(payload.output_token_ids)
    all_ids = _as_token_list(payload.text_token_ids) or prompt_ids + output_ids
    if not all_ids:
        all_ids = _as_token_list(payload.input_ids)
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
        # Bridge may have fewer rows than text_len (off-by-one between
        # prefill+decode counts). Use what's available.
        return bridge.detach().to(dtype=torch.float32)
    return bridge[-text_len:].detach().to(dtype=torch.float32)


def _speaker_embedding(payload: ThinkerStageOutput) -> torch.Tensor | None:
    return payload.speaker_embedding


def thinker2talker(payload: Any, prompt: str = "") -> TalkerInputPayload:
    """Convert one thinker result into a talker input.

    Only ``ThinkerStageOutput`` is accepted (anything else is a TypeError,
    mirroring smolvla's ``vlm2action``). ``start_pos``/``num_steps`` are
    derived here so TalkerStage never re-splits: ``start_pos`` is the
    prompt length, ``num_steps`` the generated-row count.

    ``prompt`` is part of the local runner hook signature and unused.
    """
    del prompt
    if isinstance(payload, TalkerInputPayload):
        return payload
    if not isinstance(payload, ThinkerStageOutput):
        raise TypeError(f"thinker2talker expects ThinkerStageOutput, got {type(payload).__name__}")

    prompt_ids, output_ids, all_ids = _aligned_text_ids(payload)
    # Vendor ``stream_generate`` seeds ``audio_buffer`` at the FULL prompt
    # length and feeds ``cat(audio_buffer, input_ids)`` on every forward,
    # so the talker is conditioned on prompt rows + generated rows. An
    # earlier revision sliced the bridge to the generated tail only,
    # which made the talker prompt-independent (identical frame 0 for
    # every prompt). Keep the full bridge here; start_pos/num_steps below
    # tell the talker where the generated rows begin.
    bridge = _normalise_bridge(payload, len(all_ids))
    if len(all_ids) > bridge.shape[0]:  # defensive; _normalise_bridge already checks
        drop = len(all_ids) - bridge.shape[0]
        all_ids = all_ids[drop:]
        prompt_ids = prompt_ids[drop:] if drop < len(prompt_ids) else []
    input_ids = torch.full(
        (max(1, len(prompt_ids)),),
        AUDIO_PAD_TOKEN_ID,
        dtype=torch.long,
        device=bridge.device,
    )
    start_pos = len(prompt_ids)
    num_steps = bridge.shape[0] - start_pos
    metadata = dict(payload.metadata) if isinstance(payload.metadata, Mapping) else {}
    if payload.text_state is not None:
        metadata["text_state"] = payload.text_state
    metadata["start_pos"] = start_pos
    metadata["num_steps"] = num_steps
    return TalkerInputPayload(
        input_ids=input_ids,
        bridge_states=bridge,
        text_token_ids=tuple(all_ids),
        prompt_token_ids=tuple(prompt_ids),
        output_token_ids=tuple(output_ids),
        speaker_embedding=payload.speaker_embedding,
        request_id=payload.request_id,
        metadata=metadata,
    )


def _audio_codes_from(payload: Any) -> Any:
    if isinstance(payload, Code2WavInputPayload):
        return payload.audio_codes
    return getattr(payload, "audio_codes", None)


def _normalise_audio_codes(payload: Any) -> torch.Tensor:
    audio_codes = _audio_codes_from(payload)
    request_id = getattr(payload, "request_id", None)
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


def talker2code2wav(payload: Any, prompt: str = "") -> Code2WavInputPayload:
    """Convert one talker result into a Code2Wav input.

    Only ``TalkerOutput`` (or an already-built ``Code2WavInputPayload``)
    is accepted; anything else is a TypeError, mirroring ``vlm2action``.
    """
    del prompt
    if isinstance(payload, Code2WavInputPayload):
        return payload
    from .stage import TalkerOutput

    if not isinstance(payload, TalkerOutput):
        raise TypeError(f"talker2code2wav expects TalkerOutput, got {type(payload).__name__}")

    audio_codes = _normalise_audio_codes(payload)
    metadata = dict(getattr(payload, "metadata", None) or {})
    sample_rate = metadata.get("sample_rate", MIMI_SAMPLE_RATE)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"MiniMind Code2Wav sample_rate must be a positive integer, got {sample_rate!r}."
        )
    return Code2WavInputPayload(
        audio_codes=audio_codes,
        sample_rate=sample_rate,
        request_id=getattr(payload, "request_id", None),
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
