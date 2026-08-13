"""Stage contract and deterministic fake MiniMind-Omni stages.

Issue #2 introduced the three-stage tracer bullet; issue #3 layers the
MiniMind-Omni post-EOS and Talker MTP semantics on top:

* ``FakeThinker.execute`` returns a ``ThinkerRun`` containing the visible
  step plus ``forced_padding_count`` forced steps, matching the reference
  behavior of 128 post-EOS bridge states by default.
* ``FakeTalker.execute`` consumes the aggregated ``ThinkerRun``, applies
  the delayed MTP codebook active mask ``mask[t][k] = k <= t``, and fills
  inactive codebook positions with ``AUDIO_PADDING_TOKEN_ID``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sin, tau
from typing import Generic, TypeVar

from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    THINKER_FORCED_PADDING_DEFAULT,
    AudioPayload,
    BridgePayload,
    CodecTokenPayload,
    TensorPayload,
    ThinkerRun,
    TokenPayload,
)

Input = TypeVar("Input")
Output = TypeVar("Output")


class Stage(ABC, Generic[Input, Output]):
    """A named synchronous processing boundary."""

    name: str

    @abstractmethod
    def execute(self, payload: Input) -> Output:
        """Transform one typed payload into the next."""


def _build_active_mask(frames: int, codebooks: int) -> tuple[tuple[bool, ...], ...]:
    """Delayed MTP codebook activation mask.

    At frame ``t`` codebook ``k`` is active iff ``k <= t`` (saturating once
    ``t >= codebooks``). The first codebook is "primary" (predicted at
    every step); subsequent codebooks become active one step at a time.
    """
    return tuple(tuple(k <= t for k in range(codebooks)) for t in range(frames))


@dataclass
class FakeThinker(Stage[str, ThinkerRun]):
    """Deterministically turns prompt text into a visible step plus forced steps.

    The visible step encodes the prompt bytes and ends with ``eos_token_id``.
    ``forced_padding_count`` additional bridge states follow, each tagged so
    the post-EOS state machine can observe every forced step.
    """

    name: str = "thinker"
    forced_padding_count: int = THINKER_FORCED_PADDING_DEFAULT
    eos_token_id: int = AUDIO_PADDING_TOKEN_ID

    def execute(self, payload: str) -> ThinkerRun:
        prompt_ids = tuple(payload.encode("utf-8")) or (0,)
        visible_ids = prompt_ids + (self.eos_token_id,)
        visible_bridge = BridgePayload(
            tokens=TokenPayload(token_ids=visible_ids, text=payload),
            hidden_states=TensorPayload(
                values=tuple(token / 255 for token in visible_ids),
                shape=(len(visible_ids), 1),
            ),
        )
        forced_bridges: list[BridgePayload] = []
        for step in range(self.forced_padding_count):
            forced_bridges.append(
                BridgePayload(
                    tokens=TokenPayload(
                        token_ids=(self.eos_token_id,),
                        text="",
                        metadata={"forced": "true", "step": str(step)},
                    ),
                    hidden_states=TensorPayload(
                        values=((self.eos_token_id + step + 1) / 255,),
                        shape=(1, 1),
                    ),
                )
            )
        return ThinkerRun(
            bridges=(visible_bridge, *forced_bridges),
            visible_tokens=visible_bridge.tokens,
            eos_token_id=self.eos_token_id,
            forced_padding_count=self.forced_padding_count,
        )


@dataclass
class FakeTalker(Stage[ThinkerRun, CodecTokenPayload]):
    """Maps bridge states to Mimi-like codec tokens with the delayed MTP mask.

    For every bridge frame the Talker emits ``codebooks`` codec tokens.
    Active positions are derived from the bridge hidden state; inactive
    positions receive ``AUDIO_PADDING_TOKEN_ID`` so the codec decode step
    can skip them.
    """

    name: str = "talker"
    codebooks: int = 4

    def execute(self, payload: ThinkerRun) -> CodecTokenPayload:
        frames = len(payload.bridges)
        active_mask = _build_active_mask(frames, self.codebooks)
        token_ids: list[int] = []
        for frame_idx, bridge in enumerate(payload.bridges):
            bridge_seed = sum(bridge.hidden_states.values) or 1
            for codebook_idx in range(self.codebooks):
                if active_mask[frame_idx][codebook_idx]:
                    token_ids.append((bridge_seed + frame_idx + codebook_idx) % 256)
                else:
                    token_ids.append(AUDIO_PADDING_TOKEN_ID)
        return CodecTokenPayload(
            token_ids=tuple(token_ids),
            codebooks=self.codebooks,
            active_mask=active_mask,
        )


@dataclass
class FakeCode2Wav(Stage[CodecTokenPayload, AudioPayload]):
    """Creates a deterministic playable 0.1 second waveform without a codec model."""

    name: str = "code2wav"
    duration_seconds: float = 0.1

    def execute(self, payload: CodecTokenPayload) -> AudioPayload:
        seed = sum(payload.token_ids) or 1
        frames = round(payload.sample_rate * self.duration_seconds)
        frequency = 220 + seed % 440
        samples = tuple(
            0.2 * sin(tau * frequency * frame / payload.sample_rate) for frame in range(frames)
        )
        return AudioPayload(
            samples=samples,
            sample_rate=payload.sample_rate,
            metadata={"format": "pcm_s16le", "source": "fake-code2wav"},
        )
