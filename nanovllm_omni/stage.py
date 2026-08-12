"""Stage contract and deterministic fake MiniMind-Omni stages for Issue #2."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sin, tau
from typing import Generic, TypeVar

from nanovllm_omni.payloads import (
    AudioPayload,
    BridgePayload,
    CodecTokenPayload,
    TensorPayload,
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


@dataclass
class FakeThinker(Stage[str, BridgePayload]):
    """Deterministically turns prompt text into token and bridge payloads."""

    name: str = "thinker"

    def execute(self, payload: str) -> BridgePayload:
        token_ids = tuple(payload.encode("utf-8")) or (0,)
        return BridgePayload(
            tokens=TokenPayload(token_ids=token_ids, text=payload),
            hidden_states=TensorPayload(
                values=tuple(token / 255 for token in token_ids), shape=(len(token_ids), 1)
            ),
        )


@dataclass
class FakeTalker(Stage[BridgePayload, CodecTokenPayload]):
    """Maps bridge states to stable, Mimi-like codec token IDs."""

    name: str = "talker"

    def execute(self, payload: BridgePayload) -> CodecTokenPayload:
        return CodecTokenPayload(
            token_ids=tuple(round(value * 255) for value in payload.hidden_states.values),
            codebooks=1,
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
