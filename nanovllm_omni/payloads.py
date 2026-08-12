"""Typed values passed between stages in the model-free demo."""

from dataclasses import dataclass, field
from io import BytesIO
from struct import pack
from wave import open as wave_open


@dataclass(frozen=True)
class TokenPayload:
    """Visible token IDs and the text they encode."""

    token_ids: tuple[int, ...]
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorPayload:
    """Small CPU-only stand-in for a hidden-state tensor."""

    values: tuple[float, ...]
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class BridgePayload:
    """Thinker's tokens and hidden states consumed by the Talker."""

    tokens: TokenPayload
    hidden_states: TensorPayload


@dataclass(frozen=True)
class CodecTokenPayload:
    """Mimi-like codec tokens emitted by the Talker."""

    token_ids: tuple[int, ...]
    codebooks: int
    sample_rate: int = 8_000


@dataclass(frozen=True)
class AudioPayload:
    """Mono floating-point waveform plus playback metadata."""

    samples: tuple[float, ...]
    sample_rate: int
    metadata: dict[str, str] = field(default_factory=dict)

    def wav_bytes(self) -> bytes:
        """Encode the waveform as 16-bit PCM WAV with only the stdlib."""
        frames = b"".join(
            pack("<h", max(-32768, min(32767, round(sample * 32767)))) for sample in self.samples
        )
        buffer = BytesIO()
        with wave_open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(frames)
        return buffer.getvalue()
