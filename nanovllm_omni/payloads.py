"""Typed values passed between stages in the model-free demo."""

from dataclasses import dataclass, field
from io import BytesIO
from struct import pack
from wave import open as wave_open

# Reference MiniMind-Omni post-EOS forced-padding count. After visible EOS
# the Thinker runs this many additional steps so the Talker receives enough
# bridge states to produce a full audio frame. Reference: vLLM-Omni PR #3796.
THINKER_FORCED_PADDING_DEFAULT = 128

# Audio padding token emitted by the Talker MTP codebook mask. Inactive
# codebook positions are filled with this id before the next decode step.
AUDIO_PADDING_TOKEN_ID = 0


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
    """One per-step Thinker output: tokens and hidden states consumed by the Talker."""

    tokens: TokenPayload
    hidden_states: TensorPayload


@dataclass(frozen=True)
class ThinkerRun:
    """Per-request Thinker output: the visible step plus forced padding steps.

    ``bridges[0]`` is the visible step (ending in EOS). ``bridges[1:N+1]``
    are the ``forced_padding_count`` forced steps emitted after EOS to keep
    the Talker supplied with bridge states. Total length is
    ``1 + forced_padding_count``.
    """

    bridges: tuple[BridgePayload, ...]
    visible_tokens: TokenPayload
    eos_token_id: int
    forced_padding_count: int

    def __post_init__(self) -> None:
        expected = self.forced_padding_count + 1
        if len(self.bridges) != expected:
            raise ValueError(
                f"ThinkerRun bridges length {len(self.bridges)} does not match "
                f"1 + forced_padding_count ({expected})"
            )


@dataclass(frozen=True)
class CodecTokenPayload:
    """Mimi-like codec tokens emitted by the Talker.

    ``active_mask`` records which codebook positions are real (not
    padding) at each frame, in frame-major order. Inactive positions
    carry ``AUDIO_PADDING_TOKEN_ID``.
    """

    token_ids: tuple[int, ...]
    codebooks: int
    active_mask: tuple[tuple[bool, ...], ...] = ()
    sample_rate: int = 8_000

    def __post_init__(self) -> None:
        if self.codebooks < 1:
            raise ValueError("CodecTokenPayload codebooks must be at least 1")
        if self.active_mask and len(self.token_ids) != len(self.active_mask) * self.codebooks:
            raise ValueError(
                f"CodecTokenPayload token_ids length {len(self.token_ids)} does not match "
                f"len(active_mask) * codebooks ({len(self.active_mask) * self.codebooks})"
            )


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
