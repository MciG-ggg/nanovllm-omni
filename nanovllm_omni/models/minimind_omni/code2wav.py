"""MiniMind-O code2wav (Mimi codec) stage.

Stage 2 of the 3-stage pipeline. Owns the codec-decode path
(``decode_audio`` → Mimi → float numpy) and the WAV byte serializer
(``encode_wav`` → stdlib ``wave``).

For TICKET-02 this is an identity pass-through because the thinker's
end-to-end ``generate_audio`` already produces an ``AudioPayload`` and
the codec decode happens inside the thinker glue layer via the helpers
exported below. TICKET-05 / TK-005 may move the actual decode here once
the 3-stage split is real.
"""

from __future__ import annotations

import io
import wave
from typing import Any

from .bundle import MIMI_CODE_VOCAB_LIMIT


def _code2wav_stage(deploy: Any, args: Any) -> Any:
    """Stage 2 factory: identity pass-through for TICKET-02.

    The end-to-end ``generate_audio`` already returns ``AudioPayload``; the
    codec decode happened inside the thinker glue layer.
    """

    def code2wav_forward(payload: Any, sampling: Any) -> Any:
        return payload

    return code2wav_forward


def decode_audio(
    mimi: Any,
    audio_frames: list[list[int]],
    device: str,
) -> Any:
    """Decode collected Mimi codebook frames to a float numpy array on CPU.

    Labeled ``decode`` for the benchmark harness. Moves codes to ``device``,
    runs ``mimi.decode`` under no_grad, then returns ``np.ndarray``.
    """
    import torch

    with torch.profiler.record_function("decode"):
        codes = torch.tensor(audio_frames, dtype=torch.long, device=device).T.unsqueeze(0)
        filtered = torch.where(codes >= MIMI_CODE_VOCAB_LIMIT, torch.zeros_like(codes), codes)
        with torch.no_grad():
            audio = mimi.decode(filtered).audio_values
        return audio.squeeze().float().cpu().numpy()


def encode_wav(samples: Any, sample_rate: int = 24_000) -> bytes:
    """Wrap float audio into a 16-bit mono PCM WAV byte string.

    Labeled ``wav`` for the benchmark harness. Uses stdlib ``wave`` only;
    accepts anything numpy can coerce to float32.
    """
    import numpy as np
    import torch

    with torch.profiler.record_function("wav"):
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return out.getvalue()


__all__ = ["_code2wav_stage", "decode_audio", "encode_wav"]
