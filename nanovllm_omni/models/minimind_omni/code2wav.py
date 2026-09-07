"""MiniMind-O Code2Wav stage and Mimi codec helpers.

Public symbols: ``MiniMindOmniCode2Wav``, ``load_mimi_codec``,
``decode_audio``, ``encode_wav``.
"""

from __future__ import annotations

import io
import wave
from typing import Any

from nanovllm_omni.outputs import AudioPayload

from ._stage import stage
from .bundle import (
    DEFAULT_MIMI_MODEL_ID,
    MIMI_CODE_VOCAB_LIMIT,
)

MIMI_NUM_CODEBOOKS = 8


def load_mimi_codec(
    model_id: str = DEFAULT_MIMI_MODEL_ID,
    device: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = True,
) -> Any:
    """Load only the Mimi codec used by the Code2Wav stage."""
    from transformers import MimiModel

    from .bundle import _cast_model_dtype, _pick_device, _resolve_snapshot

    device = _pick_device(device)
    mimi = MimiModel.from_pretrained(
        _resolve_snapshot(model_id),
        trust_remote_code=trust_remote_code,
    ).eval()
    return _cast_model_dtype(mimi, dtype, device).to(device)


def _is_code2wav_payload(payload: Any) -> bool:
    from .stage_processors import Code2WavInputPayload

    return isinstance(payload, Code2WavInputPayload)


class MiniMindOmniCode2Wav:
    """Decode one full-mode ``Code2WavInputPayload`` into ``AudioPayload``."""

    def __init__(self, mimi: Any, device: str | Any) -> None:
        self.mimi = mimi
        self.device = device

    def __call__(self, payload: Any, sampling: Any = None) -> Any:
        del sampling
        from .stage_processors import Code2WavInputPayload

        if not isinstance(payload, Code2WavInputPayload):
            raise TypeError(
                "MiniMind Code2Wav expected Code2WavInputPayload, " f"got {type(payload).__name__}."
            )
        import torch

        audio_codes = payload.audio_codes
        if not isinstance(audio_codes, torch.Tensor):
            raise TypeError(
                "MiniMind Code2Wav audio_codes must be a torch.Tensor, "
                f"got {type(audio_codes).__name__}."
            )
        if audio_codes.ndim != 2:
            raise ValueError(
                "MiniMind Code2Wav audio_codes must have shape [frames, codebooks], "
                f"got {tuple(audio_codes.shape)}."
            )
        if audio_codes.shape[1] != MIMI_NUM_CODEBOOKS:
            raise ValueError(
                "MiniMind Code2Wav audio_codes must have 8 codebooks, "
                f"got width {audio_codes.shape[1]}."
            )
        if audio_codes.shape[0] == 0:
            raise ValueError("MiniMind Code2Wav audio_codes must contain at least one frame.")
        if isinstance(payload.sample_rate, bool) or not isinstance(payload.sample_rate, int):
            raise ValueError(
                "MiniMind Code2Wav sample_rate must be a positive integer, "
                f"got {payload.sample_rate!r}."
            )
        if payload.sample_rate <= 0:
            raise ValueError(
                "MiniMind Code2Wav sample_rate must be a positive integer, "
                f"got {payload.sample_rate!r}."
            )

        # decode_audio owns the Mimi convention: frame-major [F, C] becomes
        # decoder input [1, C, F] before invalid vocabulary ids are filtered.
        audio_frames = audio_codes.detach().to(dtype=torch.long).cpu().tolist()
        samples = decode_audio(self.mimi, audio_frames, self.device)
        return AudioPayload(
            data=encode_wav(samples, sample_rate=payload.sample_rate),
            sample_rate=payload.sample_rate,
        )


def _code2wav_stage(deploy: Any, args: Any) -> MiniMindOmniCode2Wav:
    """Construct the local Code2Wav stage without loading the full model."""
    del deploy
    from .bundle import _cast_model_dtype, _pick_device

    extra = dict(getattr(args, "extra", None) or {})
    supplied_mimi = extra.get("mimi")
    device = _pick_device(getattr(args, "device", None))
    dtype = getattr(args, "dtype", None)
    if supplied_mimi is None or isinstance(supplied_mimi, str):
        model_id = extra.get("mimi_model_id") or supplied_mimi or DEFAULT_MIMI_MODEL_ID
        mimi = load_mimi_codec(
            model_id=model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=getattr(args, "trust_remote_code", True),
        )
    else:
        mimi = supplied_mimi.eval() if hasattr(supplied_mimi, "eval") else supplied_mimi
        mimi = _cast_model_dtype(mimi, dtype, device)
        mimi = mimi.to(device) if hasattr(mimi, "to") else mimi
    return MiniMindOmniCode2Wav(mimi, device)


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

    with stage("decode"):
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

    with stage("wav"):
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


__all__ = [
    "MiniMindOmniCode2Wav",
    "_code2wav_stage",
    "decode_audio",
    "encode_wav",
    "load_mimi_codec",
]
