"""MiniMind-O audio input + ASR helpers.

Two SenseVoice-backed capabilities: (1) audio understanding via the
thinker's ``forward(audio_inputs=..., audio_lens=...)`` prefill path,
and (2) ASR transcript surfaced as ``custom_output["transcript"]`` on
the ``OmniRequestOutput``. Everything funasr-related is imported lazily
inside ``SenseVoice.load`` so a venv without funasr (CI) can import this
module and execute all audio-free paths.

Public symbols: ``SenseVoice``, ``attach_audio_encoder``, ``load_audio``,
``SENSEVOICE_SAMPLE_RATE``, ``AUDIO_MARKER_TOKEN``.
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SENSEVOICE_SAMPLE_RATE = 16_000
# <|audio_pad|> token repeated per injected audio feature frame.
AUDIO_MARKER_TOKEN = "<|audio_pad|>"


@dataclass
class SenseVoice:
    """Loaded funasr SenseVoice handles + frontend (lazy, one per process)."""

    asr: Any  # full funasr model: ``.generate(input=...)`` -> transcription
    encoder: Any  # audio feature encoder; fed to the MiniMind thinker
    frontend: Any  # funasr frontend: (wav, nsamples) -> (fbank, frame_lens)

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> SenseVoice:
        """Build a SenseVoice handle from a local checkpoint directory.

        Keeps the full funasr model alive so the same handle can also
        transcribe (double-track ASR).
        """
        import contextlib

        from funasr import AutoModel

        with contextlib.redirect_stderr(io.StringIO()):
            m = AutoModel(
                model=model_path,
                trust_remote_code=True,
                disable_update=True,
                device="cpu",
            )
        frontend = m.kwargs["frontend"]
        encoder = m.model.encoder
        for p in encoder.parameters():
            p.requires_grad = False
        sv = cls(asr=m, encoder=encoder.eval().float(), frontend=frontend.eval())
        if device not in ("cpu", "mps"):
            sv.encoder = sv.encoder.to(device)
        return sv

    def fbank(self, samples: Any) -> tuple[Any, Any, int]:
        """(audio_inputs [1, T, F], audio_lens [1], n_frames) for engine prefill.

        ``samples``: float32 1-D mono numpy at ``SENSEVOICE_SAMPLE_RATE``.
        """
        import torch

        wav = torch.from_numpy(samples).float() if not torch.is_tensor(samples) else samples.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        n_frames = int(fbank.size(1))
        audio_lens = torch.as_tensor(flen, dtype=torch.long).reshape(-1)
        return fbank, audio_lens, n_frames

    def transcribe(self, samples: Any) -> str:
        """ASR the speech to text; ``''`` on failure or empty input."""
        if self.asr is None:
            return ""
        try:
            result = self.asr.generate(input=samples, cache={}, language="auto", use_itn=True)
        except Exception:  # funasr backends vary; transcription is best-effort
            return ""
        if not result or not result[0].get("text"):
            return ""
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            return rich_transcription_postprocess(result[0]["text"])
        except Exception:
            return str(result[0]["text"])


def attach_audio_encoder(bundle: Any, audio_encoder_path: str | None) -> Any:
    """Load SenseVoice once onto ``bundle`` and return its ``SenseVoice``.

    The loaded encoder is wired as ``bundle.model.audio_encoder`` so the
    model's own ``encode_audio_inputs`` / ``inject_audio_features`` prefill
    path can consume it. Idempotent.
    """
    sv = getattr(bundle, "audio_sensevoice", None)
    if sv is not None:
        return sv
    if not audio_encoder_path or not os.path.isdir(str(audio_encoder_path)):
        raise ValueError(
            "audio input needs a SenseVoice checkpoint directory; pass "
            f"'audio_encoder_path' (default pretrained/SenseVoiceSmall), got "
            f"{audio_encoder_path!r}"
        )
    device = getattr(bundle, "device", "cpu")
    sv = SenseVoice.load(str(audio_encoder_path), device=device)
    bundle.model.audio_encoder = sv.encoder
    bundle.audio_sensevoice = sv
    return sv


def load_audio(audio: Any) -> Any:
    """Normalise audio input to float32 mono numpy at ``SENSEVOICE_SAMPLE_RATE``.

    Accepts: numpy array (assumed 16 kHz mono), wav ``bytes``/``bytearray``,
    or a path to a wav file. Non-16 kHz sources are linearly resampled.
    """
    import numpy as np

    if isinstance(audio, (str, os.PathLike)):
        raw = Path(audio).read_bytes()
    elif isinstance(audio, (bytes, bytearray)):
        raw = bytes(audio)
    else:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32)
    with wave.open(io.BytesIO(raw)) as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[sample_width]
    arr = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / scale
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return _resample(arr, rate, SENSEVOICE_SAMPLE_RATE)


def _resample(x: Any, src_rate: int, dst_rate: int) -> Any:
    """Naive linear resample. Fine for a demo; swap in torchaudio
    (clean up-sample) if quality ever matters."""
    import numpy as np

    if src_rate == dst_rate:
        return x.astype(np.float32)
    n = int(round(x.shape[0] * dst_rate / src_rate))
    return np.interp(np.linspace(0, x.shape[0] - 1, n), np.arange(x.shape[0]), x).astype(np.float32)


__all__ = [
    "AUDIO_MARKER_TOKEN",
    "SENSEVOICE_SAMPLE_RATE",
    "SenseVoice",
    "attach_audio_encoder",
    "load_audio",
]
