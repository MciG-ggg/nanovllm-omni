"""Tests for the local MiniMind-O Code2Wav stage."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest
import torch

from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.models.minimind_omni.code2wav import (
    MiniMindOmniCode2Wav,
    _code2wav_stage,
    decode_audio,
    encode_wav,
)
from nanovllm_omni.models.minimind_omni.stage_processors import Code2WavInputPayload
from nanovllm_omni.outputs import AudioPayload


class FakeMimi:
    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def decode(self, codes: torch.Tensor) -> SimpleNamespace:
        self.input_shape = tuple(codes.shape)
        return SimpleNamespace(audio_values=torch.full((1, 1, codes.shape[-1]), 0.25))


def test_stage_construction() -> None:
    mimi = FakeMimi()
    stage = MiniMindOmniCode2Wav(mimi, "cpu")

    assert stage.mimi is mimi
    assert stage.device == "cpu"
    assert callable(stage)


def test_frame_major_codes_decode_as_batch_codebooks_frames() -> None:
    mimi = FakeMimi()
    stage = MiniMindOmniCode2Wav(mimi, "cpu")

    output = stage(Code2WavInputPayload(torch.arange(24).reshape(3, 8)))

    assert mimi.input_shape == (1, 8, 3)
    assert isinstance(output, AudioPayload)


def test_stage_output_is_mono_24khz_riff_wave() -> None:
    stage = MiniMindOmniCode2Wav(FakeMimi(), "cpu")

    output = stage(Code2WavInputPayload(torch.zeros(2, 8, dtype=torch.long)))

    assert isinstance(output, AudioPayload)
    assert output.sample_rate == 24_000
    assert output.data.startswith(b"RIFF")
    with wave.open(io.BytesIO(output.data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24_000
        assert wav.getsampwidth() == 2


@pytest.mark.parametrize(
    "codes, error",
    [
        (torch.zeros(8), "shape"),
        (torch.zeros(2, 7), "8 codebooks"),
        (torch.zeros(2, 8, 1), "shape"),
    ],
)
def test_stage_rejects_malformed_rank_or_width(codes: torch.Tensor, error: str) -> None:
    stage = MiniMindOmniCode2Wav(FakeMimi(), "cpu")

    with pytest.raises(ValueError, match=error):
        stage(Code2WavInputPayload(codes))


def test_stage_rejects_empty_code_rows() -> None:
    stage = MiniMindOmniCode2Wav(FakeMimi(), "cpu")

    with pytest.raises(ValueError, match="at least one frame"):
        stage(Code2WavInputPayload(torch.empty(0, 8, dtype=torch.long)))


def test_collapsed_audio_and_transcript_wrapper_are_identity() -> None:
    stage = MiniMindOmniCode2Wav(FakeMimi(), "cpu")
    audio = AudioPayload(data=b"RIFF", sample_rate=24_000)
    wrapped = SimpleNamespace(audio=audio, transcript="hello")

    assert stage(audio) is audio
    assert stage(wrapped) is wrapped


def test_stage_rejects_untyped_mapping_instead_of_guessing() -> None:
    stage = MiniMindOmniCode2Wav(FakeMimi(), "cpu")

    with pytest.raises(TypeError, match="Code2WavInputPayload"):
        stage({"codes": {"audio": torch.zeros(1, 8)}})


def test_factory_uses_injected_mimi_without_loading_full_model(monkeypatch) -> None:
    import nanovllm_omni.models.minimind_omni.code2wav as code2wav

    def fail_full_loader(*args, **kwargs):
        raise AssertionError("Code2Wav must not load the full MiniMind bundle")

    monkeypatch.setattr(code2wav, "load_minimind_omni_bundle", fail_full_loader, raising=False)
    mimi = FakeMimi()
    args = OmniEngineArgs(
        model="ignored",
        device="cpu",
        dtype="float32",
        extra={"mimi": mimi},
    )

    stage = _code2wav_stage(SimpleNamespace(), args)

    assert isinstance(stage, MiniMindOmniCode2Wav)
    assert stage.mimi is mimi
    assert stage.device == "cpu"


def test_existing_codec_helpers_remain_usable() -> None:
    mimi = FakeMimi()
    samples = decode_audio(mimi, [[1] * 8, [2] * 8], "cpu")
    wav_bytes = encode_wav(samples)

    assert mimi.input_shape == (1, 8, 2)
    assert wav_bytes.startswith(b"RIFF")
