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
    # ponytail: a parallel ``_FakeMimi`` lives in tests/test_optim_bench.py;
    # merge on third user.
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


def test_thinker2talker_rejects_non_strong_type():
    """C4: thinker2talker 只收 ThinkerStageOutput，其余 TypeError。"""
    import pytest

    from nanovllm_omni.models.minimind_omni.stage_processors import thinker2talker

    with pytest.raises(TypeError, match="ThinkerStageOutput"):
        thinker2talker({"bridge_states": None}, prompt="x")
    with pytest.raises(TypeError, match="ThinkerStageOutput"):
        thinker2talker("raw string", prompt="x")


def test_thinker2talker_computes_start_pos_once():
    """C4: start_pos/num_steps 由 thinker2talker 一处计算进 metadata。"""
    import torch

    from nanovllm_omni.models.minimind_omni.stage_processors import (
        ThinkerStageOutput,
        thinker2talker,
    )

    bridge = torch.randn(5, 8)
    out = thinker2talker(
        ThinkerStageOutput(
            bridge_states=bridge,
            prompt_token_ids=(1, 2),
            output_token_ids=(3, 4, 5),
            text_token_ids=(1, 2, 3, 4, 5),
        )
    )
    assert out.metadata["start_pos"] == 2
    assert out.metadata["num_steps"] == 3
    assert out.bridge_states.shape[0] == 5


def test_talker2code2wav_rejects_non_talker_output():
    """C4: talker2code2wav 只收 TalkerOutput，其余 TypeError。"""
    import pytest

    from nanovllm_omni.models.minimind_omni.stage_processors import talker2code2wav

    with pytest.raises(TypeError, match="TalkerOutput"):
        talker2code2wav({"audio_codes": None}, prompt="x")
