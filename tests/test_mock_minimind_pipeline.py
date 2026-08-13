"""CPU-only contract test for the model-free MiniMind-Omni pipeline.

Updated for issue #3: the Thinker now returns a ``ThinkerRun`` carrying
visible + forced bridges and the Talker returns a ``CodecTokenPayload``
carrying the MTP codebook active mask.
"""

from pathlib import Path
from wave import open as wave_open

from nanovllm_omni.config import load_config
from nanovllm_omni.orchestrator import Orchestrator
from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    AudioPayload,
    CodecTokenPayload,
    ThinkerRun,
)
from nanovllm_omni.pipeline import Pipeline
from nanovllm_omni.stage import FakeCode2Wav, FakeTalker, FakeThinker


def make_pipeline() -> Pipeline:
    return Pipeline((FakeThinker(), FakeTalker(), FakeCode2Wav()))


def test_mock_config_declares_the_three_ordered_stages() -> None:
    path = Path(__file__).parents[1] / "configs" / "mock_minimind_omni.yaml"
    pipeline, deploy = load_config(path)

    assert [(stage.name, stage.kind) for stage in pipeline.stages] == [
        ("thinker", "ar"),
        ("talker", "ar"),
        ("code2wav", "audio_decode"),
    ]
    assert [(edge.source, edge.target) for edge in pipeline.connectors] == [
        ("thinker", "talker"),
        ("talker", "code2wav"),
    ]
    assert deploy.device == "cpu"


def test_mock_minimind_request_traverses_typed_stages_and_returns_playable_audio() -> None:
    result = Orchestrator().submit(make_pipeline(), "hello omni")

    assert result.stage_names == ("thinker", "talker", "code2wav")
    assert isinstance(result.audio, AudioPayload)
    assert result.audio.sample_rate == 8_000
    assert result.audio.metadata == {"format": "pcm_s16le", "source": "fake-code2wav"}
    assert len(result.audio.samples) == 800

    with wave_open(__import__("io").BytesIO(result.audio.wav_bytes())) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 8_000, 800)


def test_stage_boundaries_expose_required_typed_payloads() -> None:
    _, trace = make_pipeline().run("hello")

    assert isinstance(trace[0][1], ThinkerRun)
    assert trace[0][1].visible_tokens.text == "hello"
    assert trace[0][1].forced_padding_count == 128
    assert len(trace[0][1].bridges) == trace[0][1].forced_padding_count + 1
    assert isinstance(trace[1][1], CodecTokenPayload)
    assert trace[1][1].codebooks >= 1
    assert trace[1][1].active_mask
    assert isinstance(trace[2][1], AudioPayload)
    assert AUDIO_PADDING_TOKEN_ID in trace[1][1].token_ids


def test_mock_audio_is_deterministic() -> None:
    orchestrator = Orchestrator()

    first = orchestrator.submit(make_pipeline(), "same request").audio
    second = orchestrator.submit(make_pipeline(), "same request").audio

    assert first == second
