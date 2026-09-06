"""Pure MiniMind-O stage handoff tests; no checkpoint or model execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nanovllm_omni.models.minimind_omni.pipeline import MINIMIND_OMNI_PIPELINE
from nanovllm_omni.models.minimind_omni.stage_processors import (
    AUDIO_PAD_TOKEN_ID,
    Code2WavInputPayload,
    TalkerInputPayload,
    ThinkerStageOutput,
    talker2code2wav,
    thinker2talker,
)
from nanovllm_omni.models.minimind_omni.talker import TalkerOutput
from nanovllm_omni.outputs import AudioPayload


def test_thinker2talker_preserves_text_speaker_and_metadata() -> None:
    source = ThinkerStageOutput(
        bridge_states=torch.arange(28, dtype=torch.float16).reshape(7, 4),
        prompt_token_ids=[10, 11],
        output_token_ids=[12, 13, 14],
        speaker_embedding=torch.ones(1, 3),
        text_state={"finished": True},
        request_id="req-1",
        metadata={"sample_rate": 48_000, "device_name": "cpu"},
    )

    result = thinker2talker(source, "unused prompt")

    assert isinstance(result, TalkerInputPayload)
    assert result.input_ids.tolist() == [AUDIO_PAD_TOKEN_ID, AUDIO_PAD_TOKEN_ID]
    assert result.bridge_states.shape == (5, 4)
    assert result.bridge_states.dtype == torch.float32
    assert result.text_token_ids == (10, 11, 12, 13, 14)
    assert result.prompt_token_ids == (10, 11)
    assert result.output_token_ids == (12, 13, 14)
    assert torch.equal(result.speaker_embedding, source.speaker_embedding)
    assert result.request_id == "req-1"
    assert result.metadata["sample_rate"] == 48_000
    assert result.metadata["text_state"] == {"finished": True}
    assert result.additional_information["ids"]["all"] == [10, 11, 12, 13, 14]


def test_thinker2talker_accepts_stage_mapping_and_aligns_bridge() -> None:
    source = {
        "bridge_states": torch.ones(1, 5, 3),
        "prompt_token_ids": torch.tensor([20, 21]),
        "output_token_ids": torch.tensor([22]),
        "request_id": "req-map",
    }

    result = thinker2talker(source)

    assert result.request_id == "req-map"
    assert result.bridge_states.shape == (3, 3)
    assert result.text_token_ids == (20, 21, 22)


@pytest.mark.parametrize(
    "bridge",
    [None, torch.empty(0, 4), torch.empty(3, 0)],
)
def test_thinker2talker_rejects_missing_or_empty_bridge(bridge: torch.Tensor | None) -> None:
    source = {
        "hidden_states": {"bridge": bridge},
        "ids": {"prompt": [1], "output": [2]},
    }

    with pytest.raises(ValueError, match="bridge"):
        thinker2talker(source)


def test_thinker2talker_rejects_bridge_shorter_than_text() -> None:
    source = ThinkerStageOutput(
        bridge_states=torch.zeros(1, 2),
        prompt_token_ids=[1, 2],
        output_token_ids=[3],
    )

    with pytest.raises(ValueError, match="shorter than text"):
        thinker2talker(source)


def test_talker2code2wav_normalizes_codes_and_preserves_metadata() -> None:
    codes = torch.tensor([[1.2, 2.8], [3.0, 4.0]])
    source = SimpleNamespace(
        request_id="req-2",
        multimodal_outputs={"codes": {"audio": codes}},
        metadata={"sample_rate": 16_000, "channel": "mono"},
    )

    result = talker2code2wav(source)

    assert isinstance(result, Code2WavInputPayload)
    assert result.audio_codes.tolist() == [[1, 2], [3, 4]]
    assert result.audio_codes.shape == (2, 2)
    assert result.sample_rate == 16_000
    assert result.request_id == "req-2"
    assert result.device == codes.device
    assert result.metadata["channel"] == "mono"
    assert torch.equal(result.codes["audio"], result.audio_codes)


def test_talker2code2wav_accepts_local_talker_output() -> None:
    source = TalkerOutput(
        text_hidden_states=torch.zeros(2, 3),
        multimodal_outputs={"codes": {"audio": torch.ones(2, 8, dtype=torch.long)}},
    )

    result = talker2code2wav(source)

    assert result.audio_codes.shape == (2, 8)
    assert result.sample_rate == 24_000


@pytest.mark.parametrize(
    "codes, error",
    [(torch.ones(2, 3, 4), "shape"), (torch.empty(2, 0), "non-empty"), ([], "tensor")],
)
def test_talker2code2wav_rejects_malformed_codes(codes: object, error: str) -> None:
    source = {"codes": {"audio": codes}, "request_id": "bad"}

    with pytest.raises((TypeError, ValueError), match=error):
        talker2code2wav(source)


def test_collapsed_audio_is_passed_through_by_identity() -> None:
    audio = AudioPayload(data=b"RIFF", sample_rate=24_000)
    wrapped = SimpleNamespace(audio=audio, transcript="hello")

    assert thinker2talker(audio) is audio
    assert talker2code2wav(audio) is audio
    assert thinker2talker(wrapped) is wrapped
    assert talker2code2wav(wrapped) is wrapped


def test_processors_do_not_execute_a_model() -> None:
    class NoCall:
        def __call__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("stage processor called a model")

    source = ThinkerStageOutput(
        bridge_states=torch.zeros(2, 2),
        prompt_token_ids=[1, 2],
        text_state=NoCall(),
    )
    result = thinker2talker(source)
    assert isinstance(result.metadata["text_state"], NoCall)


def test_builtin_pipeline_uses_real_processors_and_keeps_three_stages() -> None:
    assert [stage.name for stage in MINIMIND_OMNI_PIPELINE.stages] == [
        "thinker",
        "talker",
        "code2wav",
    ]
    assert MINIMIND_OMNI_PIPELINE.stages[1].process_input.endswith(
        "stage_processors:thinker2talker"
    )
    assert MINIMIND_OMNI_PIPELINE.stages[2].process_input.endswith(
        "stage_processors:talker2code2wav"
    )
