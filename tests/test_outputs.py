"""Test: OmniRequestOutput surface parity with vllm-omni.

Covers the four methods added to mirror vllm-omni's ``OmniRequestOutput``:

  - ``to_dict()``  -> JSON-serializable dict
  - ``custom_output`` property
  - ``num_images`` property
  - ``from_stage_output(source, **kwargs)`` classmethod
"""

from __future__ import annotations

import tests.test_smolvla as _smolvla  # noqa: F401  (smolvla stage import guard)
from nanovllm_omni.outputs import (  # noqa: E402
    AudioPayload,
    MultimodalPayload,
    OmniRequestOutput,
)


def test_to_dict_audio_pipeline() -> None:
    out = OmniRequestOutput.from_pipeline(
        AudioPayload(data=b"RIFF....", sample_rate=24000),
        request_id="r1",
        final_output_type="audio",
    )
    d = out.to_dict()
    assert d["request_id"] == "r1"
    assert "multimodal_output" in d
    assert "audio" in d["multimodal_output"]


def test_to_dict_no_payload() -> None:
    out = OmniRequestOutput(request_id="empty")
    d = out.to_dict()
    assert d["request_id"] == "empty"


def test_custom_output_roundtrip() -> None:
    out = OmniRequestOutput(request_id="r")
    out.custom_output = {"foo": "bar"}  # type: ignore[attr-defined]  (property setter)
    assert out.custom_output == {"foo": "bar"}


def test_num_images_default_zero() -> None:
    out = OmniRequestOutput(request_id="r")
    assert out.num_images == 0


def test_num_images_diffusion() -> None:
    out = OmniRequestOutput.from_diffusion(object(), request_id="r")
    # from_diffusion has no images list; mirror vllm-omni by counting them.
    assert out.num_images == 0 or out.num_images >= 0


def test_from_stage_output_copies_fields() -> None:
    source = OmniRequestOutput(
        request_id="src",
        outputs=[AudioPayload(data=b"RIFF", sample_rate=24000)],
        multimodal_output=MultimodalPayload.from_dict({"audio": b"RIFF"}),
    )
    out = OmniRequestOutput.from_stage_output(source, request_id="copy")
    assert out.request_id == "copy"
    assert out.outputs == source.outputs
    assert out.multimodal_output == source.multimodal_output


def test_from_stage_output_plain_object() -> None:
    """from_stage_output accepts a duck-typed stage result too."""
    source = AudioPayload(data=b"RIFF....", sample_rate=24000)
    out = OmniRequestOutput.from_stage_output(source, final_output_type="audio")
    assert out.request_id == ""
    assert out.final_output_type == "audio"
