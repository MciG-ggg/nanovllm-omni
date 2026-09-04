"""Test: OmniRequestOutput surface parity with vllm-omni.

Covers the four methods added to mirror vllm-omni's ``OmniRequestOutput``:

  - ``to_dict()``  -> JSON-serializable dict
  - ``custom_output`` property
  - ``num_images`` property
  - ``from_stage_output(source, **kwargs)`` classmethod
"""

from __future__ import annotations

import pytest

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
    # bytes are serialized to base64 so the dict survives json.dumps.
    import base64
    import json

    assert base64.b64decode(d["multimodal_output"]["audio"]) == b"RIFF...."
    json.dumps(d)  # must not raise on bytes


def test_to_dict_no_payload() -> None:
    out = OmniRequestOutput(request_id="empty")
    d = out.to_dict()
    assert d["request_id"] == "empty"


def test_to_dict_plain_str_value_stays_readable() -> None:
    out = OmniRequestOutput(request_id="r")
    out.multimodal_output = MultimodalPayload.from_dict({"text": "hi"})
    d = out.to_dict()
    assert d["multimodal_output"]["text"] == "hi"
    assert d["multimodal_output"].get("audio") is None


def test_to_dict_custom_output_and_error() -> None:
    out = OmniRequestOutput(request_id="r", error="boom")
    out.custom_output = {"metric": 1}  # type: ignore[attr-defined]
    d = out.to_dict()
    assert d["error"] == "boom"
    assert d["custom_output"] == {"metric": 1}


def test_custom_output_roundtrip() -> None:
    out = OmniRequestOutput(request_id="r")
    out.custom_output = {"foo": "bar"}  # type: ignore[attr-defined]  (property setter)
    assert out.custom_output == {"foo": "bar"}


def test_num_images_default_zero() -> None:
    out = OmniRequestOutput(request_id="r")
    assert out.num_images == 0


def test_num_images_diffusion() -> None:
    out = OmniRequestOutput.from_diffusion([object(), object()], request_id="r")
    assert out.num_images == 2
    assert len(out.images) == 2


def test_output_modality_flags() -> None:
    """TK-018: OutputModality mirrors vllm-omni's flag enum + aliases."""
    from nanovllm_omni.outputs import OutputModality

    assert OutputModality.from_string("audio") == OutputModality.AUDIO
    assert OutputModality.from_string("speech") == OutputModality.AUDIO
    assert OutputModality.from_string("pixels") == OutputModality.IMAGE
    assert OutputModality.from_string("text+image") == (OutputModality.TEXT | OutputModality.IMAGE)
    assert OutputModality.from_string("text,image") == (OutputModality.TEXT | OutputModality.IMAGE)
    assert OutputModality.from_string("") == OutputModality.TEXT
    assert OutputModality.from_string(None) == OutputModality.TEXT
    import pytest

    with pytest.raises(ValueError, match="Unknown modality"):
        OutputModality.from_string("hologram")


def test_to_dict_audio_metadata_emits_sample_rate() -> None:
    """TK-018: audio metadata rides to_dict so the HTTP adapter can read it."""
    out = OmniRequestOutput.from_pipeline(
        AudioPayload(data=b"RIFF....", sample_rate=48000),
        request_id="r1",
        final_output_type="audio",
    )
    d = out.to_dict()
    assert d["multimodal_output"]["audio_metadata"] == {
        "format": "wav",
        "sample_rate": 48000,
    }


def test_from_pipeline_double_tracks_transcript_in_custom_output() -> None:
    """MiniMind-O Q2: ``output.transcript`` from the thinker surfaces as
    ``custom_output["transcript"]`` alongside the ``multimodal_output["audio"]``
    seam, so the ASR side-channel round-trips through ``to_dict()`` without
    polluting the audio field."""

    class _ThinkerAudio:
        audio = AudioPayload(data=b"RIFF....", sample_rate=24000)
        transcript = "你好,我是 MiniMind。"

    out = OmniRequestOutput.from_pipeline(
        _ThinkerAudio(), request_id="r1", final_output_type="audio"
    )
    assert out.custom_output == {"transcript": "你好,我是 MiniMind。"}
    d = out.to_dict()
    assert d["custom_output"] == {"transcript": "你好,我是 MiniMind。"}
    # The audio seam stays untouched -- transcript is a side-channel only.
    import base64

    assert base64.b64decode(d["multimodal_output"]["audio"]) == b"RIFF...."
    assert "audio_metadata" in d["multimodal_output"]


def test_from_pipeline_without_transcript_omits_custom_output() -> None:
    """No ``.transcript`` attr on the source -> ``custom_output`` stays empty
    and ``to_dict()`` does NOT emit a ``custom_output`` key."""

    class _PlainAudio:
        audio = AudioPayload(data=b"RIFF", sample_rate=24000)

    out = OmniRequestOutput.from_pipeline(_PlainAudio(), request_id="r1")
    assert out.custom_output == {}
    assert "custom_output" not in out.to_dict()


def test_from_diffusion_racks_images() -> None:
    """TK-018: images field mirrors vllm-omni's diffusion list slot."""
    imgs = [object(), object(), object()]
    out = OmniRequestOutput.from_diffusion(imgs, request_id="r")
    assert out.images == imgs
    assert out.num_images == 3
    assert out.is_diffusion_output is True


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


# ---------------------------------------------------------------------------
# TK-015: ImageArtifact / TextArtifact round-trips through to_dict()
# ---------------------------------------------------------------------------


def test_text_artifact_roundtrip_with_token_ids() -> None:
    """TextArtifact.text survives to_dict as a plain string;
    ``token_ids`` rides along in the metadata sidecar (TK-015)."""
    from nanovllm_omni.outputs import TextArtifact

    out = OmniRequestOutput(
        multimodal_output=MultimodalPayload.from_dict(
            {"text": TextArtifact(text="hello world", token_ids=[101, 202, 303])}
        )
    )
    d = out.to_dict()
    assert d["multimodal_output"]["text"] == "hello world"
    assert d["multimodal_output"]["text_metadata"] == {"token_ids": [101, 202, 303]}
    import json

    json.dumps(d)  # must not raise


def test_text_artifact_without_token_ids_omits_metadata() -> None:
    """``TextArtifact.token_ids is None`` -> no ``text_metadata`` sidecar."""
    from nanovllm_omni.outputs import TextArtifact

    out = OmniRequestOutput(
        multimodal_output=MultimodalPayload.from_dict({"text": TextArtifact(text="hi")})
    )
    d = out.to_dict()
    assert d["multimodal_output"]["text"] == "hi"
    assert "text_metadata" not in d["multimodal_output"]


def test_image_artifact_roundtrip_emits_png_metadata() -> None:
    """ImageArtifact serializes as base64 PNG + width/height metadata."""

    from nanovllm_omni.outputs import ImageArtifact

    payload = ImageArtifact(png_bytes=b"\x89PNG\r\n\x1a\nfake-bytes", width=64, height=48)
    out = OmniRequestOutput(multimodal_output=MultimodalPayload.from_dict({"image": payload}))
    d = out.to_dict()
    import base64
    import json

    decoded = base64.b64decode(d["multimodal_output"]["image"])
    assert decoded == b"\x89PNG\r\n\x1a\nfake-bytes"
    assert d["multimodal_output"]["image_metadata"] == {
        "format": "png",
        "width": 64,
        "height": 48,
    }
    json.dumps(d)


def test_image_artifact_from_pil_roundtrip() -> None:
    """``ImageArtifact.from_pil`` encodes a PIL image and round-trips dims."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: F401

    from nanovllm_omni.outputs import ImageArtifact

    img = Image.new("RGB", (8, 4), color="red")
    artifact = ImageArtifact.from_pil(img)
    assert artifact.width == 8
    assert artifact.height == 4
    assert artifact.png_bytes.startswith(b"\x89PNG")
    out = OmniRequestOutput(multimodal_output=MultimodalPayload.from_dict({"image": artifact}))
    d = out.to_dict()
    assert d["multimodal_output"]["image_metadata"] == {
        "format": "png",
        "width": 8,
        "height": 4,
    }


def test_image_artifact_post_init_validation() -> None:
    """Width/height must be positive; png_bytes must be non-empty."""
    from nanovllm_omni.outputs import ImageArtifact

    with pytest.raises(ValueError, match="positive"):
        ImageArtifact(png_bytes=b"x", width=0, height=1)
    with pytest.raises(ValueError, match="positive"):
        ImageArtifact(png_bytes=b"x", width=1, height=-2)
    with pytest.raises(ValueError, match="non-empty"):
        ImageArtifact(png_bytes=b"", width=1, height=1)
