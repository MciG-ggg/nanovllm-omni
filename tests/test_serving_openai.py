"""Contract tests for the stdlib HTTP adapter's request parser.

The parser in ``nanovllm_omni.serving.openai_adapter`` is the gatekeeper
for ``POST /v1/chat/completions``; the OpenAI multimodal content array
is the only accepted ``messages[i].content`` shape. These tests pin
that contract.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.serving.openai_adapter import _chat_completion, _extract_text


def test_chat_completion_reads_base64_audio_from_payload() -> None:
    """The OpenAI envelope consumes ``to_dict()``'s payload, not an AudioPayload."""
    import base64
    import json

    payload = {
        "request_id": "r1",
        "final_output_type": "audio",
        "multimodal_output": {"audio": base64.b64encode(b"RIFF....").decode("ascii")},
    }
    body = _chat_completion(payload, "minimind-o", prompt_tokens=3)
    # Round-trips through the same base64 the adapter produces.
    data = json.loads(json.dumps(body))["choices"][0]["message"]["audio"]["data"]
    assert base64.b64decode(data) == b"RIFF...."
    assert body["choices"][0]["message"]["audio"]["format"] == "wav"
    assert body["choices"][0]["message"]["audio"]["sample_rate"] == 24000
    assert body["object"] == "chat.completion"
    assert body["usage"]["prompt_tokens"] == 3


def test_extract_text_concatenates_text_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你好"},
                {"type": "text", "text": "，世界"},
            ],
        }
    ]
    assert _extract_text(messages) == "你好，世界"


def test_extract_text_uses_last_user_message() -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "ignore me"}]},
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert _extract_text(messages) == "second"


def test_extract_text_rejects_bare_string_content() -> None:
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError, match="must be an array"):
        _extract_text(messages)


def test_extract_text_rejects_unknown_block_type() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "audio_url", "audio_url": {"url": "http://x"}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="unsupported content block type"):
        _extract_text(messages)


def test_extract_prompt_accepts_image_url_data_uri() -> None:
    import base64

    from nanovllm_omni.serving.openai_adapter import _extract_prompt

    png = b"\x89PNG\r\n\x1a\nFAKE"
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }
    ]
    assert _extract_prompt(messages) == ("what is this?", png)


def test_extract_prompt_rejects_remote_image_url() -> None:
    from nanovllm_omni.serving.openai_adapter import _extract_prompt

    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}],
        }
    ]
    with pytest.raises(ValueError, match="base64 data URI"):
        _extract_prompt(messages)


def test_extract_text_ignores_image_block_in_text_compat() -> None:
    """TK-017: image_url block no longer rejected; _extract_text drops it."""
    import base64

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
                    },
                },
            ],
        }
    ]
    assert _extract_text(messages) == "hi"


def test_extract_text_raises_when_no_user_message() -> None:
    messages = [{"role": "system", "content": [{"type": "text", "text": "..."}]}]
    with pytest.raises(ValueError, match="no user text message found"):
        _extract_text(messages)
