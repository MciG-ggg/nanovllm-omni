"""Contract tests for the stdlib HTTP adapter's request parser.

The parser in ``nanovllm_omni.serving.openai_adapter`` is the gatekeeper
for ``POST /v1/chat/completions``; the OpenAI multimodal content array
is the only accepted ``messages[i].content`` shape. These tests pin
that contract.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.serving.openai_adapter import _extract_text


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
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="unsupported content block type"):
        _extract_text(messages)


def test_extract_text_raises_when_no_user_message() -> None:
    messages = [{"role": "system", "content": [{"type": "text", "text": "..."}]}]
    with pytest.raises(ValueError, match="no user text message found"):
        _extract_text(messages)
