"""Thinker prompt formatting: raw text gets the checkpoint chat template."""

from nanovllm_omni.models.minimind_omni.stage import _format_prompt


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, open_thinking):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "open_thinking": open_thinking,
            }
        )
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"


def test_raw_prompt_gets_the_chat_template() -> None:
    tokenizer = _Tokenizer()

    formatted = _format_prompt(tokenizer, "你好", open_thinking=False)

    assert formatted.startswith("<|im_start|>user")
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "你好"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "open_thinking": False,
        }
    ]


def test_formatted_prompt_passes_through() -> None:
    tokenizer = _Tokenizer()
    prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"

    assert _format_prompt(tokenizer, prompt, open_thinking=False) == prompt
    assert tokenizer.calls == []


def test_open_thinking_is_forwarded() -> None:
    tokenizer = _Tokenizer()

    _format_prompt(tokenizer, "hi", open_thinking=True)

    assert tokenizer.calls[0]["open_thinking"] is True
