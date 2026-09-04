"""Real-HTTP contract tests for the stdlib adapter (in-process server).

``test_serving_openai.py`` exercises the adapter's parser functions
in-process; this file goes one layer deeper and round-trips a real
``POST /v1/chat/completions`` over a real TCP socket. ``serve()`` takes
a pre-built ``(engine, sampling)`` state, so a tiny fake engine (no
weights, no GPU) drives the real ``do_POST`` handler -- Content-Length
parsing, JSON decode, and the error-to-status-code mapping all run
unchanged. Deterministic and CPU-only, so it is CI-safe.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

# generate_audio() inside the adapter imports torch at call time. Skip the
# whole module when torch is unavailable (e.g. the lint-and-test CI job).
torch = pytest.importorskip("torch")
from nanovllm_omni.serving.openai_adapter import serve  # noqa: E402 -- after importorskip

_FAKE_WAV = b"RIFF....fake wav"


class _FakeOutput:
    def to_dict(self) -> dict:
        return {
            "multimodal_output": {
                "audio": base64.b64encode(_FAKE_WAV).decode("ascii"),
                "audio_metadata": {"sample_rate": 24000, "format": "wav"},
            }
        }


class _FakeEngine:
    """Minimal engine exposing only what the adapter's handler touches."""

    def __init__(self, model: str = "minimind-o") -> None:
        self.model = model
        self.last_prompts: list | None = None
        self.last_sampling: object | None = None

    def generate(self, prompts, sampling_params=None):
        self.last_prompts = prompts
        self.last_sampling = sampling_params
        return [_FakeOutput()]


@dataclass
class _Server:
    url: str
    engine: _FakeEngine


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail("adapter did not start listening")


@pytest.fixture
def server() -> _Server:
    engine = _FakeEngine()
    port = _free_port()
    thread = threading.Thread(
        target=serve,
        args=((engine, "fake-sampling"), "127.0.0.1", port),
        daemon=True,
    )
    thread.start()
    _wait_ready(port)
    return _Server(f"http://127.0.0.1:{port}", engine)


def _post(url: str, body: dict) -> tuple[int, str, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                json.loads(response.read()),
            )
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), json.loads(error.read())


def test_post_text_returns_chat_completion_envelope(server: _Server) -> None:
    status, content_type, body = _post(
        server.url + "/v1/chat/completions",
        {
            "model": "minimind-o",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
        },
    )
    assert status == 200
    assert content_type == "application/json"
    assert body["object"] == "chat.completion"
    assert body["model"] == "minimind-o"
    audio = body["choices"][0]["message"]["audio"]
    assert audio["format"] == "wav"
    assert audio["sample_rate"] == 24000
    assert base64.b64decode(audio["data"]) == _FAKE_WAV
    assert body["usage"]["prompt_tokens"] == 1
    # The aligned Omni seam received the plain-text prompt and the deploy sampling.
    assert server.engine.last_prompts == ["你好"]
    assert server.engine.last_sampling == "fake-sampling"


def test_post_image_prompt_reaches_engine(server: _Server) -> None:
    png = b"\x89PNG\r\n\x1a\nFAKE"
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    status, _, _ = _post(
        server.url + "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ],
        },
    )
    assert status == 200
    assert server.engine.last_prompts == [{"prompt": "what is this?", "image": png}]


def test_post_bare_string_content_returns_400(server: _Server) -> None:
    status, _, body = _post(
        server.url + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


def test_post_unknown_path_returns_404(server: _Server) -> None:
    status, _, body = _post(
        server.url + "/v1/other",
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    )
    assert status == 404
    assert body["error"]["type"] == "invalid_request_error"
