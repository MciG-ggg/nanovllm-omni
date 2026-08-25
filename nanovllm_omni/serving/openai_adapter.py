"""OpenAI-shaped HTTP adapter for the MiniMind-Omni audio pipeline.

Maps ``POST /v1/chat/completions`` to the aligned ``Omni`` API and returns
audio as base64 WAV embedded in a ``ChatCompletion``-shaped JSON response.
Stdlib only -- no fastapi, no uvicorn, no pydantic.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from nanovllm_omni import Omni, SamplingParams
from nanovllm_omni.config.registry import load_deploy_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "minimind_omni.yaml"


_ALLOWED_BLOCK_TYPES = frozenset({"text"})


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate the text blocks of the last user-role message.

    `content` MUST be a list of OpenAI-shape content blocks, e.g.
    ``[{"type": "text", "text": "..."}, ...]``. Bare strings and any
    block type other than ``text`` raise ``ValueError`` so the handler
    returns HTTP 400 instead of silently dropping user intent.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError(
                "message.content must be an array of content blocks "
                '(e.g. [{"type": "text", "text": "..."}]); '
                f"got {type(content).__name__}"
            )
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError(f"content block must be an object, got {type(block).__name__}")
            block_type = block.get("type")
            if block_type not in _ALLOWED_BLOCK_TYPES:
                raise ValueError(
                    f"unsupported content block type {block_type!r}; "
                    f"only {sorted(_ALLOWED_BLOCK_TYPES)} are accepted"
                )
            parts.append(block.get("text", ""))
        return "".join(parts)
    raise ValueError("no user text message found")


def _chat_completion(payload: dict[str, Any], model: str, prompt_tokens: int) -> dict[str, Any]:
    """Shape one aligned Omni output's ``to_dict()`` payload into an OpenAI response.

    ``payload`` is ``OmniRequestOutput.to_dict()``: multimodal bytes (audio)
    arrive already base64-encoded, so this function only builds the
    ChatCompletion envelope around them.
    """
    audio_b64 = payload["multimodal_output"]["audio"]
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "audio": {
                        "data": audio_b64,
                        "format": "wav",
                        "sample_rate": 24000,
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": 0,
            "total_tokens": int(prompt_tokens),
        },
    }


def _build_state(config_path: Path, model_id: str, mimi_id: str, device: str | None):
    """Load deployment options and construct the aligned Omni engine."""
    deploy = load_deploy_config(config_path)
    defaults = next(
        (stage.default_sampling_params for stage in deploy.stages if stage.name == "thinker"), {}
    )
    allowed = {"temperature", "top_p", "top_k", "max_tokens", "stop", "seed", "n"}
    sampling = SamplingParams(**{key: value for key, value in defaults.items() if key in allowed})
    return (
        Omni(model_id, device=device, mimi_model_id=mimi_id, extra={"deploy_config": deploy}),
        sampling,
    )


def serve(state, host: str, port: int) -> None:
    engine, sampling = state

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # silence stderr access log
            pass

        def _json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler contract
            if self.path != "/v1/chat/completions":
                self._json(
                    404,
                    {
                        "error": {
                            "message": f"unknown path {self.path}",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty array")
                text = _extract_text(messages)
                model = body.get("model", engine.model)
                output = engine.generate([text], sampling_params=sampling)[0]
                payload = output.to_dict()
                self._json(200, _chat_completion(payload, model, len(text.split())))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:
                self._json(500, {"error": {"message": str(exc), "type": "server_error"}})

    print(f"nanovllm-omni adapter on http://{host}:{port}/v1/chat/completions", file=sys.stderr)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-shaped adapter for MiniMind-Omni.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi-model-id", default="kyutai/mimi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    state = _build_state(args.config, args.model_id, args.mimi_model_id, args.device)
    serve(state, args.host, args.port)


if __name__ == "__main__":
    main()
