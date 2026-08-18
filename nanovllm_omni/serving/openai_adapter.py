"""OpenAI-shaped HTTP adapter for the MiniMind-Omni 3-stage pipeline.

Maps ``POST /v1/chat/completions`` to ``Orchestrator.submit(pipeline, text)``
and returns the decoded ``AudioPayload`` as base64 WAV embedded in a
``ChatCompletion``-shaped JSON response. Stdlib only -- no fastapi,
no uvicorn, no pydantic.

What this is NOT, by design (see ``docs/design_mapping.md``):

  * A complete OpenAI-compatible server. vllm-omni's
    ``vllm_omni/entrypoints/openai/`` directory is ~20k LOC across 40
    files; this file is the design-mapping complement.
  * Multimodal input (image_url / input_audio / video_url / file).
  * Streaming (stream=true is silently treated as false), tool calling,
    response_format, logprobs, chat template rendering, token counting,
    usage, cancellation, metrics, middleware.

Add each of the above only when a deployment actually needs it.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from nanovllm_omni import Omni, SamplingParams
from nanovllm_omni.config import load_deploy_config
from nanovllm_omni.payloads import AudioPayload

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "minimind_omni.yaml"



def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Last user-role message, text content only. No multimodal blocks."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    raise ValueError("no user text message found")


def _chat_completion(audio: AudioPayload, model: str, prompt_tokens: int) -> dict[str, Any]:
    """Shape one aligned Omni audio output into an OpenAI response."""
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "audio": {
                    "data": base64.b64encode(audio.wav_bytes()).decode("ascii"),
                    "format": "wav",
                    "sample_rate": 24000,
                },
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": 0,
            "total_tokens": int(prompt_tokens),
        },
    }


def _build_state(config_path: Path, model_id: str, mimi_id: str, device: str | None):
    """Load deployment options and construct the aligned Omni engine."""
    deploy = load_deploy_config(config_path)
    defaults = next((stage.default_sampling_params for stage in deploy.stages if stage.name == "thinker"), {})
    allowed = {"temperature", "top_p", "top_k", "max_tokens", "stop", "seed", "n"}
    sampling = SamplingParams(**{key: value for key, value in defaults.items() if key in allowed})
    return Omni(model_id, device=device, extra={"deploy_config": deploy}), sampling


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
                self._json(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})
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
                audio = output.multimodal_output["audio"]
                self._json(200, _chat_completion(audio, model, len(text.split())))
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