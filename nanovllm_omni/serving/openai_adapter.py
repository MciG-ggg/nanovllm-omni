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

from nanovllm_omni.config import load_config
from nanovllm_omni.models import load_minimind_omni_bundle
from nanovllm_omni.payloads import AudioPayload
from nanovllm_omni.runtime import Orchestrator, build_pipeline

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "minimind_omni.yaml"


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


def _chat_completion(audio: AudioPayload, model: str) -> dict[str, Any]:
    """Shape one AudioPayload into a non-streaming ChatCompletion response."""
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
                    "sample_rate": audio.sample_rate,
                },
            },
            "finish_reason": "stop",
        }],
    }


def _build_state(config_path: Path, model_id: str, mimi_id: str, device: str):
    cfg, deploy = load_config(config_path)
    bundle = load_minimind_omni_bundle(
        model_id=model_id, mimi_model_id=mimi_id, device=device or deploy.device,
    )
    return build_pipeline(cfg, bundle=bundle), Orchestrator()


def serve(state, host: str, port: int) -> None:
    pipeline, orchestrator = state

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
                body = json.loads(self.rfile.read(length) or b"{}")
                text = _extract_text(body.get("messages", []))
                model = body.get("model", "minimind-omni")
                result = orchestrator.submit(pipeline, text)
                self._json(200, _chat_completion(result.audio, model))
            except ValueError as exc:
                # ponytail: only ValueError -> 400; everything else -> 500.
                # Split once a second exception type starts surfacing.
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