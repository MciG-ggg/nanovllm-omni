# 03 — HTTP `POST /v1/chat/completions` returns OpenAI-shape response (HTTP seam)

**What to build:** The stdlib HTTP server returns OpenAI-shape `chat.completion` responses. A client sending an OpenAI-style `messages` request gets a response with `choices[0].message.audio` carrying base64 WAV data, plus all the standard OpenAI envelope fields (`id`, `object`, `created`, `model`, `usage`). The response shape is field-name-compatible with vllm-omni's HTTP endpoint for the same request.

**Blocked by:** TICKET-02 (the Python `Omni` API must exist as the seam under the HTTP adapter).

**Status:** ready-for-agent

- [ ] `python -m nanovllm_omni.serving.openai_adapter --config <yaml> [--port PORT]` starts the stdlib HTTP server.
- [ ] `curl -X POST http://localhost:PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "<id>", "messages": [{"role": "user", "content": "hi"}]}'` returns HTTP 200 with a JSON body.
- [ ] Response body has top-level fields: `id` (string), `object: "chat.completion"`, `created` (unix seconds, int), `model` (string), `choices` (array), `usage` (object).
- [ ] `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens` as integers.
- [ ] `choices` has length 1; `choices[0]` has `index: 0`, `finish_reason: "stop"`, `message.role: "assistant"`, `message.audio` object.
- [ ] `choices[0].message.audio` has `data` (base64 string), `format: "wav"`, `sample_rate: 24000`.
- [ ] `base64.b64decode(choices[0].message.audio.data)` round-trips to a valid WAV file openable by Python's `wave.open` module.
- [ ] Invalid request body (malformed JSON, missing `messages`) returns HTTP 400 with a JSON error body containing a human-readable message.
- [ ] Internal engine error returns HTTP 500 with a JSON error body containing a human-readable message.
- [ ] All previously existing tests still pass.
- [ ] No new third-party dependency introduced (stdlib only).