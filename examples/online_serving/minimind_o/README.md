# MiniMind-O: Online serving

OpenAI-shape HTTP client examples for the stdlib server
(`python -m nanovllm_omni.serving.openai_adapter`). Mirrors vllm-omni's
per-model `examples/online_serving/<model>/` layout: one folder per
model, README + curl/shell wrappers + a stdlib python decoder.

## Setup

Same as `examples/offline_inference/minimind_o/README.md` — pull the
two checkpoints into local directories with `hf download`, point the
server at them:

```bash
hf download jingyaogong/minimind-3o --local-dir /home/mcig/minimind-3o
hf download kyutai/mimi             --local-dir /home/mcig/mimi
```

The bundle loader is offline-first and never auto-fetches.

## Run examples

### Start the server

```bash
HF_HUB_OFFLINE=1 python -m nanovllm_omni.serving.openai_adapter \
    --model-id /home/mcig/minimind-3o \
    --mimi-model-id /home/mcig/mimi \
    --device cuda --host 127.0.0.1 --port 8000
```

The server binds stdlib `ThreadingHTTPServer` to `host:port` and
serves a single route: `POST /v1/chat/completions`. Sampling defaults
are read from `deploy/minimind_omni.yaml`.

### Send a request

```bash
cd examples/online_serving/minimind_o
HOST=127.0.0.1 PORT=8000 bash curl_chat.sh "你好，请用一句话介绍你自己。" > resp.json
python save_audio.py resp.json --out hello.wav
```

`curl_chat.sh` is a one-line `POST` plus the request envelope; `save_audio.py`
decodes `choices[0].message.audio.data` (base64 WAV) and writes it to
disk. Stdlib only.

### Inspect the response

```bash
HOST=127.0.0.1 PORT=8000 bash curl_chat.sh "你好" | python -m json.tool | head -30
```

The response shape is OpenAI `chat.completion` JSON. Required fields:
`id`, `object: "chat.completion"`, `created`, `model`, `choices`,
`usage`. Audio payload lives under
`choices[0].message.audio = {"data": "<base64 wav>", "format": "wav",
"sample_rate": 24000}`.

## Notes

- Only `POST /v1/chat/completions` is implemented. The contract is
  pinned by `tests/test_serving_openai.py` and matches vllm-omni's
  `online_serving/<model>/` client expectations.
- The stdlib server has no streaming; use the offline
  `Omni.generate(...)` if you need per-step output.
- Run from a separate terminal so the shell wrappers don't block the
  inference.