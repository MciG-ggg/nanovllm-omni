#!/usr/bin/env bash
# Start the stdlib HTTP adapter for MiniMind-O.
#
# Usage:
#   MODEL_DIR=/path/to/models bash start_server.sh
#   HOST=0.0.0.0 PORT=8080 DEVICE=cuda MODEL_DIR=/path/to/models bash start_server.sh
#
# Env:
#   MODEL_DIR (required): directory containing minimind-3o/ and mimi/ subdirs
#   HOST  (default 127.0.0.1)
#   PORT  (default 8000)
#   DEVICE (default cuda)
#
# Model directories are pre-provisioned with `hf download --local-dir`
# (see examples/offline_inference/minimind_o/README.md); HF_HUB_OFFLINE=1
# is exported here so the server never tries to fetch.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "$MODEL_DIR" ]]; then
    echo "MODEL_DIR is required (directory with minimind-3o/ and mimi/ subdirs)" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
exec python -m nanovllm_omni.serving.openai_adapter \
    --model-id "$MODEL_DIR/minimind-3o" \
    --mimi-model-id "$MODEL_DIR/mimi" \
    --device "$DEVICE" --host "$HOST" --port "$PORT"