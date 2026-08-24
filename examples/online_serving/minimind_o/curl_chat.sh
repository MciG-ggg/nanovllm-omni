#!/usr/bin/env bash
# curl-based /v1/chat/completions client for MiniMind-O.
#
# Usage:
#   HOST=127.0.0.1 PORT=8000 bash curl_chat.sh "你好，请用一句话介绍你自己。" > resp.json
#   HOST=127.0.0.1 PORT=8000 bash curl_chat.sh <prompt> | tee resp.json | jq .
#
# Env: HOST (default 127.0.0.1), PORT (default 8000), MODEL (default
# "minimind-o"). Requires python3 (always present) or `jq`; falls back
# to a tiny python heredoc if jq is missing.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-minimind-o}"
PROMPT="${1:-你好，请用一句话介绍你自己。}"

# Build the JSON body with python so the nested content array is
# correctly quoted regardless of the prompt's contents (quotes,
# newlines, etc.). The adapter requires the OpenAI multimodal content
# array form, not a bare string.
BODY=$(python3 -c '
import json, os, sys
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": [{"type": "text", "text": sys.argv[1]}]}],
}))' "$PROMPT")

curl -sS --retry 5 --retry-delay 1 --retry-connrefused \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data "$BODY"
echo