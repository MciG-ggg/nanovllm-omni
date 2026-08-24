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

# build the JSON body with python (no extra deps; jq if present, then
# we still need python for base64 decode in save_audio.py)
read -r -d '' BODY <<EOF || true
{"model":"$MODEL","messages":[{"role":"user","content":"$PROMPT"}]}
EOF

curl -sS -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data "$BODY"
echo