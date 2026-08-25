#!/usr/bin/env bash
# SD-Turbo offline demo (TK-009 rev2).
# Reads the snapshot from $1 (default: $HOME/models/sd_turbo on jingrui,
# the bare stabilityai/sd-turbo HF id otherwise). One step, 512x512.
#
# Usage:
#   bash run.sh                          # bare repo id, requires --allow-download
#   bash run.sh /path/to/snapshot        # local snapshot, no network
#
set -euo pipefail
MODEL="${1:-${SD_TURBO_MODEL:-stabilityai/sd-turbo}}"
PROMPT="${PROMPT:-a cute cat, studio photo}"
OUT="${OUT:-output.png}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python examples/offline_inference/sd_turbo/run.py \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --output "$OUT" \
        "$@"