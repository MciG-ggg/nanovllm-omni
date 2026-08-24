#!/usr/bin/env bash
# Single-prompt smoke for MiniMind-O offline inference.
# Usage: bash run_end2end.sh --model /path/to/minimind-3o --mimi /path/to/mimi --out audio.wav
set -euo pipefail
cd "$(dirname "$0")"
exec python end2end.py "$@"