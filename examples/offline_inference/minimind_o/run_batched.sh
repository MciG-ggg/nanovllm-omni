#!/usr/bin/env bash
# Batched continuous-batching smoke for MiniMind-O offline inference.
# Usage: bash run_batched.sh --model /path/to/minimind-3o --mimi /path/to/mimi --out batched_smoke
set -euo pipefail
cd "$(dirname "$0")"
exec python batched.py "$@"