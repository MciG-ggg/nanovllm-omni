#!/usr/bin/env bash
# SmolVLA real-env LIBERO evaluation (requires lerobot + a GPU host).
# Usage: bash run_libero.sh --model /path/to/smolvla_libero --num-episodes 2
set -euo pipefail
cd "$(dirname "$0")"
exec python libero_eval.py "$@"