#!/usr/bin/env bash
# SmolVLA synthetic-obs L1 demo (no real env, just verify action shape).
# Usage: bash run_synthetic.sh --model /path/to/smolvla_libero --dtype int8 --device cuda
set -euo pipefail
cd "$(dirname "$0")"
exec python synthetic_obs.py "$@"