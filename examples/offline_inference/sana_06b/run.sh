#!/usr/bin/env bash
# Sana-0.6B L1 demo (TK-009): one prompt -> output.png.
# Usage: bash run.sh --model /path/to/sana_snapshot --device cuda --steps 20
set -euo pipefail
cd "$(dirname "$0")"
exec python run.py "$@"