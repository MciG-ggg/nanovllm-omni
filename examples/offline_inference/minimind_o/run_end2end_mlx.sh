#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# thinker + talker on MLX, code2wav on torch/MPS (ADR 0024). The `mlx` extra
# declares the whole set, so this does not re-list it by hand.
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/../../..:$PWD/../../../third_party/nano-vllm" \
  exec uv run --no-project --with-editable "$PWD/../../..[mlx]" --python 3.12 \
    python end2end_mlx.py "$@"
