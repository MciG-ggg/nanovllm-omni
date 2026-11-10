#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# thinker + talker on MLX, code2wav on torch/MPS (ADR 0024). mlx and mlx-lm are
# Mac-track extras the project venv does not carry, so run in a throwaway env.
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/../../..:$PWD/../../../third_party/nano-vllm" \
  exec uv run --no-project --with mlx --with mlx-lm --with torch --with transformers \
    --with safetensors --with pyyaml --with numpy --python 3.12 \
    python end2end_mlx.py "$@"
