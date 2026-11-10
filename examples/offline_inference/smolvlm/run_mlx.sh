#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# mlx / mlx-vlm are Mac-track extras the project venv does not carry, so run
# in a throwaway uv env. torchvision is required by transformers 5's Idefics3
# image processor; without it the processor falls back to numpy and the MLX
# embedding lookup fails.
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/../../..:$PWD/../../../third_party/nano-vllm" \
  exec uv run --no-project --with mlx --with mlx-vlm --with torchvision \
    --with pillow --with torch --with transformers --with pyyaml --python 3.12 \
    python run_mlx.py "$@"
