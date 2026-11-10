#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# The `mlx` extra carries mlx, mlx-lm, mlx-vlm and torchvision (see
# pyproject.toml), so this installs the same set the package declares
# instead of repeating a hand-written list that can drift.
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/../../..:$PWD/../../../third_party/nano-vllm" \
  exec uv run --no-project --with-editable "$PWD/../../..[mlx]" --python 3.12 \
    python run_mlx.py "$@"
