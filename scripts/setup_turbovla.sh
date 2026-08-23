#!/usr/bin/env bash
# Setup script for TurboVLA integration (TK-011-rev Step 1).
#
# What it does:
#   1. Creates a conda env with Python 3.10 + torch 2.3.1 + cu121.
#      (TurboVLA requires Py3.10; see issue with LIBERO's Py3.8 below.)
#   2. Installs nanovllm-omni with the [turbovla] extra.
#   3. Installs the upstream TurboVLA package from GitHub.
#   4. Downloads the model checkpoint + DINOv3 + BERT into ./pretrained/.
#
# LIBERO env conflict (known): LIBERO upstream requires Python 3.8.13 +
# torch 1.11.0, which is incompatible with TurboVLA. Step 2 of TK-011-rev
# (examples/turbovla_libero.py) will need a separate env + bridge, or a
# newer LIBERO fork that supports modern torch. This script does NOT
# install LIBERO.
#
# Usage:
#   bash scripts/setup_turbovla.sh
#
# After it completes, run:
#   conda activate turbovla
#   python examples/turbovla.py \
#     --ckpt   pretrained/TurboVLA/checkpoints/libero/libero_object.pth \
#     --dinov3 pretrained/dinov3-vitb \
#     --bert   pretrained/bert-base-uncased

set -euo pipefail

ENV_NAME=${TURBOVLA_ENV:-turbovla}
PY_VERSION=${TURBOVLA_PY:-3.10}
PRETRAINED_DIR=${PRETRAINED_DIR:-pretrained}

echo "[setup] creating conda env ${ENV_NAME} (python=${PY_VERSION})"
conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[setup] installing torch 2.3.1 + cu121"
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip

echo "[setup] installing nanovllm-omni with [turbovla] extra"
pip install -e ".[turbovla]"

echo "[setup] installing upstream TurboVLA from github"
pip install "turbovla @ git+https://github.com/H-EmbodVis/TurboVLA.git"

mkdir -p "${PRETRAINED_DIR}"

if [ ! -d "${PRETRAINED_DIR}/dinov3-vitb" ]; then
    echo "[setup] downloading DINOv3 ViT-B"
    huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
        --local-dir "${PRETRAINED_DIR}/dinov3-vitb"
else
    echo "[setup] DINOv3 already present at ${PRETRAINED_DIR}/dinov3-vitb, skipping"
fi

if [ ! -d "${PRETRAINED_DIR}/bert-base-uncased" ]; then
    echo "[setup] downloading BERT-base-uncased"
    huggingface-cli download google-bert/bert-base-uncased \
        --local-dir "${PRETRAINED_DIR}/bert-base-uncased"
else
    echo "[setup] BERT already present, skipping"
fi

if [ ! -d "${PRETRAINED_DIR}/TurboVLA" ]; then
    echo "[setup] downloading TurboVLA model checkpoints from HF"
    huggingface-cli download H-EmbodVis/TurboVLA \
        --local-dir "${PRETRAINED_DIR}/TurboVLA"
else
    echo "[setup] TurboVLA checkpoint already present, skipping"
fi

echo
echo "[setup] done. To run the L1 demo:"
echo "  conda activate ${ENV_NAME}"
echo "  python examples/turbovla.py \\"
echo "    --ckpt   ${PRETRAINED_DIR}/TurboVLA/checkpoints/libero/libero_object.pth \\"
echo "    --dinov3 ${PRETRAINED_DIR}/dinov3-vitb \\"
echo "    --bert   ${PRETRAINED_DIR}/bert-base-uncased"
