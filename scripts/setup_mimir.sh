#!/usr/bin/env bash
# One-off setup for the Mimir-1.6B-Instruct family (TK-021).
#
# Steps:
#   1. pip-install PyPI deps (torch pinned to 2.5.1, sonar-space,
#      wtpsplit, omegaconf).
#   2. Add Meta's fairseq2 RC index + install fairseq2==v0.3.0rc1
#      against it. fairseq2 is NOT on PyPI; it lives at
#      https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu121.
#   3. Clone the upstream ``large_concept_model`` repo into a local
#      ``lcm/`` directory. The ``lcm`` Python package is NOT on PyPI
#      (the PyPI ``lcm`` is the unrelated robotics comms library); it
#      ships inside that GitHub repo and is loaded as ``import lcm``
#      once ``lcm/`` is on PYTHONPATH.
#
# Run from repo root: ``./scripts/setup_mimir.sh``
# Idempotent: skips steps that are already satisfied.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LCM_DIR="${REPO_ROOT}/lcm"

cd "${REPO_ROOT}"

echo "[1/3] pip install [mimir] extra (torch 2.5.1, sonar-space, wtpsplit, omegaconf)"
pip install -e ".[mimir]"

echo "[2/3] fairseq2==v0.3.0rc1 from Meta's RC index"
pip install \
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu121 \
  --pre \
  fairseq2==v0.3.0rc1 || echo "fairseq2 already installed; continuing"

echo "[3/3] cloning large_concept_model into ${LCM_DIR}"
if [[ -d "${LCM_DIR}" ]]; then
  echo "  ${LCM_DIR} already exists; pulling latest"
  git -C "${LCM_DIR}" pull --ff-only || true
else
  git clone https://github.com/facebookresearch/large_concept_model.git "${LCM_DIR}"
fi

echo
echo "Mimir setup complete. Verify with:"
echo "  PYTHONPATH=${LCM_DIR}:\${PYTHONPATH:-} python -c 'import lcm; lcm.setup_fairseq2(); print(\"ok\")'"
echo
echo "Then pre-provision weights (one-off, ~3.3 GB):"
echo "  hf download mimir-lcm/Mimir-1.6B-Instruct --local-dir ~/models/Mimir-1.6B-Instruct"