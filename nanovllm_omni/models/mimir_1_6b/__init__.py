"""Mimir-1.6B-Instruct model family (TK-021).

Three-stage pipeline over SONAR embedding space:

- ``prompt_encoder`` (CODEC)  — text → SONAR sentence embeddings via SaT
- ``mimir_lcm``      (DIFFUSION) — LCM diffusion in embedding space
- ``text_decoder``   (CODEC)  — SONAR embeddings → text (terminal)

Pipeline topology is declarative; per-stage implementations live in
``prompt_encoder.py`` / ``mimir_lcm.py`` / ``text_decoder.py``. Heavy
deps (``sonar-space``, ``wtpsplit``, ``fairseq2``, the upstream ``lcm``
GitHub clone) are NOT in the main ``nanovllm-omni`` dep set -- install
with ``pip install -e ".[mimir]"`` and ``./scripts/setup_mimir.sh``.
"""

from .pipeline import MIMIR_1_6B_PIPELINE, PIPELINE

__all__ = ["PIPELINE", "MIMIR_1_6B_PIPELINE"]
