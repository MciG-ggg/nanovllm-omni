"""Mimir stage 0 — prompt encoder (CODEC).

SaT sentence splitter + SONAR text encoder pipeline: text -> (B, S, 1024)
SONAR embeddings + sentence list. Stage 0 owns the "raw prompt -> LCM
input space" transform; stage 1 (mimir_lcm) and stage 2 (text_decoder)
operate entirely in embedding space.

Heavy deps are imported inside the factory and forward so that the
pipeline registry imports cleanly without ``sonar-space`` /
``wtpsplit`` installed. Failure mode is a helpful ``ImportError``
pointing at ``pip install -e ".[mimir]"`` + ``./scripts/setup_mimir.sh``.
"""

from __future__ import annotations

import re
from typing import Any

_INFERENCE_DTYPE = "float16"
# SONAR text_sonar_basic_encoder / decoder is the multilingual sentence
# embedding model Mimir was trained against. We pin the names so the
# stage does not silently pick a different SONAR variant and skew the
# embedding space (which would destroy the LCM's learned manifold).
_TEXT_ENCODER = "text_sonar_basic_encoder"
_TEXT_DECODER = "text_sonar_basic_decoder"

# ponytail: regex fallback for short prompts when SaT can't load (offline host,
# HF blocked, or 428 MB ONNX too heavy for the smoke). SaT is multilingual +
# threshold-aware; this fallback is English-bias, sentence-final punctuation
# only. Replace with SaT when the model is cached locally.
_FALLBACK_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\['\"\-])")


def _embeddings_to_batch(payload: Any, prompt: str) -> Any:
    """Bridge stage 0 -> stage 1.

    Stage 0 emits ``{"embeddings": tensor, "sentences": [...]}``; the LCM
    consumes an ``EmbeddingsBatch(seqs=tensor, padding_mask=None)`` from
    the upstream ``lcm.datasets.batch``. We construct it here so stage 1
    stays free of any dict unpacking.
    """
    from lcm.datasets.batch import EmbeddingsBatch

    embeddings = payload["embeddings"]
    return EmbeddingsBatch(seqs=embeddings, padding_mask=None)


def _prompt_encoder_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: load SaT + SONAR text encoder, return forward."""
    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    extra = dict(getattr(args, "extra", None) or {})
    if not bool(extra.get("allow_hf_download", False)):
        # Force HF Hub into offline mode so SONAR / SaT fail loud instead
        # of silently reaching out when the local cache is cold.
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = getattr(args, "dtype", None) or _INFERENCE_DTYPE
    dtype = getattr(torch, dtype_str, torch.float16)

    sat: Any = None
    try:
        from wtpsplit import SaT

        sat = SaT("segment-any-text/sat-3l")
        if torch.cuda.is_available():
            sat.half().to(device)
    except Exception as exc:  # pragma: no cover - exercised via offline smoke
        # SaT unavailable (offline cache cold, HF blocked, missing dep).
        # Fall back to regex so the pipeline still produces embeddings;
        # quality is lower on multilingual / long-form text.
        sat = None
        print(
            f"[mimir.prompt_encoder] SaT unavailable ({type(exc).__name__}: "
            f"{str(exc)[:80]}); falling back to regex sentence split",
            flush=True,
        )

    encoder = TextToEmbeddingModelPipeline(
        encoder=_TEXT_ENCODER,
        tokenizer=_TEXT_ENCODER,
        device=torch.device(device),
    )

    def _split(text: str, threshold: float) -> list[str]:
        # Real SaT takes a probability threshold; map it to a no-op for the
        # regex fallback (always use the punctuation splits).
        if sat is None:
            return [s for s in _FALLBACK_SENT_END.split(text) if s.strip()]
        out = list(sat.split([text], threshold=threshold))
        return [s.strip() for s in out[0] if s.strip()]

    def prompt_encoder_forward(payload: Any, sampling: Any) -> Any:
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict):
            text = str(payload.get("prompt", ""))
        else:
            text = str(payload)
        extras = (
            dict(sampling.extra)
            if sampling is not None and getattr(sampling, "extra", None)
            else {}
        )
        # SaT threshold is exposed via ``sampling.extra`` for safety; the
        # upstream README uses 0.02 which we mirror as the default.
        threshold = float(extras.get("sat_threshold", 0.02))
        sentences = _split(text, threshold)
        if not sentences:
            # Empty prompt -> single empty-sentence embedding so stage 1
            # still sees a well-shaped tensor (B=1, S=1, D=1024).
            sentences = [""]
        embeddings = encoder.predict(sentences, source_lang="eng_Latn", batch_size=1024)
        embeddings = embeddings.to(device=device, dtype=dtype)
        # LCM expects (B, S, D); SONAR returns (S, D).
        embeddings = embeddings.unsqueeze(0)
        return {"embeddings": embeddings, "sentences": sentences, "device": device}

    return prompt_encoder_forward


__all__ = ["_embeddings_to_batch", "_prompt_encoder_stage"]
