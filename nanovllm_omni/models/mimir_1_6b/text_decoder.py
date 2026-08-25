"""Mimir stage 2 — text decoder (CODEC, terminal).

SONAR text decoder: (1, S', 1024) embedding tensor -> ``str``.
Terminal stage; ``final_output_type="text"`` so the engine wraps the
string via ``OmniRequestOutput.from_pipeline(final_output_type="text")``
and lands it under ``multimodal_output["text"]``.

Heavy dep (``sonar-space``) is imported lazily inside the factory.
The ``monkeypatch __import__`` test in ``tests/test_mimir_1_6b.py``
locks the import-guard contract.
"""

from __future__ import annotations

from typing import Any

_TEXT_DECODER = "text_sonar_basic_decoder"


def _text_decoder_stage(deploy: Any, args: Any) -> Any:
    """Stage 2 factory: load SONAR text decoder, return forward."""
    import torch
    from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

    extra = dict(getattr(args, "extra", None) or {})
    if not bool(extra.get("allow_hf_download", False)):
        # Force HF Hub into offline mode so SONAR fails loud instead of
        # silently reaching out when the local cache is cold.
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    decoder = EmbeddingToTextModelPipeline(
        decoder=_TEXT_DECODER,
        tokenizer=_TEXT_DECODER,
        device=torch.device(device),
    )

    def text_decoder_forward(payload: Any, sampling: Any) -> Any:
        # ``payload`` is the bridge output of stage 1: a (1, S', 1024)
        # tensor on the LCM's device; the bridge already stripped the
        # prompt prefix.
        embeddings = payload
        if hasattr(embeddings, "device"):
            embeddings = embeddings.to(device=device, dtype=torch.float32)
        results = decoder.predict(embeddings, target_lang="eng_Latn")
        # ``results`` is a list[str] (one per row of the input batch);
        # for batch size 1 we return the single string. Larger batches
        # concatenate with newlines; the chat use-case is always B=1.
        text = "\n".join(results) if isinstance(results, list) else str(results)
        return text

    return text_decoder_forward


__all__ = ["_text_decoder_stage"]
