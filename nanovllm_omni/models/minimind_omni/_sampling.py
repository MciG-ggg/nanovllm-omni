"""Stateless sampling primitives shared by MiniMind-O's generation paths.

Shared by ``generation.stream_generate`` (single-request) and
``batched_generation.BatchedThinkerRunner`` (batched runner) -- the only
piece they share; per-request state (buffers, RNG, frame emission, EOS
bookkeeping) stays with the caller. Constants match the vendored upstream
model exactly so we stay bit-exact across both paths.

Public symbols: ``sample_text_token``, ``sample_one_audio_layer``, the
``AUDIO_VOCAB_BOUNDARY`` / ``NUM_AUDIO_LAYERS`` constants, and the
``DEFAULT_*`` recipe constants.
"""

from __future__ import annotations

from typing import Any

# Audio codebook boundary: codes >= this are stop tokens.
AUDIO_VOCAB_BOUNDARY = 2048
NUM_AUDIO_LAYERS = 8

# Default MiniMind-O sampling recipe. Callers can override per-stage via
# deploy YAML.
DEFAULT_TEXT_TEMPERATURE = 0.75
DEFAULT_TEXT_TOP_P = 0.90
DEFAULT_AUDIO_TEMPERATURE = 0.2
DEFAULT_AUDIO_PENALTY = 1.05
DEFAULT_AUDIO_TOP_K = 50
DEFAULT_AUDIO_HISTORY_WINDOW = 3


def sample_text_token(
    logits_row: Any,
    *,
    history_ids: Any,
    temperature: float = DEFAULT_TEXT_TEMPERATURE,
    top_p: float = DEFAULT_TEXT_TOP_P,
    rp: float = 1.0,
    gen: Any = None,
) -> int:
    """Sample one text token from a single-row logits tensor.

    ``history_ids`` may be any on-device tensor or a Python list of ints;
    it is moved to ``logits_row.device`` once via ``torch.as_tensor``.
    ``gen=None`` uses the process-default RNG; pass a ``torch.Generator``
    for per-request determinism.
    """
    import torch
    import torch.nn.functional as functional

    logits = logits_row.clone() / (temperature + 1e-9)
    if rp != 1.0:
        # Keep the penalty on-device; .tolist() here would force a sync.
        uniq = torch.unique(torch.as_tensor(history_ids, device=logits.device))
        logits[uniq] /= rp
    if top_p and top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        mask = torch.cumsum(functional.softmax(sorted_l, dim=-1), dim=-1) > top_p
        # Rotate the kept-prefix one slot so the boundary token is sampled.
        mask[1:], mask[0] = mask[:-1].clone(), False
        logits[sorted_i[mask]] = -float("Inf")
    return int(torch.multinomial(functional.softmax(logits, dim=-1), 1, generator=gen).item())


def sample_one_audio_layer(
    logits_row: Any,
    history: list[int],
    *,
    temperature: float = DEFAULT_AUDIO_TEMPERATURE,
    penalty: float = DEFAULT_AUDIO_PENALTY,
    history_window: int = DEFAULT_AUDIO_HISTORY_WINDOW,
    top_k: int = DEFAULT_AUDIO_TOP_K,
    gen: Any = None,
) -> int:
    """Sample one audio codebook code for one layer at one step.

    Vendor recipe: divide by temperature, divide the last ``history_window``
    history codes by ``penalty``, topk, multinomial over the kept tokens.
    """
    import torch
    import torch.nn.functional as functional

    logits = logits_row.clone() / temperature
    for prev in history[-history_window:]:
        logits[prev] /= penalty
    top_v, top_i = logits.topk(top_k)
    idx = int(torch.multinomial(functional.softmax(top_v, dim=-1), 1, generator=gen).item())
    return int(top_i[idx])


__all__ = [
    "AUDIO_VOCAB_BOUNDARY",
    "DEFAULT_AUDIO_HISTORY_WINDOW",
    "DEFAULT_AUDIO_PENALTY",
    "DEFAULT_AUDIO_TEMPERATURE",
    "DEFAULT_AUDIO_TOP_K",
    "DEFAULT_TEXT_TEMPERATURE",
    "DEFAULT_TEXT_TOP_P",
    "NUM_AUDIO_LAYERS",
    "sample_one_audio_layer",
    "sample_text_token",
]
