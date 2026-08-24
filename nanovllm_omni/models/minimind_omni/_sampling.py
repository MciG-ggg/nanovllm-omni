"""Stateless sampling primitives shared by MiniMind-O's generation paths.

Both ``models/minimind_omni/generation.py`` (``stream_generate``, the
single-request wrapper) and ``models/minimind_omni/batched_generation.py``
(``BatchedThinkerRunner``, the underlying batched runner) need to:

  - sample one text token from logits with temperature / top_p / repetition
    penalty
  - sample one audio codebook code for one layer at one step with the
    per-layer temperature / penalty / top_k recipe

These helpers are the only piece they share; per-request state (buffers,
RNG, frame emission, EOS bookkeeping) stays with the caller. Constants
match the vendored upstream model exactly so we stay bit-exact across
both paths.

The earlier hand-rolled ``stream_generate_optimized`` used a *batched*
multinomial across active layers (one Philox call) for single-request
audio. That path is now retired; ``BatchedThinkerRunner`` (used by both
``stream_generate`` and ``run_batched_generate``) loops per layer per
request, calling this helper once per active layer.
"""

from __future__ import annotations

from typing import Any

# Audio codebook boundary: codes >= this are stop tokens. Matches the literal
# 2048 used in generation.py and ``_AUDIO_VOCAB_BOUNDARY`` in
# batched_generation.py. The two were duplicated across files before this
# extraction.
AUDIO_VOCAB_BOUNDARY = 2048
NUM_AUDIO_LAYERS = 8

# Default MiniMind-O sampling recipe. Vendored model uses these values;
# callers can override per-stage if a deploy YAML asks for it.
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
    for per-request determinism (Q10a).
    """
    import torch
    import torch.nn.functional as functional

    logits = logits_row.clone() / (temperature + 1e-9)
    if rp != 1.0:
        # Keep the penalty on-device; .tolist() here would force a sync.
        # Skip when rp == 1.0 (no-op) to avoid torch.unique + index overhead.
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
