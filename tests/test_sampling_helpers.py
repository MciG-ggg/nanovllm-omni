"""Focused tests for the shared sampling helpers.

These are the lowest-level building blocks of the MiniMind-O generation loop.
The helpers must:

  - be deterministic given the same input + same torch.Generator
  - obey temperature / top_p / repetition-penalty as documented
  - keep the per-layer audio penalty scoped to ``history_window``

Each helper gets ONE focused test -- the smallest thing that fails if the
math breaks. No model, no fixtures.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni._sampling import (  # noqa: E402
    AUDIO_VOCAB_BOUNDARY,
    DEFAULT_AUDIO_HISTORY_WINDOW,
    NUM_AUDIO_LAYERS,
    sample_one_audio_layer,
    sample_text_token,
)

# ---------------------------------------------------------------------------
# sample_text_token
# ---------------------------------------------------------------------------


def test_sample_text_token_is_deterministic_with_same_generator() -> None:
    """Same input + same Generator -> same token. Different RNG -> can differ."""
    torch.manual_seed(0)
    vocab = 32
    logits = torch.randn(vocab)
    history = [1, 2, 3, 1, 2]

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    t1 = sample_text_token(logits, history_ids=history, temperature=0.7, top_p=0.9, rp=1.1, gen=g1)
    t2 = sample_text_token(logits, history_ids=history, temperature=0.7, top_p=0.9, rp=1.1, gen=g2)
    assert t1 == t2
    assert 0 <= t1 < vocab


def test_sample_text_token_repetition_penalty_demotes_history_tokens() -> None:
    """With rp > 1, history tokens are *less* likely to be sampled than non-history.

    We can't assert exact token identity (RNG), but the *fraction* of draws
    landing on history ids with rp > 1 must be strictly lower than with
    rp == 1.0, averaged over enough samples.
    """
    torch.manual_seed(0)
    vocab = 16
    # Non-zero logits so the penalty ``logits /= rp`` has something to bite into.
    # With ``logits == 0`` the penalty is a no-op (0/2 == 0) and the test would
    # spuriously pass / fail.
    logits = torch.ones(vocab)
    history = list(range(8))  # half the vocab

    n = 2000
    g_no_rp = torch.Generator().manual_seed(7)
    draws_no_rp = [
        sample_text_token(
            logits, history_ids=history, temperature=1.0, top_p=1.0, rp=1.0, gen=g_no_rp
        )
        for _ in range(n)
    ]
    g_rp = torch.Generator().manual_seed(7)
    draws_rp = [
        sample_text_token(logits, history_ids=history, temperature=1.0, top_p=1.0, rp=2.0, gen=g_rp)
        for _ in range(n)
    ]

    frac_no_rp = sum(d in set(history) for d in draws_no_rp) / n
    frac_rp = sum(d in set(history) for d in draws_rp) / n
    # History tokens drop from logits=1 to logits=0.5; expected drop ~0.12.
    assert (
        frac_rp < frac_no_rp - 0.1
    ), f"rp penalty had no effect: {frac_no_rp=:.3f}, {frac_rp=:.3f}"


def test_sample_text_token_top_p_narrows_distribution() -> None:
    """With top_p < 1, only the top-prob mass is sampled -- small-vocab tokens are unreachable."""
    torch.manual_seed(0)
    vocab = 64
    # One token dominates; the rest are uniform low. Top_p should pin us to
    # the dominant token's vicinity.
    logits = torch.full((vocab,), -5.0)
    logits[3] = 5.0  # sharp peak at id 3
    history: list[int] = []

    n = 500
    g = torch.Generator().manual_seed(11)
    draws = [
        sample_text_token(logits, history_ids=history, temperature=1.0, top_p=0.5, rp=1.0, gen=g)
        for _ in range(n)
    ]
    # With top_p=0.5 and a sharp peak, almost all draws should be id 3.
    frac_top = sum(d == 3 for d in draws) / n
    assert frac_top > 0.9, f"top_p didn't concentrate on peak: {frac_top=:.3f}"


# ---------------------------------------------------------------------------
# sample_one_audio_layer
# ---------------------------------------------------------------------------


def test_sample_one_audio_layer_penalty_only_affects_history_window() -> None:
    """Codes older than ``history_window`` must NOT be touched by the penalty.

    We feed an empty history (no penalty applied) and a history of length 5
    where the last 3 ids should be penalised. With penalty=2.0, the
    probability mass on the last-3 ids should drop vs the empty-history
    baseline; the first-2 ids should be unchanged.
    """
    torch.manual_seed(0)
    # Vocab must be >= the helper's default top_k=50; MiniMind's audio vocab is
    # ~2048, so 64 fits comfortably without bumping the kwarg.
    vocab = 64
    logits = torch.ones(vocab)
    last3 = [10, 11, 12]
    older = [20, 21]
    full_history = older + last3  # [20, 21, 10, 11, 12]
    n = 4000

    g_empty = torch.Generator().manual_seed(99)
    draws_empty = [
        sample_one_audio_layer(logits, [], temperature=1.0, gen=g_empty) for _ in range(n)
    ]
    g_full = torch.Generator().manual_seed(99)
    draws_full = [
        sample_one_audio_layer(logits, full_history, temperature=1.0, penalty=2.0, gen=g_full)
        for _ in range(n)
    ]

    last3_set = set(last3)
    older_set = set(older)
    frac_empty_last3 = sum(d in last3_set for d in draws_empty) / n
    frac_full_last3 = sum(d in last3_set for d in draws_full) / n
    frac_empty_older = sum(d in older_set for d in draws_empty) / n
    frac_full_older = sum(d in older_set for d in draws_full) / n

    # Penalty halves the last-3 logits (1 -> 0.5), so their softmax mass drops
    # by ~38%; older-than-window logits are untouched. We measure relative drop
    # to absorb the total-mass-shift artefact on the older ids.
    ratio_last3 = frac_full_last3 / frac_empty_last3
    ratio_older = frac_full_older / frac_empty_older

    assert ratio_last3 < 0.8, (
        f"penalty missed last-3 history: empty={frac_empty_last3:.3f}, "
        f"penalised={frac_full_last3:.3f}, ratio={ratio_last3:.3f}"
    )
    assert abs(ratio_older - 1.0) < 0.1, (
        f"penalty leaked outside history window: empty={frac_empty_older:.3f}, "
        f"penalised={frac_full_older:.3f}, ratio={ratio_older:.3f}"
    )


def test_audio_vocab_boundary_constant_matches_module_default() -> None:
    """The vendored model uses 2048 as the audio stop-token boundary; lock it in."""
    assert AUDIO_VOCAB_BOUNDARY == 2048
    assert NUM_AUDIO_LAYERS == 8
    assert DEFAULT_AUDIO_HISTORY_WINDOW == 3
