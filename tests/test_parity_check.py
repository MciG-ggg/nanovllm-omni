"""ADR 0021's three-tier criterion on constructed arms.

``mlx`` is a Mac-track extra and not in the project venv, so the whole file is
guarded; nothing here loads a model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")

from tools.parity_check import EXACT, FAIL, TOP_K_MATCH, Arm, parity_report  # noqa: E402


def _arm(token_ids: list[int], top_k: dict[int, list[int]] | None = None) -> Arm:
    """An arm whose step ``i`` offers ``top_k[i]`` as its candidates, each at 0.0."""
    if top_k is None:
        return Arm(token_ids)
    return Arm(token_ids, [[(token, 0.0) for token in top_k[i]] for i in sorted(top_k)])


def test_identical_tokens_to_max_tokens_are_exact() -> None:
    reference = _arm([4, 5, 6], {0: [4, 9], 1: [5, 8], 2: [6, 7]})
    candidate = _arm([4, 5, 6], {0: [4, 9], 1: [5, 8], 2: [6, 7]})

    verdict = parity_report(reference, candidate, max_tokens=3)

    assert verdict.tier == EXACT
    assert verdict.index is None


def test_divergence_inside_both_top_k_is_top_k_match() -> None:
    reference = _arm([4, 5, 6], {0: [4, 9], 1: [5, 7], 2: [6, 7]})
    candidate = _arm([4, 7, 6], {0: [4, 9], 1: [7, 5], 2: [6, 7]})

    verdict = parity_report(reference, candidate, max_tokens=3)

    assert verdict.tier == TOP_K_MATCH
    assert verdict.index == 1
    assert verdict.reference_token == 5
    assert verdict.candidate_token == 7
    assert [token for token, _ in verdict.reference_top] == [5, 7]
    assert [token for token, _ in verdict.candidate_top] == [7, 5]


def test_divergence_outside_top_k_is_fail() -> None:
    reference = _arm([4, 5], {0: [4, 9], 1: [5, 6]})
    candidate = _arm([4, 7], {0: [4, 9], 1: [7, 8]})

    verdict = parity_report(reference, candidate, max_tokens=2)

    assert verdict.tier == FAIL
    assert verdict.index == 1
    assert [token for token, _ in verdict.reference_top] == [5, 6]
    assert [token for token, _ in verdict.candidate_top] == [7, 8]


def test_agreement_short_of_max_tokens_is_fail() -> None:
    reference = _arm([4, 5])
    candidate = _arm([4, 5])

    verdict = parity_report(reference, candidate, max_tokens=8)

    assert verdict.tier == FAIL
    assert "max_tokens=8" in verdict.reason


def test_missing_logprobs_degrades_to_fail_not_top_k_match() -> None:
    reference = _arm([4, 5])
    candidate = _arm([4, 7])

    verdict = parity_report(reference, candidate, max_tokens=2)

    assert verdict.tier == FAIL
    assert verdict.index == 1
    assert "no logprobs" in verdict.reason
