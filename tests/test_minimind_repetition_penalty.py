"""The vectorised repetition penalty is bit-exact vs the per-token loop.

``decode_minimind`` applies HF's repetition penalty to the last position's
logits before top-p + multinomial. It used to divide one logit at a time
(``for token in set(...): logits_i[token] /= rp``); that is two kernel
launches per history token on every decode step. The replacement does one
indexed divide. A different rounding here would change the sampled token
and every bridge/audio row after it, so the equivalence is worth a CPU
test: same elements, same divisor, same result.

The full decode loop cannot run on CPU (fork ``ModelRunner`` needs CUDA +
NCCL), so this locks the arithmetic that changed, not the loop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.stage_runner import (  # noqa: E402 -- after importorskip
    _apply_repetition_penalty,
)

# A history that repeats tokens: the set() must divide each one exactly once.
_HISTORY = [3, 7, 7, 11, 60, 3, 0]


def _per_token_loop(logits, token_ids, penalty):
    for token in set(token_ids):
        logits[token] /= penalty
    return logits


@pytest.mark.parametrize("penalty", [1.05, 1.2, 1.0])
def test_matches_per_token_loop(penalty) -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, dtype=torch.float32) * 5.0

    vectorised = logits.clone()
    _apply_repetition_penalty(vectorised, _HISTORY, penalty)

    expected = _per_token_loop(logits.clone(), _HISTORY, penalty)

    assert torch.equal(vectorised, expected)


def test_leaves_unseen_tokens_untouched() -> None:
    torch.manual_seed(1)
    logits = torch.randn(64, dtype=torch.float32) * 5.0

    penalised = logits.clone()
    _apply_repetition_penalty(penalised, _HISTORY, 1.05)

    untouched = [token for token in range(64) if token not in set(_HISTORY)]
    assert torch.equal(penalised[untouched], logits[untouched])
