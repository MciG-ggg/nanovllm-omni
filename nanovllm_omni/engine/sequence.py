"""Per-request sequence state for the per-stage continuous-batching scheduler.

Data structures lifted from the reference's ``Sequence`` / ``PrefillChunk`` shape
(TK-004). ``Sequence`` is the per-request mutable state a stage's scheduler
owns; ``PrefillChunk`` marks a sub-range of a prompt being prefilled this
round so the same scheduler can interleave prefill with decode (Q8a).

The fields are deliberately minimal -- PagedAttention, prefix caching, and
speculative decoding are out of scope.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class SequenceStatus(enum.Enum):
    """Lifecycle states a single sequence passes through.

    Finer than the reference's coarse WAITING/FINISHED split so the scheduler
    can interleave prefill with decode (a sequence in PREFILL does not yet
    appear in decode groups).
    """

    WAITING = "waiting"  # admitted but no KV written yet
    PREFILL = "prefill"  # KV is being filled by a chunked prefill forward
    DECODE = "decode"  # KV is fully prefilled; eligible for decode groups
    FINISHED = "finished"  # done (EOS / max_tokens / aborted)


@dataclass
class Sequence:
    """Per-request mutable state, owned by a single stage's scheduler.

    ``kv_blocks`` is a thin reference into the stage's KV cache (one block
    per layer). The teaching shape keeps it as ``list[Any]``; a PagedAttention
    pool would slot in here without touching the scheduler.
    """

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    kv_blocks: list[Any] = field(default_factory=list)
    finished_reason: str | None = None


@dataclass
class PrefillChunk:
    """A sub-range of a prompt being prefilled this scheduler round.

    Used when a sequence is admitted but its prompt is longer than the stage
    can consume in one forward (chunked prefill). For MiniMind-O all current
    prompts fit in one chunk, so the scheduler always emits ``start=0,
    end=len(prompt_ids)`` -- but the data structure is here for future
    long-prompt workloads (Q8a + Q12).
    """

    sequence: Sequence
    start: int
    end: int


__all__ = ["PrefillChunk", "Sequence", "SequenceStatus"]
