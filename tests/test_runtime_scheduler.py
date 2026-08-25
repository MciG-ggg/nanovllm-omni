"""Focused tests for the per-stage RuntimeScheduler (TK-004).

The SPEC requires a per-stage scheduler shaped like vLLM's: one instance per
LLM stage, Sequence-shaped state, chunked prefill + decode groups. These
tests pin the data structure and the basic lifecycle transitions; the
real-needle test (driving a forward through the scheduler) lives in
``test_batched_generation.py`` where ``BatchedThinkerRunner`` is the runner.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.engine.runtime_scheduler import (
    RuntimeGroup,
    RuntimeScheduler,
    RuntimeSchedulerOutput,
)
from nanovllm_omni.engine.sequence import PrefillChunk, Sequence, SequenceStatus

# ---------------------------------------------------------------------------
# Sequence data structure
# ---------------------------------------------------------------------------


def test_sequence_defaults_to_waiting_with_empty_state() -> None:
    """Fresh Sequence is WAITING, no tokens, no KV."""
    seq = Sequence(request_id="r0")
    assert seq.request_id == "r0"
    assert seq.token_ids == []
    assert seq.num_tokens == 0
    assert seq.status is SequenceStatus.WAITING
    assert seq.kv_blocks == []
    assert seq.finished_reason is None


def test_sequence_status_enum_has_four_states() -> None:
    """Per SPEC: WAITING -> PREFILL -> DECODE -> FINISHED.

    If a new state is added (e.g. ABORTED), the scheduler's branching needs
    to grow with it -- catch the drift here.
    """
    assert len(SequenceStatus) == 4
    names = {s.name for s in SequenceStatus}
    assert names == {"WAITING", "PREFILL", "DECODE", "FINISHED"}


def test_prefill_chunk_carries_subrange() -> None:
    """PrefillChunk records a [start, end) slice of a sequence's prompt."""
    seq = Sequence(request_id="r0", token_ids=[1, 2, 3, 4, 5], num_tokens=5)
    chunk = PrefillChunk(seq=seq, start=1, end=4)
    assert chunk.seq is seq
    assert chunk.start == 1
    assert chunk.end == 4


# ---------------------------------------------------------------------------
# RuntimeScheduler lifecycle
# ---------------------------------------------------------------------------


def test_scheduler_admits_to_max_num_seqs_then_blocks() -> None:
    """Admission is capped by max_num_seqs (Q8a)."""
    sched = RuntimeScheduler(max_num_seqs=2)
    for i in range(5):
        sched.add_sequence(Sequence(request_id=f"r{i}", num_tokens=4))

    out = sched.schedule()
    # First call admits 2; remaining 3 stay WAITING.
    assert sched.has_work()
    assert sum(len(g.items) for g in out.prefill_groups) == 2
    assert len(sched.running) == 2
    assert len(sched.waiting) == 3

    # Finish both running sequences -- next call admits the next 2.
    sched.update_from_output(prefilled=["r0", "r1"], finished=["r0", "r1"])
    out = sched.schedule()
    assert sum(len(g.items) for g in out.prefill_groups) == 2
    assert len(sched.running) == 2


def test_scheduler_prefill_then_decode_then_finished() -> None:
    """One sequence's full lifecycle through 3 rounds.

    Round 1: schedule() puts the seq into PREFILL status, emits a prefill chunk.
    Round 2 (after update_from_output(prefilled=[rid])): seq moves to DECODE,
    a decode group is emitted.
    Round 3 (after update_from_output(finished=[rid])): seq is FINISHED and
    dropped from running.
    """
    sched = RuntimeScheduler(max_num_seqs=4)
    sched.add_sequence(Sequence(request_id="r0", num_tokens=3))

    out = sched.schedule()
    assert len(out.prefill_groups) == 1
    assert isinstance(out.prefill_groups[0].items[0], PrefillChunk)
    assert sched.running["r0"].status is SequenceStatus.PREFILL
    assert len(out.decode_groups) == 0

    sched.update_from_output(prefilled=["r0"])
    out = sched.schedule()
    assert len(out.prefill_groups) == 0
    assert len(out.decode_groups) == 1
    assert sched.running["r0"].status is SequenceStatus.DECODE

    sched.update_from_output(finished=["r0"])
    assert "r0" not in sched.running
    assert sched.is_finished("r0")
    assert sched.get("r0").status is SequenceStatus.FINISHED


def test_scheduler_decode_groups_partition_by_num_tokens() -> None:
    """Decode groups form only over sequences sharing a ``num_tokens`` value."""
    sched = RuntimeScheduler(max_num_seqs=8)
    # 3 sequences at num_tokens=4, 2 at num_tokens=7.
    for rid, n in [("a", 4), ("b", 4), ("c", 4), ("d", 7), ("e", 7)]:
        sched.add_sequence(Sequence(request_id=rid, num_tokens=n))
    # First schedule() admits and emits prefill chunks; update_from_output moves
    # them to DECODE; second schedule() emits the decode groups partitioned
    # by num_tokens.
    sched.schedule()
    sched.update_from_output(prefilled=["a", "b", "c", "d", "e"])
    out = sched.schedule()

    decode_groups = out.decode_groups
    assert len(decode_groups) == 2
    by_n = {len(g.items): g.items for g in decode_groups}
    assert sorted(seq.request_id for seq in by_n[3]) == ["a", "b", "c"]
    assert sorted(seq.request_id for seq in by_n[2]) == ["d", "e"]


def test_scheduler_rejects_duplicate_request_id() -> None:
    sched = RuntimeScheduler()
    sched.add_sequence(Sequence(request_id="dup"))
    with pytest.raises(ValueError, match="duplicate request_id"):
        sched.add_sequence(Sequence(request_id="dup"))


def test_scheduler_max_num_seqs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_num_seqs must be >= 1"):
        RuntimeScheduler(max_num_seqs=0)


def test_scheduler_output_is_empty_when_idle() -> None:
    sched = RuntimeScheduler()
    out = sched.schedule()
    assert isinstance(out, RuntimeSchedulerOutput)
    assert out.is_empty


def test_runtime_group_carries_items_without_kind_flag() -> None:
    """A group is just a list of items; prefill vs decode is implied by which
    scheduler-output list it lives in."""
    prefill_group = RuntimeGroup(items=[])
    decode_group = RuntimeGroup(items=[])
    assert prefill_group.items == []
    assert decode_group.items == []
