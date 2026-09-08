"""Per-stage runtime scheduler.

Mirrors the reference's per-stage ``Scheduler`` shape:

  - one ``RuntimeScheduler`` instance per LLM stage (thinker, talker)
  - each instance owns its own waiting / running / finished queues
  - ``schedule()`` returns ``prefill_chunks`` + ``decode_groups``
  - ``update_from_output()`` advances sequence status from one round

The MiniMind-O thinker stage does NOT need a separate scheduler instance
because thinker and talker share the same underlying model in memory (the
pipeline just calls the model twice with different ``process_input``
shaping). Each LLM-style stage factory instantiates one scheduler when
constructed.

Deliberately not a subclass of ``OmniScheduler`` (which is the global
mini-mind-specific scheduler in ``sched.py``); the design calls for a
per-stage model with ``Sequence``-shaped state, and ``OmniScheduler`` is
the request-grouping + FSM convenience used by
``BatchedThinkerRunner``. The two are linked: ``OmniRequest`` carries a
``Sequence`` in vLLM's shape; ``RuntimeScheduler`` operates on the
``Sequence`` directly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .sequence import OmniSequence, PrefillChunk, SequenceStatus


@dataclass
class RuntimeGroup:
    """One unit of work a single stage forward can execute.

    A prefill group is a list of ``PrefillChunk`` (chunked prefill supported
    via the ``start``/``end`` range); a decode group is a list of
    ``Sequence`` whose current ``num_tokens`` matches a single stage
    forward's single-scalar ``start_pos``. The caller already distinguishes
    prefill from decode by which list it reads (``prefill_groups`` vs
    ``decode_groups``).
    """

    items: list[Any] = field(default_factory=list)  # list[PrefillChunk] | list[OmniSequence]


@dataclass
class RuntimeSchedulerOutput:
    """What ``schedule()`` produced for one round, in execution order."""

    prefill_groups: list[RuntimeGroup] = field(default_factory=list)
    decode_groups: list[RuntimeGroup] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_groups and not self.decode_groups


class RuntimeScheduler:
    """Per-stage scheduler with Sequence state.

    Lifecycle for a single sequence: WAITING -> (admit) -> PREFILL ->
    (prefill round done) -> DECODE -> (decode round done) -> DECODE -> ...
    -> (finished) -> FINISHED.
    """

    def __init__(self, *, max_num_seqs: int = 2) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[OmniSequence] = deque()
        self.running: dict[str, OmniSequence] = {}
        self.finished: dict[str, OmniSequence] = {}

    # -- submission ----------------------------------------------------------

    def add_sequence(self, sequence: OmniSequence) -> None:
        """Admit a new sequence (typically a freshly-constructed request)."""
        if (
            sequence.request_id in self.running
            or sequence.request_id in self.finished
            or any(s.request_id == sequence.request_id for s in self.waiting)
        ):
            raise ValueError(f"duplicate request_id {sequence.request_id!r}")
        self.waiting.append(sequence)

    # -- queries -------------------------------------------------------------

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def get(self, rid: str) -> OmniSequence | None:
        if rid in self.running:
            return self.running[rid]
        return self.finished.get(rid)

    def is_finished(self, rid: str) -> bool:
        return rid in self.finished

    # -- scheduler core ------------------------------------------------------

    def schedule(self) -> RuntimeSchedulerOutput:
        """Form prefill chunks (waiting) and decode groups (running, by len)."""
        out = RuntimeSchedulerOutput()

        # Admission: fill running up to max_num_seqs with waiting sequences.
        while self.waiting and len(self.running) < self.max_num_seqs:
            sequence = self.waiting.popleft()
            sequence.status = SequenceStatus.PREFILL
            self.running[sequence.request_id] = sequence

        # Prefill chunks: every PREFILL sequence becomes a single chunk covering
        # its full prompt. Partition by ``num_tokens`` so the model forward sees
        # a rectangular [B, 9, P] tensor (one length per prefill group). Chunked
        # prefill (start > 0) is supported by the data structure but not emitted
        # here.
        if self.running:
            decode_items: list[OmniSequence] = []
            prefill_by_len: dict[int, list[PrefillChunk]] = {}
            for sequence in self.running.values():
                if sequence.status is SequenceStatus.PREFILL:
                    prefill_by_len.setdefault(sequence.num_tokens, []).append(
                        PrefillChunk(sequence=sequence, start=0, end=sequence.num_tokens)
                    )
                elif sequence.status is SequenceStatus.DECODE:
                    decode_items.append(sequence)
            for _n_pos, chunks in sorted(prefill_by_len.items()):
                out.prefill_groups.append(RuntimeGroup(items=chunks))
            # Decode groups: identical num_tokens -> one rectangular forward.
            by_len: dict[int, list[OmniSequence]] = {}
            for sequence in decode_items:
                by_len.setdefault(sequence.num_tokens, []).append(sequence)
            for _num_tokens, group in sorted(by_len.items()):
                out.decode_groups.append(RuntimeGroup(items=group))

        return out

    def update_from_output(
        self,
        *,
        prefilled: list[str] | None = None,
        finished: list[str] | None = None,
    ) -> dict[str, int]:
        """Advance per-sequence status from one round's runner output.

        ``prefilled``: request_ids whose prefill forward just wrote KV.
        ``finished``: request_ids whose stage generation is done.
        Returns ``{rid: num_tokens}`` for the just-finished sequences, so the
        caller can iterate them in submission order for codec hand-off.
        """
        for rid in prefilled or ():
            sequence = self.running.get(rid)
            if sequence is not None and sequence.status is SequenceStatus.PREFILL:
                sequence.status = SequenceStatus.DECODE
        just_finished: dict[str, int] = {}
        for rid in finished or ():
            sequence = self.running.pop(rid, None)
            if sequence is not None:
                sequence.status = SequenceStatus.FINISHED
                self.finished[rid] = sequence
                just_finished[rid] = sequence.num_tokens
        return just_finished


__all__ = ["RuntimeGroup", "RuntimeScheduler", "RuntimeSchedulerOutput"]
