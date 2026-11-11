"""Stage engine: the queue of requests waiting for one pipeline stage.

A stage owns its queue and decides how many requests it serves per
``step``. That is the split ``vllm_omni`` makes — a ``StagePool`` holds
each stage's queue and that stage's own scheduler picks the batch —
reduced to one process, because a single GPU runs one stage at a time.

Every ``step`` is incremental: it advances the work it already holds by
one unit and returns only the requests that finished in that unit. A stage
whose backend batches keeps its own scheduler across steps and re-selects
the batch every step, so a request added between steps joins the running
batch. The default policy serves one request per step to completion; how
many a batching stage can really run together is its own ``max_num_seqs``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageRequest:
    """One request waiting in a stage queue."""

    request_id: str
    payload: Any
    sampling: Any


class StageEngine:
    """Base stage: serves its queued requests one at a time.

    Single-threaded: the queue and ``step`` are not atomic as a pair, so
    every caller must hold the same lock (``PipelineExecutor`` does).
    """

    def __init__(self) -> None:
        self._pending: deque[StageRequest] = deque()

    def add_request(self, request: StageRequest) -> None:
        """Queue one request for this stage."""
        self._pending.append(request)

    def has_pending(self) -> bool:
        """Whether the stage holds unfinished work.

        A batching stage has in-flight work after its queue drains, so it
        overrides this to report it. The runner skips a stage that returns
        False, and a skipped in-flight batch would never advance.
        """
        return bool(self._pending)

    def step(self) -> list[tuple[str, Any]]:
        """Advance this stage by one increment; return what finished in it.

        Returns ``(request_id, payload)`` pairs. The base increment is one
        whole request: it is popped only after ``__call__`` returns, so a
        failure leaves it queued for ``drop_pending`` to clean up instead
        of stranding it. A batching stage overrides this and returns only
        the subset that reached completion in this step.
        """
        if not self._pending:
            return []
        request = self._pending[0]
        output = self(request.payload, request.sampling)
        self._pending.popleft()
        return [(request.request_id, output)]

    def drop_pending(self) -> list[str]:
        """Forget the queued requests; returns their ids."""
        request_ids = [request.request_id for request in self._pending]
        self._pending.clear()
        return request_ids


__all__ = ["StageEngine", "StageRequest"]
