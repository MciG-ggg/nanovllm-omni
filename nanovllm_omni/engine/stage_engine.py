"""Stage engine: the queue of requests waiting for one pipeline stage.

A stage owns its queue and decides how many requests it serves per
``step``. That is the split ``vllm_omni`` makes — a ``StagePool`` holds
each stage's queue and that stage's own scheduler picks the batch —
reduced to one process, because a single GPU runs one stage at a time.

The default policy serves one request per step. A stage whose backend
batches overrides ``step`` and takes its whole queue at once; how many
the backend can really run together is its own ``max_num_seqs``.
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
        """Whether a request is waiting for this stage."""
        return bool(self._pending)

    def step(self) -> list[tuple[str, Any]]:
        """Serve the queued requests this stage takes per step.

        Returns ``(request_id, payload)`` pairs in queue order. The request
        is popped only after ``__call__`` returns, so a failure leaves it
        queued for ``drop_pending`` to clean up instead of stranding it.
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
