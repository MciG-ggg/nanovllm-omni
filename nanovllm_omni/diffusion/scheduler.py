"""RequestScheduler: lightweight FIFO scheduler for diffusion requests.

Uses stdlib ``collections.deque`` — no paged KV, no priority, no
preemption.  Matches ADR-023 from the migration doc.

See ``docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md`` §3 ADR-023.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ScheduledOutput:
    """Output of one ``schedule()`` call — the request to process this tick."""

    request_id: str
    step_id: int
    request: Any = None


class RequestScheduler:
    """FIFO scheduler with waiting / running queues.

    Single-request scope: at most one request runs at a time.
    No capacity limit beyond that.
    """

    def __init__(self) -> None:
        self._waiting: deque[Any] = deque()
        self._running: dict[str, Any] = {}  # request_id → request
        self._step_counters: dict[str, int] = {}

    def add_request(self, request: Any) -> None:
        """Enqueue a request for processing."""
        self._waiting.append(request)

    def has_requests(self) -> bool:
        """True if there are waiting or running requests."""
        return bool(self._waiting) or bool(self._running)

    def schedule(self) -> ScheduledOutput | None:
        """Pick the next request to run.  Returns None if nothing to do."""
        # Finish any completed running requests first.
        if not self._waiting and not self._running:
            return None
        # If nothing is running, promote from waiting.
        if not self._running and self._waiting:
            req = self._waiting.popleft()
            rid = getattr(req, "request_id", id(req))
            self._running[rid] = req
            self._step_counters[rid] = 0
            return ScheduledOutput(
                request_id=rid,
                step_id=0,
                request=req,
            )
        # Something is running — return its next step.
        if self._running:
            rid = next(iter(self._running))
            step = self._step_counters[rid]
            return ScheduledOutput(
                request_id=rid,
                step_id=step,
                request=self._running[rid],
            )
        return None

    def update_from_output(self, output: ScheduledOutput, _result: Any) -> None:
        """Advance the step counter after a successful denoise step."""
        self._step_counters[output.request_id] = output.step_id + 1

    def finish(self, request_id: str) -> None:
        """Remove a request from the running queue."""
        self._running.pop(request_id, None)
        self._step_counters.pop(request_id, None)

    def num_running(self) -> int:
        return len(self._running)

    def num_waiting(self) -> int:
        return len(self._waiting)


__all__ = ["RequestScheduler", "ScheduledOutput"]
