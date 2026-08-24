"""Per-stage replica load balancer (TK-007).

Mirrors vllm-omni's ``distributed/omni_coordinator/load_balancer.py`` shape --
an ABC with concrete strategies. nanovllm-omni is single-device, so the
overhead here is the teaching minimum: one strategy (round-robin) on top of
the ABC, no distributed membership. ``num_replicas`` comes from
``StageConfig.num_replicas`` (default 1, so ``select`` always returns 0 and
single-replica behaviour is unchanged).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict


class LoadBalancer(ABC):
    """Pick a replica id for a request entering a stage.

    The interface is the vllm-omni seam: ``select(stage_id, num_replicas)``
    returns an int in ``[0, num_replicas)``. Adding a strategy (random,
    least-queue-length) is a subclass, not a change to callers.
    """

    @abstractmethod
    def select(self, stage_id: int, num_replicas: int) -> int:
        raise NotImplementedError


class RoundRobinBalancer(LoadBalancer):
    """Distribute requests round-robin across a stage's replicas.

    Per-stage cursor so each stage's requests are balanced independently.
    Thread-safe enough for the single-device GIL-held dispatch path (the
    cursor increment is a couple of bytecodes).
    """

    def __init__(self) -> None:
        self._cursor: defaultdict[int, int] = defaultdict(int)

    def select(self, stage_id: int, num_replicas: int) -> int:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        idx = self._cursor[stage_id] % num_replicas
        self._cursor[stage_id] = idx + 1
        return int(idx)


__all__ = ["LoadBalancer", "RoundRobinBalancer"]
