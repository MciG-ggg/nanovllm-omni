"""Single-device Orchestrator (vllm-omni shape, no distributed plumbing).

vllm-omni's Orchestrator owns request lifecycle + stage-to-stage routing
across replica pools, running in a background thread with janus queues and
distributed membership. This module keeps the *shape* in the single-process,
single-GPU scope the project targets:

  - no background thread: ``submit()`` is synchronous and drives every
    replica to completion before returning
  - no janus queues / membership: each replica is a plain ``(scheduler,
    runner)`` pair built at pool construction
  - the lifecycle FSM is the ``RuntimeScheduler`` WAITING -> PREFILL ->
    DECODE -> FINISHED progression

Flow (single-stage MiniMind-O: the thinker pool, codec runs inside the
drive hook):

    Omni.generate(prompts)
        -> Orchestrator.submit(prompts)
        -> route each prompt to a replica via the LoadBalancer; drive every
           replica's schedulers to completion (codec chain included).
        -> [outputs in submission order]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nanovllm_omni.engine.load_balancer import LoadBalancer, RoundRobinBalancer


@dataclass
class Replica:
    """A single stage replica: one scheduler + one runner built over it.

    ``sched`` is the ``RuntimeScheduler``; ``runner`` is the model-level
    batched runner (e.g. ``BatchedThinkerRunner``) that executes the
    scheduler's groups. Mirrors vllm-omni's "one engine core per replica"
    without the distributed executor.
    """

    sched: Any
    runner: Any
    replica_id: int = 0


@dataclass
class StagePool:
    """Per-stage replica pool + the LoadBalancer routing requests to replicas."""

    stage_id: int
    num_replicas: int = 1
    balancer: LoadBalancer = field(default_factory=RoundRobinBalancer)
    replicas: list[Replica] = field(default_factory=list)

    def select(self) -> int:
        return self.balancer.select(self.stage_id, self.num_replicas)

    def add_replica(self, sched: Any, runner: Any) -> int:
        replica_id = len(self.replicas)
        self.replicas.append(Replica(sched=sched, runner=runner, replica_id=replica_id))
        return replica_id

    def __getitem__(self, replica_id: int) -> Replica:
        return self.replicas[replica_id]


class Orchestrator:
    """Dispatch over a single ``StagePool``: fan out, drain, return in order.

    ``submit(prompts)`` routes each prompt to a replica via the pool's
    LoadBalancer, then drives every replica's scheduler to completion via the
    ``drive`` hook. Returns ``[outputs]`` in submission order (the codec /
    downstream chain runs inside ``add_request`` / ``drive``).

    The ``drive`` and ``add_request`` hooks keep this class model-agnostic
    (same seam vllm-omni uses for per-stage executors):
      - ``add_request(replica, prompt) -> rid`` submits one request to a
        replica.
      - ``drive(replica) -> dict[rid, output]`` drains one replica's
        scheduler and returns its finished outputs.
    """

    def __init__(self, pool: StagePool) -> None:
        self.pool = pool

    def submit(
        self,
        prompts: list[Any],
        *,
        add_request: Callable[[Replica, Any], str],
        drive: Callable[[Replica], dict[str, Any]],
    ) -> list[Any]:
        """Route ``prompts`` through the pool's replicas; return outputs in order.

        ``add_request`` must mint globally-unique rids (per-replica scheduler
        ids collide across replicas); ``drive`` is called once per replica.
        """
        pool = self.pool
        order: list[str] = []
        for prompt in prompts:
            rid = add_request(pool[pool.select()], prompt)
            order.append(rid)

        by_rid: dict[str, Any] = {}
        for replica in pool.replicas:
            by_rid.update(drive(replica))

        return [by_rid[rid] for rid in order]


__all__ = ["Orchestrator", "Replica", "StagePool"]
