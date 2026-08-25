"""Single-device Orchestrator (vllm-omni shape, no distributed plumbing).

vllm-omni's Orchestrator owns request lifecycle + stage-to-stage routing
across replica pools, running in a background thread with janus queues and
distributed membership. This module keeps the *shape* in the single-process,
single-GPU scope the project targets:

  - no background thread: ``submit()`` is synchronous and drives every stage
    pool to completion before returning
  - no janus queues / membership: each replica is a plain ``(scheduler,
    runner)`` pair built at pool construction
  - the lifecycle FSM is each stage's ``RuntimeScheduler`` WAITING ->
    PREFILL -> DECODE -> FINISHED progression

Flow (mirrors vllm-omni's request-through-stages):

    Omni.generate(prompts)
        -> Orchestrator.submit(prompts)
        -> stage0 (LLM/thinker): route each prompt to a replica via
           LoadBalancer; drive the pool's schedulers to completion. Emits
           per-rid intermediate outputs (e.g. Mimi codebook frames).
        -> stage1 (codec): for each finished rid, run the codec stage to
           produce the final payload (WAV bytes).
        -> [{rid: output}]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nanovllm_omni.engine.load_balancer import LoadBalancer, RoundRobinBalancer


@dataclass
class Replica:
    """A single stage replica: one scheduler + one runner built over it.

    ``sched`` is the per-replica ``RuntimeScheduler``; ``runner`` is the
    model-level batched runner (e.g. ``BatchedThinkerRunner``) that executes
    the scheduler's groups. Mirrors vllm-omni's "one engine core per
    replica" without the distributed executor.
    """

    sched: Any
    runner: Any
    replica_id: int = 0

    @property
    def states(self) -> dict[str, Any]:
        return getattr(self.runner, "states", {})


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
    """Pipeline-level dispatcher over one or more ``StagePool``s.

    ``submit(prompts)`` fans each prompt across the stage-0 pool via its
    LoadBalancer, drains every replica's scheduler (the ``drive`` hook), then
    passes each finished rid through the remaining stages. Returns a dict
    ``rid -> final_output``.

    The ``drive`` and ``finalize`` hooks keep this class model-agnostic
    (same seam vllm-omni uses for per-stage executors):
      - ``drive(replica, finished_rids) -> dict[rid, output]`` runs one
        replica's scheduler loop and returns its finished outputs.
      - ``finalize(stage_id, rid, intermediate) -> output`` transforms one
        intermediate into the next stage's input / the final payload.
    """

    def __init__(self, pools: list[StagePool]) -> None:
        if not pools:
            raise ValueError("Orchestrator needs at least one StagePool")
        self.pools = pools

    def submit(
        self,
        prompts: list[Any],
        *,
        add_request: Callable[[Replica, Any], str],
        drive: Callable[[Replica, list[str]], dict[str, Any]],
        finalize: Callable[[int, str, Any], Any] | None = None,
    ) -> dict[str, Any]:
        """Route ``prompts`` through stage 0, then downstream stages.

        ``add_request(replica, prompt) -> rid`` submits one request to a
        replica. ``drive(replica, rids)`` drains that replica's scheduler for
        the given rids, returning ``{rid: intermediate}``. ``finalize``
        transforms intermediates through the remaining stages; when ``None``
        the stage-0 output is returned as-is.
        """
        pool = self.pools[0]
        order: list[str] = []
        entered: dict[int, list[str]] = {r.replica_id: [] for r in pool.replicas}
        for prompt in prompts:
            replica_id = pool.select()
            replica = pool[replica_id]
            rid = add_request(replica, prompt)
            order.append(rid)
            entered[replica_id].append(rid)

        stage0_out: dict[str, Any] = {}
        for replica in pool.replicas:
            stage0_out.update(drive(replica, entered[replica.replica_id]))

        if finalize is None:
            self._order = order
            return stage0_out

        full: dict[str, Any] = {}
        for rid in order:
            intermediate = stage0_out[rid]
            out = intermediate
            for stage_id in range(1, len(self.pools)):
                out = finalize(stage_id, rid, out)
            full[rid] = out
        self._order = order
        return full


__all__ = ["Orchestrator", "Replica", "StagePool"]
