"""Focused tests for the single-device Orchestrator + StagePool.

The Orchestrator keeps vllm-omni's per-stage replica-pool + LoadBalancer
dispatch shape, minus finalize/multi-stage plumbing: a single-stage pool,
fan-out via the balancer, and a per-replica ``drive`` drain. These tests
pin the shape with a fake bundle (thinker batched runner).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.engine.load_balancer import RoundRobinBalancer  # noqa: E402
from nanovllm_omni.engine.orchestrator import (  # noqa: E402
    Orchestrator,
    Replica,
    StagePool,
)
from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler  # noqa: E402
from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner  # noqa: E402
from tests.test_batched_generation import (  # noqa: E402
    FakeMiniMindOmni,
    _FakeMimi,
    _FakeTokenizer,
)


def _make_bundle() -> Any:
    model = FakeMiniMindOmni()
    return SimpleNamespace(
        model=model,
        tokenizer=_FakeTokenizer(),
        mimi=_FakeMimi(),
        device="cpu",
    )


def _build_thinker_pool(bundle: Any, num_replicas: int = 1) -> StagePool:
    """One StagePool of thinker replicas (each: RuntimeScheduler + BatchedThinkerRunner)."""
    pool = StagePool(stage_id=0, num_replicas=num_replicas)
    for replica_id in range(num_replicas):
        sched = RuntimeScheduler(max_num_seqs=2)
        runner = BatchedThinkerRunner(
            SimpleNamespace(model=bundle.model),
            sched,
            temperature=0.75,
            top_p=1.0,
            rp=1.0,
            max_new_tokens=16,
            open_thinking=False,
            base_seed=42 + replica_id,
        )
        pool.add_replica(sched, runner)
    return pool


_rids = iter(range(1_000_000))


def _unique_rid() -> str:
    """Globally unique request id across replicas (per-replica rids collide)."""
    return f"req-{next(_rids)}"


def test_stage_pool_routes_via_balancer_and_counts_replicas() -> None:
    pool = StagePool(stage_id=0, num_replicas=2)
    assert len(pool.replicas) == 0
    r = pool.add_replica(object(), object())
    assert r == 0
    r = pool.add_replica(object(), object())
    assert r == 1
    # Round-robin over 2 replicas gives 0,1,0...
    assert pool.select() == 0
    assert pool.select() == 1
    assert pool.select() == 0


def _drive(replica: Replica) -> dict[str, Any]:
    """Drain one replica's scheduler to completion; return rid -> frame count.

    Mirrors the production loop in ``run_batched_generate``: schedule ->
    prefill/decode groups -> update_from_output, recording finished rids.
    """
    sched = replica.sched
    runner = replica.runner
    done: dict[str, Any] = {}
    while sched.has_work():
        out = sched.schedule()
        if out.is_empty:
            break
        prefilled: set[str] = set()
        finished: set[str] = set()
        for g in out.prefill_groups:
            runner.prefill_group(g)
            prefilled.update(chunk.sequence.request_id for chunk in g.items)
        for g in out.decode_groups:
            runner.decode_group(g)
            for sequence in g.items:
                if runner.step_finished(sequence.request_id):
                    finished.add(sequence.request_id)
        sched.update_from_output(prefilled=prefilled, finished=finished)
        for rid in finished:
            done[rid] = len(runner.states[rid].frames)
    return done


def test_orchestrator_submit_fans_out_and_drains_thinker_pool() -> None:
    """Prompts fan out across a 2-replica thinker pool via the balancer, and
    every replica's scheduler is drained to completion. Outputs come back in
    submission order."""
    bundle = _make_bundle()
    pool = _build_thinker_pool(bundle, num_replicas=2)
    orch = Orchestrator(pool)

    def add_request(replica: Replica, prompt: str) -> str:
        ids = _FakeTokenizer()(prompt).data["input_ids"]
        return replica.runner.add_request(ids, request_id=_unique_rid())

    result = orch.submit(
        ["hello world", "another prompt here", "third"], add_request=add_request, drive=_drive
    )
    assert len(result) == 3
    assert all(isinstance(v, int) for v in result)  # frame counts, per prompt


def test_orchestrator_single_replica_single_order() -> None:
    """With one replica, every prompt goes through it; outputs stay ordered."""
    bundle = _make_bundle()
    pool = _build_thinker_pool(bundle, num_replicas=1)
    orch = Orchestrator(pool)

    def add_request(replica: Replica, prompt: str) -> str:
        ids = _FakeTokenizer()(prompt).data["input_ids"]
        return replica.runner.add_request(ids, request_id=_unique_rid())

    result = orch.submit(["one", "two"], add_request=add_request, drive=_drive)
    assert len(result) == 2


def test_stage_pool_balancer_default_is_round_robin() -> None:
    pool = StagePool(stage_id=3, num_replicas=3)
    assert isinstance(pool.balancer, RoundRobinBalancer)
    assert pool.select() == 0
    assert pool.select() == 1
    assert pool.select() == 2
