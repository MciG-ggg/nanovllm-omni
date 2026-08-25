"""Focused tests for the single-device Orchestrator + StagePool.

The Orchestrator mirrors vllm-omni's request-through-stages shape: per-stage
replica pools, LoadBalancer dispatch, and a stage-0 drain followed by a
finalize pass over downstream stages. These tests pin the shape with a fake
bundle (thinker batched runner + a trivial codec finalize).
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


_rid_counter = {"n": 0}


def _unique_rid() -> str:
    """Globally unique request id across replicas (per-replica rids collide)."""
    rid = f"req-{_rid_counter['n']}"
    _rid_counter["n"] += 1
    return rid


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


def test_orchestrator_submit_fans_out_and_drains_thinker_pool() -> None:
    """Prompts fan out across a 2-replica thinker pool via the balancer, and
    every replica's scheduler is drained to completion."""
    _rid_counter["n"] = 0
    bundle = _make_bundle()
    pool = _build_thinker_pool(bundle, num_replicas=2)

    orch = Orchestrator([pool])

    def add_request(replica: Replica, prompt: str) -> str:
        ids = _FakeTokenizer()(prompt).data["input_ids"]
        return replica.runner.add_request(ids, request_id=_unique_rid())

    def drive(replica: Replica, rids: list[str]) -> dict[str, Any]:
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
                prefilled.update(chunk.seq.request_id for chunk in g.items)
            for g in out.decode_groups:
                runner.decode_group(g)
                for seq in g.items:
                    if runner.step_finished(seq.request_id):
                        finished.add(seq.request_id)
            sched.update_from_output(prefilled=prefilled, finished=finished)
            # record anything finished this round
            for rid in finished:
                done[rid] = len(runner.states[rid].frames)
        return done

    result = orch.submit(
        ["hello world", "another prompt here", "third"], add_request=add_request, drive=drive
    )
    assert len(result) == 3
    # Submission order is preserved on ``_order``; ``result`` is keyed by rid.
    assert orch._order == ["req-0", "req-1", "req-2"]
    assert set(result.keys()) == set(orch._order)


def test_orchestrator_finalize_runs_downstream_stages() -> None:
    """A finalize callable transforms stage-0 output into downstream-stage
    payloads; the Orchestrator applies it per finished rid in order."""
    _rid_counter["n"] = 0
    bundle = _make_bundle()
    pool = _build_thinker_pool(bundle, num_replicas=1)
    # Fake downstream stage: codec -> prefix "wav:" to frame count.
    pool2 = StagePool(stage_id=1, num_replicas=1)
    orch = Orchestrator([pool, pool2])

    def add_request(replica: Replica, prompt: str) -> str:
        ids = _FakeTokenizer()(prompt).data["input_ids"]
        return replica.runner.add_request(ids, request_id=_unique_rid())

    def drive(replica: Replica, rids: list[str]) -> dict[str, Any]:
        sched = replica.sched
        runner = replica.runner
        done: dict[str, Any] = {}
        while sched.has_work():
            out = sched.schedule()
            if out.is_empty:
                break
            prefilled, finished = set(), set()
            for g in out.prefill_groups:
                runner.prefill_group(g)
                prefilled.update(chunk.seq.request_id for chunk in g.items)
            for g in out.decode_groups:
                runner.decode_group(g)
                for seq in g.items:
                    if runner.step_finished(seq.request_id):
                        finished.add(seq.request_id)
            sched.update_from_output(prefilled=prefilled, finished=finished)
            for rid in finished:
                done[rid] = len(runner.states[rid].frames)
        return done

    def finalize(stage_id: int, rid: str, intermediate: Any) -> Any:
        # stage 1 (codec): wrap the frame count
        return f"wav:{intermediate}"

    result = orch.submit(
        ["hello world", "another"],
        add_request=add_request,
        drive=drive,
        finalize=finalize,
    )
    assert len(result) == 2
    assert all(v.startswith("wav:") for v in result.values())


def test_orchestrator_requires_at_least_one_pool() -> None:
    with pytest.raises(ValueError, match="at least one StagePool"):
        Orchestrator([])


def test_replica_exposes_states() -> None:
    bundle = _make_bundle()
    pool = _build_thinker_pool(bundle, num_replicas=1)
    replica = pool[0]
    assert hasattr(replica, "sched")
    assert hasattr(replica, "runner")
    assert replica.states == {}


def test_stage_pool_balancer_default_is_round_robin() -> None:
    pool = StagePool(stage_id=3, num_replicas=3)
    assert isinstance(pool.balancer, RoundRobinBalancer)
    # original pool select() delegates to the shared balancer
    assert pool.select() == 0
    assert pool.select() == 1
    assert pool.select() == 2
