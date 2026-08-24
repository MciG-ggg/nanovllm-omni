"""Focused tests for the TK-007 LoadBalancer.

The spec requires a LoadBalancer ABC + at least one strategy (round-robin),
with single-replica default behaviour unchanged. These tests pin the
round-robin sequence, per-stage isolation, the ABC's abstract method, and
the num_replicas guard.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.engine.load_balancer import LoadBalancer, RoundRobinBalancer


def test_round_robin_cycles_through_replicas() -> None:
    """With num_replicas=3, consecutive selects give 0,1,2,0,1,2,..."""
    lb = RoundRobinBalancer()
    picks = [lb.select(stage_id=0, num_replicas=3) for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2]


def test_round_robin_single_replica_is_always_zero() -> None:
    """num_replicas=1 (the default) -> every stage_id selects 0 (back-compat)."""
    lb = RoundRobinBalancer()
    for stage in range(4):
        assert lb.select(stage_id=stage, num_replicas=1) == 0


def test_round_robin_balances_stages_independently() -> None:
    """Per-stage cursor: stage 0 and stage 1 do not interfere."""
    lb = RoundRobinBalancer()
    picks = [lb.select(stage_id=s, num_replicas=2) for s in (0, 1, 0, 1, 0)]
    # stage 0 (3 calls) -> 0,1,0 ; stage 1 (2 calls) -> 0,1
    assert picks == [0, 0, 1, 1, 0]


def test_round_robin_rejects_zero_replicas() -> None:
    lb = RoundRobinBalancer()
    with pytest.raises(ValueError, match="num_replicas must be >= 1"):
        lb.select(stage_id=0, num_replicas=0)


def test_load_balancer_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        LoadBalancer()  # type: ignore[abstract]


def test_concrete_strategy_is_a_load_balancer() -> None:
    assert isinstance(RoundRobinBalancer(), LoadBalancer)
