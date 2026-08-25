"""TK-007 acceptance: per-replica scheduler isolation + round-robin routing.

With ``num_replicas > 1`` the engine instantiates one ``RuntimeScheduler``
+ one ``BatchedThinkerRunner`` per replica and the ``LoadBalancer`` picks
which replica takes each prompt. Each replica runs its own scheduler so
two replicas can independently batch their own requests.

Tests pin the round-robin assignment (replica_id per prompt) and that the
engine loop drives both replicas' schedulers until both are empty.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.engine.load_balancer import RoundRobinBalancer  # noqa: E402
from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler  # noqa: E402
from tests.test_batched_generation import (  # noqa: E402
    FakeMiniMindOmni,
    _FakeMimi,
    _FakeTokenizer,
)


class _RecordBalancer(RoundRobinBalancer):
    """RoundRobinBalancer that records every ``select`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int, int]] = []

    def select(self, stage_id: int, num_replicas: int) -> int:
        idx = super().select(stage_id, num_replicas)
        self.calls.append((stage_id, num_replicas, idx))
        return idx


def _make_bundle() -> Any:
    model = FakeMiniMindOmni()
    return SimpleNamespace(
        model=model,
        tokenizer=_FakeTokenizer(),
        mimi=_FakeMimi(),
        device="cpu",
    )


def test_balancer_picks_replica_in_round_robin_order() -> None:
    """The balancer sees stage_id=0 (thinker) and a ``num_replicas=2`` knob."""
    lb = _RecordBalancer()
    picks = [lb.select(stage_id=0, num_replicas=2) for _ in range(4)]
    # Round-robin over 2 replicas: 0, 1, 0, 1.
    assert picks == [0, 1, 0, 1]
    # All calls tagged stage 0; the engine loop can dispatch by replica_id.
    assert all(stage == 0 for stage, _, _ in lb.calls)
    assert all(n == 2 for _, n, _ in lb.calls)


def test_run_batched_generate_uses_per_replica_runner_when_num_replicas_2() -> None:
    """``num_replicas=2`` drives two independent schedulers + runners.

    Each replica's scheduler gets only its assigned requests; the engine loop
    drains both before returning. (Existence + correctness via payloads.)
    """
    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.outputs import AudioPayload

    bundle = _make_bundle()
    lb = _RecordBalancer()
    out = run_batched_generate(
        bundle,
        ["hello world", "another prompt here", "third prompt"],
        max_batch=2,
        max_new_tokens=12,
        base_seed=42,
        num_replicas=2,
        balancer=lb,
    )
    assert len(out) == 3
    assert all(isinstance(p, AudioPayload) for p in out)
    # Three prompts with two replicas: assignments are 0, 1, 0.
    replica_ids = [idx for _, _, idx in lb.calls]
    assert replica_ids == [0, 1, 0]


def test_run_batched_generate_single_replica_default_unchanged() -> None:
    """Default ``num_replicas=1`` keeps the pre-TK-007 execution shape."""
    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.outputs import AudioPayload

    bundle = _make_bundle()
    out = run_batched_generate(
        bundle,
        ["hello world", "another prompt here"],
        max_batch=2,
        max_new_tokens=12,
        base_seed=42,
        # num_replicas omitted -> defaults to 1
    )
    assert len(out) == 2
    assert all(isinstance(p, AudioPayload) for p in out)
    assert all(p.data[:4] == b"RIFF" for p in out)


def test_each_replica_has_its_own_runtime_scheduler_instance() -> None:
    """Two replicas -> two independent ``RuntimeScheduler`` objects (no aliasing)."""
    from nanovllm_omni.engine.batched_runner import run_batched_generate

    bundle = _make_bundle()
    # Reach into the engine by patching RuntimeScheduler + checking identity.
    seen: list[RuntimeScheduler] = []

    from nanovllm_omni.engine import batched_runner as br

    original_init = br.RuntimeScheduler.__init__

    def traced_init(self, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(self)
        original_init(self, **kwargs)

    br.RuntimeScheduler.__init__ = traced_init  # type: ignore[method-assign]
    try:
        run_batched_generate(
            bundle,
            ["a", "b"],
            max_batch=2,
            max_new_tokens=8,
            base_seed=7,
            num_replicas=2,
            balancer=RoundRobinBalancer(),
        )
    finally:
        br.RuntimeScheduler.__init__ = original_init  # type: ignore[method-assign]

    assert len(seen) == 2
    # Each replica gets a distinct scheduler instance (no aliasing).
    assert seen[0] is not seen[1]
