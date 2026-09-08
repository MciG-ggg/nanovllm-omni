"""Contract tests for the minimal batched engine path (post-orchestration-cut).

The orchestration layer (``StagePool`` / ``Orchestrator`` / ``LoadBalancer``
/ ``num_replicas`` / multi-replica) was removed after measuring zero
contribution on the single-GPU scope (docs/perf/model-family-bottlenecks-2026-08-27.md
appendix: num_replicas=1 == num_replicas=2). These tests pin the surviving
public contract of ``run_batched_generate``:

1. no ``num_replicas`` / ``balancer`` / ``Orchestrator`` knobs remain
2. one scheduler drains all prompts; outputs come back in submission order
3. ``max_batch`` still defaults from ``deploy.max_batch`` (kept in
   ``tests/test_batched_generation.py``)
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.batched_runner import run_batched_generate  # noqa: E402
from nanovllm_omni.outputs import AudioPayload  # noqa: E402
from tests.test_batched_generation import FakeMiniMindOmni, _FakeMimi, _FakeTokenizer  # noqa: E402


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        model=FakeMiniMindOmni(),
        tokenizer=_FakeTokenizer(),
        mimi=_FakeMimi(),
        device="cpu",
    )


def test_run_batched_generate_signature_has_no_orchestration_knobs() -> None:
    params = inspect.signature(run_batched_generate).parameters
    # deploy stays (max_batch default source); only orchestration knobs must be gone.
    assert "num_replicas" not in params
    assert "balancer" not in params


def test_run_batched_generate_returns_outputs_in_submission_order() -> None:
    out = run_batched_generate(
        _bundle(),
        ["hello world", "another prompt here", "third and final one"],
        max_batch=2,
        max_new_tokens=12,
        base_seed=42,
    )
    assert len(out) == 3
    for payload in out:
        assert isinstance(payload, AudioPayload)
