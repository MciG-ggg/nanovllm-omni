"""Phase 3 engine wiring contract (skips if ``_engine.py`` not landed yet).

ADR-002 commits the engine to:

- one ModelRunner per stage
- a shared BlockManager (block table is shared; KV caches stay per-stage)
- a StageRunner per stage that owns set_context / forward / sample / reset_context

This test does NOT exercise a real model.  We use ``unittest.mock`` for the
``ModelRunner`` so a wrong shape in the adapter would surface here on CI
without ever touching CUDA.  If ``_engine.py`` isn't there yet (Phase 3
hasn't landed), the test is silently skipped — that's the contract.

Usage of ``pytest.importorskip`` instead of ``find_spec``:
----------------------------------------------------------
``importorskip`` would force the parent package ``minimind_omni`` to be
imported, which loads the fork layers (SiluAndMul uses ``@torch.compile``)
and triggers torch._dynamo.  In a CPU-only test env without triton that
chain crashes before we can decide whether ``_engine`` exists.
``pytest.importorskip`` is the task-specified mechanism and works when
the conftest triton stubs include ``dtype`` (see conftest.py).  In CI /
clean envs it produces a tidy ``SKIPPED`` marker when the module is
missing.
"""

from __future__ import annotations

import inspect
from unittest import mock

import pytest

_ENGINE = pytest.importorskip(
    "nanovllm_omni.models.minimind_omni._engine",
    reason="Phase 3 _engine.py not landed yet — test will engage once it ships",
)


def test_shared_block_manager_exposed() -> None:
    """Shared block table lives on one object both stages route through."""
    assert hasattr(_ENGINE, "SharedBlockManager"), (
        "Expected SharedBlockManager in _engine (ADR-002: shared block table)"
    )


def test_stage_runner_exposed() -> None:
    assert hasattr(_ENGINE, "StageRunner"), (
        "Expected StageRunner in _engine (ADR-002: per-stage runner)"
    )


def test_stage_runner_lifecycle_methods() -> None:
    """StageRunner must expose set_context / forward / sample / reset_context."""
    runner_cls = _ENGINE.StageRunner
    for name in ("set_context", "forward", "sample", "reset_context"):
        assert hasattr(runner_cls, name), f"StageRunner missing method {name!r}"
        assert callable(getattr(runner_cls, name)), (
            f"StageRunner.{name} is not callable"
        )


def test_stage_runner_constructs_with_mocked_runner() -> None:
    """StageRunner.__init__ accepts (model_class, config, shared_block_manager, rank=0).

    We mock ModelRunner at the module level so the constructor never hits
    CUDA / triton / dist.init_process_group.  This is the contract the task
    asks for: "用 unittest.mock mock 掉 ModelRunner 的 GPU 操作".
    """
    # Patch ModelRunner before StageRunner.__init__ does its lazy import.
    fake_model_runner_mod = mock.MagicMock()
    fake_model_runner = mock.MagicMock(name="ModelRunner")
    fake_model_runner_mod.ModelRunner = fake_model_runner
    fake_model_runner_mod.return_value.run_model = mock.MagicMock(
        return_value=mock.MagicMock(name="logits"),
    )
    # nanovllm.engine.model_runner is the import path StageRunner uses.
    sys.modules["nanovllm.engine.model_runner"] = fake_model_runner_mod
    # Also patch the sampler path referenced in __init__.
    sys.modules["nanovllm.layers.sampler"] = mock.MagicMock()
    sys.modules["nanovllm.layers.sampler"].Sampler = mock.MagicMock(
        name="Sampler",
    )

    # Use documented positional args: model_class, config, shared_block_manager, rank=0
    model_class = mock.MagicMock(name="ModelClass")
    config = mock.MagicMock(name="Config")
    shared_block_mgr = mock.MagicMock(name="SharedBlockManager")

    runner = _ENGINE.StageRunner(model_class, config, shared_block_mgr, rank=0)

    assert runner is not None
    # Verify the patched ModelRunner was called exactly once by __init__.
    assert fake_model_runner.called, (
        "StageRunner.__init__ did not invoke ModelRunner (check the mock setup)"
    )


def test_forward_calls_underlying_runner() -> None:
    """StageRunner.forward must delegate to ``ModelRunner.run_model``.

    The fork exposes ``run_model(input_ids, positions, is_prefill)``;
    if Phase 3 routed to something else, this test fails loudly without GPU.
    """
    # Patch ModelRunner before construction so StageRunner.__init__ loads our stub.
    fake_model_runner = mock.MagicMock(name="ModelRunner")
    fake_model_runner.return_value.run_model = mock.MagicMock(
        return_value=mock.MagicMock(name="logits"),
    )
    sys.modules["nanovllm.engine.model_runner"] = mock.MagicMock(
        ModelRunner=fake_model_runner,
    )
    sys.modules["nanovllm.layers.sampler"] = mock.MagicMock()
    sys.modules["nanovllm.layers.sampler"].Sampler = mock.MagicMock(name="Sampler")

    model_class = mock.MagicMock(name="ModelClass")
    config = mock.MagicMock(name="Config")
    shared_block_mgr = mock.MagicMock(name="SharedBlockManager")

    runner = _ENGINE.StageRunner(model_class, config, shared_block_mgr, rank=0)
    inner = runner.model_runner  # the MagicMock that StageRunner stored

    out = runner.forward(
        input_ids=mock.MagicMock(name="input_ids"),
        positions=mock.MagicMock(name="positions"),
        is_prefill=False,
    )
    assert out is not None
    # The call path went through ModelRunner.run_model:
    assert inner.run_model.called, (
        "StageRunner.forward did not route to ModelRunner.run_model"
    )