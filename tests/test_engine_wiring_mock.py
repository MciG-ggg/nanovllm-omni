"""Phase 3 engine wiring contract (skips if stage_runner.py not landed).

Tests use unittest.mock so no GPU / triton needed. If stage_runner.py
isn't there yet (Phase 3 hasn't landed), tests skip cleanly.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


def _get_stage_runner():
    """Lazy import — only runs inside test functions, not at collection time.

    StageRunner / SharedBlockManager live in ``nanovllm_omni.engine.stage_runner``
    (canonical home; the historical re-exports through ``_engine`` were removed
    in review Step 1, and ``_engine`` itself was renamed to ``stage`` in the
    following refactor).
    """
    try:
        from nanovllm_omni.engine.stage_runner import SharedBlockManager, StageRunner

        return SharedBlockManager, StageRunner
    except Exception:
        return None


def test_shared_block_manager_exposed() -> None:
    wiring = _get_stage_runner()
    if wiring is None:
        pytest.skip("stage_runner.py not importable (Phase 3 not landed or triton broken)")
    shared_block_manager, _ = wiring
    assert (
        shared_block_manager is not None
    ), "Expected SharedBlockManager in stage_runner (ADR-002: shared block table)"


def test_stage_runner_exposed() -> None:
    wiring = _get_stage_runner()
    if wiring is None:
        pytest.skip("stage_runner.py not importable")
    _, stage_runner = wiring
    assert (
        stage_runner is not None
    ), "Expected StageRunner in stage_runner (ADR-002: per-stage runner)"


def test_stage_runner_lifecycle_methods() -> None:
    """StageRunner must expose set_context / forward / sample / reset_context."""
    wiring = _get_stage_runner()
    if wiring is None:
        pytest.skip("stage_runner.py not importable")
    _, stage_runner = wiring
    for name in ("set_context", "forward", "sample", "reset_context"):
        assert hasattr(stage_runner, name), f"StageRunner missing method {name!r}"
        assert callable(getattr(stage_runner, name)), f"StageRunner.{name} is not callable"


def test_stage_runner_constructs_with_mocked_runner() -> None:
    """StageRunner.__init__ accepts (model_class, config, shared_block_manager, rank=0).

    We mock ModelRunner so the constructor never hits CUDA / triton.
    """
    wiring = _get_stage_runner()
    if wiring is None:
        pytest.skip("stage_runner.py not importable")
    _, stage_runner = wiring

    fake_mr = mock.MagicMock(name="ModelRunner")
    fake_mr.return_value.run_model = mock.MagicMock(
        return_value=mock.MagicMock(name="logits"),
    )
    sys.modules["nanovllm.engine.model_runner"] = mock.MagicMock(ModelRunner=fake_mr)
    sys.modules["nanovllm.layers.sampler"] = mock.MagicMock()
    sys.modules["nanovllm.layers.sampler"].Sampler = mock.MagicMock(name="Sampler")

    model_class = mock.MagicMock(name="ModelClass")
    config = mock.MagicMock(name="Config")
    shared_block_mgr = mock.MagicMock(name="SharedBlockManager")

    runner = stage_runner(model_class, config, shared_block_mgr, rank=0)

    assert runner is not None
    assert fake_mr.called, "StageRunner.__init__ did not invoke ModelRunner"


def test_forward_calls_underlying_runner() -> None:
    """StageRunner.forward delegates to ModelRunner.run_model."""
    wiring = _get_stage_runner()
    if wiring is None:
        pytest.skip("stage_runner.py not importable")
    _, stage_runner = wiring

    fake_mr = mock.MagicMock(name="ModelRunner")
    fake_mr.return_value.run_model = mock.MagicMock(
        return_value=mock.MagicMock(name="logits"),
    )
    sys.modules["nanovllm.engine.model_runner"] = mock.MagicMock(ModelRunner=fake_mr)
    sys.modules["nanovllm.layers.sampler"] = mock.MagicMock()
    sys.modules["nanovllm.layers.sampler"].Sampler = mock.MagicMock(name="Sampler")

    model_class = mock.MagicMock(name="ModelClass")
    config = mock.MagicMock(name="Config")
    shared_block_mgr = mock.MagicMock(name="SharedBlockManager")

    runner = stage_runner(model_class, config, shared_block_mgr, rank=0)
    inner = runner.model_runner

    out = runner.forward(
        input_ids=mock.MagicMock(name="input_ids"),
        positions=mock.MagicMock(name="positions"),
        is_prefill=False,
    )
    assert out is not None
    assert inner.run_model.called, "StageRunner.forward did not route to ModelRunner.run_model"
