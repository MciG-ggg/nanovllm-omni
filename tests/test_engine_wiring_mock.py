"""Phase 3 engine wiring contract (skips if _engine.py not landed).

Tests use unittest.mock so no GPU / triton needed. If _engine.py
isn't there yet (Phase 3 hasn't landed), tests skip cleanly.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


def _get_engine():
    """Lazy import — only runs inside test functions, not at collection time."""
    try:
        import nanovllm_omni.models.minimind_omni._engine as mod

        return mod
    except Exception:
        return None


def test_shared_block_manager_exposed() -> None:
    engine = _get_engine()
    if engine is None:
        pytest.skip("_engine.py not importable (Phase 3 not landed or triton broken)")
    assert hasattr(
        engine, "SharedBlockManager"
    ), "Expected SharedBlockManager in _engine (ADR-002: shared block table)"


def test_stage_runner_exposed() -> None:
    engine = _get_engine()
    if engine is None:
        pytest.skip("_engine.py not importable")
    assert hasattr(
        engine, "StageRunner"
    ), "Expected StageRunner in _engine (ADR-002: per-stage runner)"


def test_stage_runner_lifecycle_methods() -> None:
    """StageRunner must expose set_context / forward / sample / reset_context."""
    engine = _get_engine()
    if engine is None:
        pytest.skip("_engine.py not importable")
    runner_cls = engine.StageRunner
    for name in ("set_context", "forward", "sample", "reset_context"):
        assert hasattr(runner_cls, name), f"StageRunner missing method {name!r}"
        assert callable(getattr(runner_cls, name)), f"StageRunner.{name} is not callable"


def test_stage_runner_constructs_with_mocked_runner() -> None:
    """StageRunner.__init__ accepts (model_class, config, shared_block_manager, rank=0).

    We mock ModelRunner so the constructor never hits CUDA / triton.
    """
    engine = _get_engine()
    if engine is None:
        pytest.skip("_engine.py not importable")

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

    runner = engine.StageRunner(model_class, config, shared_block_mgr, rank=0)

    assert runner is not None
    assert fake_mr.called, "StageRunner.__init__ did not invoke ModelRunner"


def test_forward_calls_underlying_runner() -> None:
    """StageRunner.forward delegates to ModelRunner.run_model."""
    engine = _get_engine()
    if engine is None:
        pytest.skip("_engine.py not importable")

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

    runner = engine.StageRunner(model_class, config, shared_block_mgr, rank=0)
    inner = runner.model_runner

    out = runner.forward(
        input_ids=mock.MagicMock(name="input_ids"),
        positions=mock.MagicMock(name="positions"),
        is_prefill=False,
    )
    assert out is not None
    assert inner.run_model.called, "StageRunner.forward did not route to ModelRunner.run_model"
