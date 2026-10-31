"""The thinker stage reads the bench's CUDA-Graph knob.

``--use-thinker-cuda-graph`` reaches the stage as
``OmniEngineArgs.extra["use_thinker_cuda_graph"]`` (unknown kwargs land
in ``extra``). The stage turns it into the fork ``Config``'s
``enforce_eager``: graph by default (3.8x on RTX 3050 with byte-identical
output, ADR-0005 amendment 2026-10-30), eager when the bench asks for the
comparison cell.

The fork layers are heavy, so the stage is built with the three fork
touch points stubbed: ``get_stage_config`` (returns the config the
stage mutates), ``StageRunner`` (owns the GPU model) and
``_ensure_dist`` (needs a process group).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from nanovllm_omni.models.minimind_omni import stage as stage_mod


def _build_stage(monkeypatch: pytest.MonkeyPatch, extra: dict) -> object:
    config = SimpleNamespace(enforce_eager=None)
    monkeypatch.setattr(stage_mod, "_resolve_model_path", lambda *a, **k: "/tmp/model")
    monkeypatch.setattr(stage_mod, "get_stage_config", lambda *a, **k: config)
    monkeypatch.setattr(stage_mod, "_ensure_dist", lambda: None)
    monkeypatch.setattr(stage_mod, "StageRunner", lambda **kw: SimpleNamespace(**kw))
    args = SimpleNamespace(model="/tmp/model", device="cpu", extra=extra)
    thinker = stage_mod.ThinkerStage(deploy=None, args=args)
    return thinker, config


def _captured_stage_kwargs(monkeypatch: pytest.MonkeyPatch, extra: dict) -> dict:
    """Build the stage and return the fork ``Config`` kwargs it asked for."""
    config = SimpleNamespace(enforce_eager=None)
    seen: dict = {}
    monkeypatch.setattr(stage_mod, "_resolve_model_path", lambda *a, **k: "/tmp/model")
    monkeypatch.setattr(stage_mod, "get_stage_config", lambda *a, **k: seen.update(k) or config)
    monkeypatch.setattr(stage_mod, "_ensure_dist", lambda: None)
    monkeypatch.setattr(stage_mod, "StageRunner", lambda **kw: SimpleNamespace(**kw))
    args = SimpleNamespace(model="/tmp/model", device="cpu", extra=extra)
    stage_mod.ThinkerStage(deploy=None, args=args)
    return seen


def test_thinker_uses_the_graph_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    thinker, config = _build_stage(monkeypatch, {})

    assert thinker.config.enforce_eager is False


def test_thinker_eager_flag_selects_the_comparison_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinker, config = _build_stage(monkeypatch, {"use_thinker_cuda_graph": False})

    assert thinker.config.enforce_eager is True


def test_thinker_captures_only_the_single_request_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capturing more buckets changes the bs=1 graph's arithmetic.

    The fork captures every bucket into one shared memory pool, so a
    bucket captured alongside bigger ones reduces in a different order and
    the bridge rows stop matching the eager path (ADR-0012).
    """
    kwargs = _captured_stage_kwargs(monkeypatch, {})

    assert kwargs["max_capture_bs"] == 1


def test_thinker_capture_bucket_cap_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _captured_stage_kwargs(monkeypatch, {"thinker_max_capture_bs": 8})

    assert kwargs["max_capture_bs"] == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
