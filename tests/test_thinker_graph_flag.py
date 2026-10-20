"""The thinker stage reads the bench's CUDA-Graph knob.

``--use-thinker-cuda-graph`` reaches the stage as
``OmniEngineArgs.extra["use_thinker_cuda_graph"]`` (unknown kwargs land
in ``extra``). The stage turns it into the fork ``Config``'s
``enforce_eager``: eager by default (graph capture fails on RTX 3050,
ADR-0005), graph when the flag is set.

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


def test_thinker_is_eager_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    thinker, config = _build_stage(monkeypatch, {})

    assert thinker.config.enforce_eager is True


def test_thinker_graph_flag_enables_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    thinker, config = _build_stage(monkeypatch, {"use_thinker_cuda_graph": True})

    assert thinker.config.enforce_eager is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
