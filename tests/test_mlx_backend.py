"""MLX backend contracts that hold without mlx or a model on disk.

``mlx`` / ``mlx-lm`` are optional Mac-track dependencies and are not in the
project venv, so the runner is built against stub modules here. The real
decode loop runs in ``tools/bench_mlx_thinker.py``.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from nanovllm_omni.backend import StageRunner as StageRunnerProtocol
from nanovllm_omni.backend.mlx_backend import MlxBackend, MlxConfig, MlxStageRunner


def _stub_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the two modules ``MlxStageRunner.__init__`` imports."""
    mx = types.ModuleType("mlx.core")
    mx.float16 = "float16"
    mx.float32 = "float32"
    mx.gpu = "gpu"
    mx.set_default_device = mock.MagicMock()
    mx.eval = mock.MagicMock()
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    utils = types.ModuleType("mlx_lm.utils")
    utils.load = mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)


def test_mlx_runner_satisfies_the_stage_runner_protocol(monkeypatch) -> None:
    _stub_mlx(monkeypatch)

    runner = MlxStageRunner(model_class=object, config=MlxConfig(model_path="unused"))

    assert isinstance(runner, StageRunnerProtocol)


def test_mlx_backend_builds_a_protocol_runner(monkeypatch) -> None:
    _stub_mlx(monkeypatch)
    backend = MlxBackend()

    runner = backend.make_stage_runner(object, backend.make_config("unused"))

    assert isinstance(runner, StageRunnerProtocol)
