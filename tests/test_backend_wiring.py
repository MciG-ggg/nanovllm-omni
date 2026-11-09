"""Backend registry and stage-wiring contracts (no GPU / triton needed).

The registry must stay importable without either array framework: the
torch backend pulls in the fork's triton kernels and the MLX backend
pulls in ``mlx``, so both load inside their own module's methods.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from nanovllm_omni.backend import StageRunner as StageRunnerProtocol
from nanovllm_omni.backend import available_backends, get_backend

REPO = Path(__file__).resolve().parents[1]


def test_registry_lists_both_backends() -> None:
    assert {"torch", "mlx"} <= set(available_backends())


def test_torch_capabilities() -> None:
    caps = get_backend("torch").capabilities

    assert caps.varlen is True
    assert caps.graph is True


def test_mlx_capabilities() -> None:
    caps = get_backend("mlx").capabilities

    assert caps.varlen is False
    assert caps.graph is True
    assert caps.max_batch == 1


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("does-not-exist")


def test_default_backend_is_torch() -> None:
    assert get_backend().name == "torch"


def test_registry_import_does_not_load_a_framework() -> None:
    code = (
        "import sys\n"
        "import nanovllm_omni.backend\n"
        "print('torch' in sys.modules, 'mlx' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )

    assert result.stdout.split() == ["False", "False"]


def test_engine_stage_runner_satisfies_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model_runner = mock.MagicMock(name="ModelRunner")
    monkeypatch.setitem(
        sys.modules,
        "nanovllm.engine.model_runner",
        mock.MagicMock(ModelRunner=fake_model_runner),
    )
    monkeypatch.setitem(
        sys.modules,
        "nanovllm.layers.sampler",
        mock.MagicMock(Sampler=mock.MagicMock()),
    )

    from nanovllm_omni.engine.stage_runner import StageRunner

    runner = StageRunner(model_class=object, config=object())

    assert isinstance(runner, StageRunnerProtocol)


@pytest.mark.parametrize(
    "relative_path",
    [
        "nanovllm_omni/models/minimind_omni/stage_runner.py",
        "nanovllm_omni/models/smolvlm/stage.py",
    ],
)
def test_stage_code_does_not_touch_the_fork_runner(relative_path: str) -> None:
    source = (REPO / relative_path).read_text(encoding="utf-8")

    assert ".model_runner" not in source


def test_decode_minimind_passes_the_vendor_sampler_per_call() -> None:
    source = (REPO / "nanovllm_omni/models/minimind_omni/stage_runner.py").read_text(
        encoding="utf-8"
    )

    assert "sampler=_sampler_with_rp" in source
    assert ".sampler =" not in source
