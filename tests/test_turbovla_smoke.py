"""Smoke tests for TurboVLA wrapper (TK-011-rev Step 1).

These tests do NOT require the upstream ``turbovla`` package, real
checkpoints, or any GPU. They cover:

- ``ActionArtifact.from_array`` shape/dtype contract
- ``ActionArtifact.__post_init__`` rejects wrong shapes
- ``TurboVLAConfig`` dataclass construction
- ``TurboVLAOmni`` raises a clear ``ImportError`` when turbovla is absent
- ``TurboVLAOmni`` raises a clear ``FileNotFoundError`` when ckpt is missing
- ``examples/turbovla.py`` parses args and produces the documented
  ``ActionArtifact`` shape on a stubbed predict()

A real end-to-end smoke (with model loaded) lives in the
``@pytest.mark.slow`` lane and is run by CI nightly; it is intentionally
NOT exercised here because it requires the heavy turbovla + DINOv3 + BERT
stack on a CUDA host.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanovllm_omni.models.turbovla import TurboVLAConfig
from nanovllm_omni.outputs import ActionArtifact

# ---------------------------------------------------------------------------
# ActionArtifact contract (TK-015-rev)
# ---------------------------------------------------------------------------


def _fake_array(rows: int = 12, cols: int = 7) -> object:
    """Tiny class mimicking np.ndarray's shape/dtype attributes."""

    class _A:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    return _A((rows, cols), "float32")


def test_action_artifact_from_array_populates_dims():
    arr = _fake_array(rows=12, cols=7)
    a = ActionArtifact.from_array(arr)
    assert a.action_dim == 7
    assert a.chunk_size == 12
    assert a.dtype == "float32"


def test_action_artifact_from_array_rejects_1d():
    arr = _fake_array()._A((7,), "float32")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="2-D"):
        ActionArtifact.from_array(arr)


def test_action_artifact_from_array_rejects_3d():
    arr = _fake_array()._A((4, 12, 7), "float32")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="2-D"):
        ActionArtifact.from_array(arr)


def test_action_artifact_post_init_validates_shape_mismatch():
    arr = _fake_array(rows=12, cols=7)
    with pytest.raises(ValueError, match="action_dim"):
        ActionArtifact(array=arr, action_dim=99, chunk_size=12, dtype="float32")
    with pytest.raises(ValueError, match="chunk_size"):
        ActionArtifact(array=arr, action_dim=7, chunk_size=99, dtype="float32")


# ---------------------------------------------------------------------------
# TurboVLAConfig dataclass
# ---------------------------------------------------------------------------


def test_turbovla_config_construction():
    cfg = TurboVLAConfig(
        ckpt_path="/tmp/ckpt.pth",
        dinov3_path="/tmp/dinov3",
        bert_path="/tmp/bert",
    )
    assert cfg.ckpt_path == "/tmp/ckpt.pth"
    assert cfg.precision == "bf16"
    assert cfg.allow_hf_download is False


# ---------------------------------------------------------------------------
# TurboVLAOmni construction failures (no real model loading)
# ---------------------------------------------------------------------------


def test_turbovla_omni_raises_helpful_import_error_when_package_missing(monkeypatch):
    # Hide the upstream turbovla import.
    monkeypatch.setitem(sys.modules, "turbovla", None)
    monkeypatch.setitem(sys.modules, "turbovla.evaluation", None)
    monkeypatch.setitem(sys.modules, "turbovla.evaluation.policy", None)
    monkeypatch.delitem(sys.modules, "nanovllm_omni.models.turbovla.wrapper", raising=False)

    # Re-import the wrapper so the import dance runs fresh.
    if "nanovllm_omni.models.turbovla.wrapper" in sys.modules:
        del sys.modules["nanovllm_omni.models.turbovla.wrapper"]
    from nanovllm_omni.models.turbovla import wrapper  # noqa: F401

    cfg = TurboVLAConfig(ckpt_path="/x.pth", dinov3_path="/dinov3", bert_path="/bert")
    with pytest.raises(ImportError, match="turbovla package not installed"):
        from nanovllm_omni.models.turbovla import TurboVLAOmni

        TurboVLAOmni(cfg)


def test_turbovla_omni_raises_filenotfound_when_ckpt_missing(monkeypatch, tmp_path: Path):
    # Stub the upstream package so import succeeds.
    fake_policy_module = MagicMock()
    monkeypatch.setitem(sys.modules, "turbovla", MagicMock())
    monkeypatch.setitem(sys.modules, "turbovla.evaluation", MagicMock())
    monkeypatch.setitem(sys.modules, "turbovla.evaluation.policy", fake_policy_module)

    # Force a fresh import of the wrapper so the patched turbovla is used.
    if "nanovllm_omni.models.turbovla.wrapper" in sys.modules:
        del sys.modules["nanovllm_omni.models.turbovla.wrapper"]

    cfg = TurboVLAConfig(
        ckpt_path=str(tmp_path / "missing.pth"),
        dinov3_path=str(tmp_path / "dinov3"),
        bert_path=str(tmp_path / "bert"),
    )
    (tmp_path / "dinov3").mkdir()
    (tmp_path / "bert").mkdir()

    from nanovllm_omni.models.turbovla import TurboVLAOmni

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        TurboVLAOmni(cfg)


# ---------------------------------------------------------------------------
# examples/turbovla.py end-to-end on stubbed predict
# ---------------------------------------------------------------------------


def test_examples_turbovla_prints_action_artifact(monkeypatch, tmp_path: Path, capsys):
    """Run examples/turbovla.py with a stubbed TurboVLAOmni that returns a
    fixed shape; verify it prints an ActionArtifact summary and exits 0.
    """
    import numpy as np

    # Lay down the three "files" the script requires so path validation passes.
    ckpt = tmp_path / "libero_object.pth"
    ckpt.touch()
    dinov3 = tmp_path / "dinov3"
    dinov3.mkdir()
    bert = tmp_path / "bert"
    bert.mkdir()

    # Build the stubbed model that predict() returns an ActionArtifact directly.
    artifact = ActionArtifact.from_array(np.zeros((12, 7), dtype=np.float32))
    payload_holder: list = []

    class _Stub:
        chunk_size = 12
        action_dim = 7
        precision = "bf16"

        def predict(self, primary_image, wrist_image, instruction, state, execute_steps):
            from nanovllm_omni.outputs import MultimodalPayload, OmniRequestOutput

            payload_holder.append(
                OmniRequestOutput(
                    multimodal_output=MultimodalPayload.from_dict({"action": artifact}),
                    metrics={"chunk_size": 12.0, "action_dim": 7.0},
                )
            )
            return payload_holder[-1]

    # Patch the import path used inside examples/turbovla.py.
    stub_module = SimpleNamespace(
        TurboVLAConfig=TurboVLAConfig, TurboVLAOmni=lambda *a, **k: _Stub()
    )
    monkeypatch.setitem(sys.modules, "nanovllm_omni.models.turbovla", stub_module)

    # Invoke the example as a subprocess would: import-and-call main().
    from pathlib import Path as _Path

    examples_path = _Path(__file__).resolve().parents[1] / "examples" / "turbovla.py"
    script_globals = {"__file__": str(examples_path), "__name__": "__not_main__"}
    exec(compile(examples_path.read_text(), str(examples_path), "exec"), script_globals)

    argv = [
        "turbovla.py",
        "--ckpt",
        str(ckpt),
        "--dinov3",
        str(dinov3),
        "--bert",
        str(bert),
        "--seed",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = script_globals["main"]()

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK: action shape=(12, 7)" in out
    assert "metrics:" in out
    assert payload_holder, "predict() was not called"
