"""ActionArtifact contract (TK-015). Independent of any VLA backend."""

from __future__ import annotations

import pytest

from nanovllm_omni.outputs import ActionArtifact


def _fake_array(rows: int = 12, cols: int = 7) -> object:
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
    class _A:
        shape = (7,)
        dtype = "float32"

    with pytest.raises(ValueError, match="2-D"):
        ActionArtifact.from_array(_A())


def test_action_artifact_from_array_rejects_3d():
    class _A:
        shape = (4, 12, 7)
        dtype = "float32"

    with pytest.raises(ValueError, match="2-D"):
        ActionArtifact.from_array(_A())


def test_action_artifact_post_init_validates_shape_mismatch():
    arr = _fake_array(rows=12, cols=7)
    with pytest.raises(ValueError, match="action_dim"):
        ActionArtifact(array=arr, action_dim=99, chunk_size=12, dtype="float32")
    with pytest.raises(ValueError, match="chunk_size"):
        ActionArtifact(array=arr, action_dim=7, chunk_size=99, dtype="float32")
