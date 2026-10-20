"""A local model dir with ``enable_audio_output: false`` resolves to the
thinker-only topology.

``_pipeline_from_local_dir`` falls back to the MiniMind-O default when
the model type is not registered. That fallback must still see the
dir's ``config.json``, otherwise ``enable_audio_output: false`` is
ignored and the caller loads a talker the checkpoint does not have.
"""

from __future__ import annotations

import json
import sys

import pytest

from nanovllm_omni.entrypoints.base import _pipeline_from_local_dir
from nanovllm_omni.models.minimind_omni.pipeline import (
    MINIMIND_OMNI_PIPELINE,
    MINIMIND_OMNI_THINKER_ONLY_PIPELINE,
)


def _write_config(model_dir, enable_audio_output: bool) -> None:
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "minimind", "enable_audio_output": enable_audio_output}),
        encoding="utf-8",
    )


def test_local_dir_without_audio_output_resolves_thinker_only(tmp_path) -> None:
    _write_config(tmp_path, enable_audio_output=False)

    pipeline = _pipeline_from_local_dir(tmp_path)

    assert pipeline is MINIMIND_OMNI_THINKER_ONLY_PIPELINE


def test_local_dir_with_audio_output_resolves_full(tmp_path) -> None:
    _write_config(tmp_path, enable_audio_output=True)

    pipeline = _pipeline_from_local_dir(tmp_path)

    assert pipeline is MINIMIND_OMNI_PIPELINE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
