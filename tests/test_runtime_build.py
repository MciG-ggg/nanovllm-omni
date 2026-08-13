"""Smoke tests for the ``build_pipeline`` translation layer.

These tests do not load real model weights -- a tiny namespace acts as
the bundle. They confirm the contract: stage names from the config
resolve to attributes on the bundle, missing attributes fail loud, and
the resulting ``Pipeline`` exposes the same ordered stage list.
"""

from types import SimpleNamespace

import pytest

from nanovllm_omni.config import PipelineConfig, StageConfig
from nanovllm_omni.runtime import build_pipeline
from nanovllm_omni.runtime.pipeline import Pipeline
from nanovllm_omni.stage import Stage


class _EchoStage(Stage[str, str]):
    """Trivial stage that echoes its name back as the payload."""

    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, payload: str) -> str:
        return f"{payload}:{self.name}"


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        thinker=_EchoStage("thinker"),
        talker=_EchoStage("talker"),
        code2wav=_EchoStage("code2wav"),
    )


def test_build_pipeline_chains_stages_in_config_order() -> None:
    cfg = PipelineConfig(
        stages=[
            StageConfig(name="thinker", kind="ar", model_id="m/th"),
            StageConfig(name="talker", kind="ar", model_id="m/tk"),
            StageConfig(name="code2wav", kind="audio_decode", model_id="m/c2w"),
        ]
    )

    pipeline = build_pipeline(cfg, bundle=_bundle())

    assert isinstance(pipeline, Pipeline)
    assert tuple(s.name for s in pipeline.stages) == ("thinker", "talker", "code2wav")


def test_build_pipeline_propagates_missing_bundle_attribute() -> None:
    cfg = PipelineConfig(stages=[StageConfig(name="ghost", kind="ar", model_id="m/g")])

    with pytest.raises(AttributeError, match="ghost"):
        build_pipeline(cfg, bundle=_bundle())
