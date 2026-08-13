"""Translate a ``PipelineConfig`` into an executable ``Pipeline``.

The runtime stays model-free: this module imports only ``PipelineConfig``,
not any concrete model bundle. The caller passes a ``bundle`` whose
attributes are named after the stages declared in the config (e.g. a
``MinimindBundle`` exposes ``.thinker``, ``.talker``, ``.code2wav``).

``cfg.connectors`` is accepted by the schema but not consumed here —
execution follows ``cfg.stages`` declaration order. Wire connectors
when the orchestrator schedules stages by edge rather than by list
position.
"""

from typing import Any, Protocol

from nanovllm_omni.config import PipelineConfig
from nanovllm_omni.runtime.pipeline import Pipeline
from nanovllm_omni.stage import Stage


class _StageBundle(Protocol):
    """Anything with stage-typed attributes named after stage configs.

    A ``MinimindBundle`` satisfies this implicitly; tests use a tiny
    namespace object. Only the attributes whose names match
    ``PipelineConfig.stages`` are read.
    """


def build_pipeline(cfg: PipelineConfig, *, bundle: Any) -> Pipeline:
    """Resolve each ``StageConfig`` to a ``Stage`` on ``bundle`` and chain them.

    Raises ``AttributeError`` if a stage name from the config has no
    matching attribute on ``bundle`` — fail loud at startup, not at
    first request.
    """
    stages: list[Stage[Any, Any]] = [getattr(bundle, stage.name) for stage in cfg.stages]
    return Pipeline(tuple(stages))
