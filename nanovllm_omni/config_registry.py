"""Pipeline and deploy configuration registry.

The data-driven pipeline abstraction (PipelineConfig + StageConfig) and the
deploy / runtime split (DeployConfig + DeployStageConfig) live here. This
module is the registry-of-record: it exposes ``OMNI_PIPELINES``,
``register_pipeline``, ``resolve_pipeline_config``, ``load_deploy_config``,
and ``merge_pipeline_deploy``.

Design basis: 10-round grill session in /docs/design-grill.md. Field set and
alignment boundary are documented in /docs/aligned_interfaces.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StageConfig:
    """Frozen description of a single pipeline stage.

    ``kind`` is a string (``"ar"``, ``"codec"``, ``"diffusion"``, ...) rather
    than an enum so future execution classes can be introduced without code
    changes. ``factory`` is a direct Python callable; we do not use
    dotted-path reflective resolution. ``process_input`` is the bridge hook
    that converts the previous stage's output into this stage's input.
    """

    stage_id: int
    name: str
    kind: str
    factory: Callable[..., Any]
    process_input: Callable[..., Any] | None = None
    input_sources: tuple[int, ...] = ()
    is_terminal: bool = False
    final_output_type: str | None = None
    model_subdir: str | None = None
    tokenizer_subdir: str | None = None
    diffusers_class_name: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen description of a pipeline (a model family)."""

    name: str
    stages: tuple[StageConfig, ...]
    default_deploy_config_name: str
    registration_handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeployStageConfig:
    """Per-stage runtime / sampling defaults loaded from a deploy YAML."""

    name: str
    default_sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeployConfig:
    """Runtime knob bundle loaded from one deploy YAML."""

    stages: tuple[DeployStageConfig, ...] = ()


OMNI_PIPELINES: dict[str, PipelineConfig] = {}


def register_pipeline(
    pipeline: PipelineConfig,
    model_type: str | None = None,
) -> None:
    """Register a PipelineConfig as the canonical model and (optionally) under
    additional handles (e.g. a HF repo id)."""
    if not isinstance(pipeline, PipelineConfig):
        raise TypeError(f"register_pipeline expected PipelineConfig, got {type(pipeline).__name__}")
    OMNI_PIPELINES[pipeline.name] = pipeline
    if model_type is not None:
        OMNI_PIPELINES[model_type] = pipeline
    for handle in pipeline.registration_handles:
        OMNI_PIPELINES[handle] = pipeline


def resolve_pipeline_config(name: str) -> PipelineConfig | None:
    """Look up a registered pipeline by name or alias handle."""
    return OMNI_PIPELINES.get(name)


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Parse a deploy YAML into a DeployConfig."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return DeployConfig(
        tuple(
            DeployStageConfig(
                name=str(s.get("name", "")),
                default_sampling_params=dict(s.get("default_sampling_params", {}) or {}),
            )
            for s in data.get("stages", [])
        )
    )


def merge_pipeline_deploy(
    pipeline_cfg: PipelineConfig,
    deploy_cfg: DeployConfig,
) -> tuple[tuple[StageConfig, dict[str, Any]], ...]:
    """Combine a PipelineConfig with a DeployConfig into per-stage (config, defaults).

    The merge is name-based: each StageConfig's ``name`` is matched against
    DeployStageConfig entries. Stages with no deploy entry receive an empty
    sampling-params dict. Returns a tuple aligned with ``pipeline_cfg.stages``,
    in stage order.
    """
    deploy_by_name = {s.name: s for s in deploy_cfg.stages}
    return tuple(
        (
            stage,
            dict(
                deploy_by_name.get(
                    stage.name, DeployStageConfig(stage.name)
                ).default_sampling_params
            ),
        )
        for stage in pipeline_cfg.stages
    )


def _load_builtin_pipelines() -> None:
    """Import model-family pipeline modules for side-effect registration.

    Kept at module bottom so ``register_pipeline`` is already defined when
    the imported modules call it. New families add one import here.
    """
    from nanovllm_omni.models.minimind_omni import pipeline as _minimind_pipeline  # noqa: F401


_load_builtin_pipelines()
