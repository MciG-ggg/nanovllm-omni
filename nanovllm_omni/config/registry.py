"""Pipeline and deploy configuration registry.

The data-driven pipeline abstraction (PipelineConfig + StageConfig) and the
deploy / runtime split (DeployConfig + DeployStageConfig) live here. This
module is the registry-of-record: it exposes ``OMNI_PIPELINES``,
``register_pipeline``, ``resolve_pipeline_config``, ``load_deploy_config``,
``merge_pipeline_deploy``, ``StageExecutionType``, and
``resolve_stage_factory``.

Design basis: 10-round grill session in /docs/design-grill.md. Field set and
alignment boundary are documented in /docs/aligned_interfaces.md.

Phase 2 (TK-016): ``StageConfig.factory`` / ``process_input`` are now
dotted-path strings (``"package.module:attr"``) resolved via
``resolve_stage_factory``; ``kind`` is the :class:`StageExecutionType` enum
mirroring vllm-omni's LLM_AR / LLM_GENERATION / DIFFUSION / CODEC taxonomy.
Per-stage modules are no longer statically imported by ``pipeline.py``; the
pipeline topology file is now fully declarative. See SPEC.md "Module
structure" for the contract.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class StageExecutionType(StrEnum):
    """Pipeline-stage execution taxonomy.

    Mirrors vllm-omni's StageExecutionType (LLM_AR / LLM_GENERATION /
    DIFFUSION / CODEC). ``StrEnum`` so members compare equal to legacy
    string values (``StageExecutionType.LLM_AR == "ar"``) while remaining
    a closed set. Add new members here when a new execution class is
    introduced; callers compare against the member, not the value.
    """

    LLM_AR = "ar"
    LLM_GENERATION = "generation"
    DIFFUSION = "diffusion"
    CODEC = "codec"


def resolve_stage_factory(path: str) -> Callable[..., Any]:
    """Resolve a ``"package.module:attr"`` string to a live Python callable.

    Convention: module path and attribute are separated by a single colon,
    matching Python's entry-point / setuptools convention. The module is
    imported (or fetched from ``sys.modules``) and the attribute looked up
    by ``getattr``.

    Raises:
        ValueError: path is empty, not a string, or not in ``module:attr``
            form.
        ImportError: the module cannot be imported.
        AttributeError: the module imported but the attribute is missing.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"factory path must be a non-empty string, got {path!r}")
    module_name, sep, attr = path.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"factory path {path!r} must be in 'package.module:attr' form")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"factory path {path!r}: cannot import module {module_name!r}") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise AttributeError(
            f"factory path {path!r}: module {module_name!r} has no attribute {attr!r}"
        ) from e


@dataclass(frozen=True)
class StageConfig:
    """Frozen description of a single pipeline stage.

    ``kind`` is a :class:`StageExecutionType` member. ``factory`` and
    ``process_input`` are dotted-path strings (``"package.module:attr"``)
    that :func:`resolve_stage_factory` turns into live callables.
    ``process_input`` is the bridge hook that converts the previous
    stage's output into this stage's input; ``None`` means the engine
    passes the previous output through unchanged.
    """

    stage_id: int
    name: str
    kind: StageExecutionType
    factory: str
    process_input: str | None = None
    input_sources: tuple[int, ...] = ()
    is_terminal: bool = False
    final_output_type: str | None = None
    model_subdir: str | None = None
    tokenizer_subdir: str | None = None
    diffusers_class_name: str | None = None

    def __post_init__(self) -> None:
        # Eagerly validate kind + factory / process_input paths so
        # topology mistakes fail at construction time (when the pipeline
        # is registered) rather than at first request run. The factory
        # resolution also forces per-stage modules to be importable from
        # the registry's vantage point, mirroring vllm-omni's
        # pipeline-registry contract.
        if not isinstance(self.kind, StageExecutionType):
            raise TypeError(
                f"StageConfig.kind must be a StageExecutionType member, "
                f"got {type(self.kind).__name__}: {self.kind!r}"
            )
        if not isinstance(self.factory, str):
            raise TypeError(
                f"StageConfig.factory must be a dotted-path string "
                f"('package.module:attr'), got {type(self.factory).__name__}"
            )
        resolve_stage_factory(self.factory)
        if self.process_input is not None:
            if not isinstance(self.process_input, str):
                raise TypeError(
                    f"StageConfig.process_input must be a dotted-path "
                    f"string or None, got {type(self.process_input).__name__}"
                )
            resolve_stage_factory(self.process_input)


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen description of a pipeline (a model family)."""

    name: str
    stages: tuple[StageConfig, ...]
    default_deploy_config_name: str
    registration_handles: tuple[str, ...] = ()
    # HF architecture aliases: disambiguates siblings that ship the same
    # ``model_type`` (e.g. SmolVLA + a hypothetical next-gen variant). When
    # ``try_infer_model_type`` finds multiple candidates by ``model_type`` or
    # path basename, it intersects ``hf_config.architectures`` against this
    # tuple and returns the first pipeline with a non-empty intersection
    # whose ``hf_config_predicate`` (if any) accepts the loaded config.
    hf_architectures: tuple[str, ...] = ()
    hf_config_predicate: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class DeployStageConfig:
    """Per-stage runtime / sampling defaults loaded from a deploy YAML."""

    name: str
    default_sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeployConfig:
    """Runtime knob bundle loaded from one deploy YAML.

    ``stages`` carries per-stage sampling defaults; engine-level resource
    knobs (continuous-batching width, etc.) are top-level keys in the YAML
    (AGENTS: runtime knobs belong in ``deploy/*.yaml``, not in pipeline code).
    """

    stages: tuple[DeployStageConfig, ...] = ()
    max_batch: int = 2


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
    max_batch = int(data.get("max_batch", 2))
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    return DeployConfig(
        stages=tuple(
            DeployStageConfig(
                name=str(s.get("name", "")),
                default_sampling_params=dict(s.get("default_sampling_params", {}) or {}),
            )
            for s in data.get("stages", [])
        ),
        max_batch=max_batch,
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
    from nanovllm_omni.models.smolvla import pipeline as _smolvla_pipeline  # noqa: F401


_load_builtin_pipelines()
