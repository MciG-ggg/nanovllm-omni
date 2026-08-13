"""Typed YAML configuration for stage pipelines.

Pipeline topology (`PipelineConfig`) and deployment/lifecycle settings
(`DeployConfig`) are deliberately decoupled so that the same topology
can be re-deployed across different runtime profiles (CPU smoke tests,
multi-GPU serving, eager/lazy loading, etc.) without editing the file.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Accepted stage kinds. An ordered pipeline may contain multiple AR stages:
# MiniMind-Omni's Thinker and Talker are separate autoregressive boundaries.
_VALID_KINDS = frozenset({"ar", "diffusion", "action", "audio_decode"})


@dataclass
class ConnectorSpec:
    """Directed edge between two stages describing how data flows.

    Attributes:
        source: Name of the producing stage (must match a stage in the
            enclosing ``PipelineConfig``).
        target: Name of the consuming stage (same constraint as ``source``).
        payload_type: Hint for the shape of the data carried by the
            edge (``"audio"``, ``"tokens"``, ``"latents"``, ...).
            Defaults to ``"auto"`` to let the runtime infer.
    """

    source: str
    target: str
    payload_type: str = "auto"

    def __post_init__(self) -> None:
        for field_name in ("source", "target", "payload_type"):
            if (
                not isinstance(getattr(self, field_name), str)
                or not getattr(self, field_name).strip()
            ):
                raise ValueError(f"connector {field_name} must be a non-empty string")


@dataclass
class StageConfig:
    """A single stage in a pipeline.

    Attributes:
        name: Unique identifier referenced by connectors and by runtime
            logging. Used as the lookup key for stage-level state.
        kind: Stage category; one of ``"ar"``, ``"diffusion"``,
            ``"action"``, ``"audio_decode"``. Determines which runtime
            adapter loads and serves the stage.
        model_id: Identifier the runtime uses to resolve model weights
            (a Hub repo id, a local path, or a registry key — exact
            semantics depend on the adapter for ``kind``).
        model_kwargs: Extra keyword arguments forwarded verbatim to the
            model loader (precision, quantization flags,
            ``trust_remote_code``, etc.). Defaults to an empty dict.
    """

    name: str
    kind: str
    model_id: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must be a non-empty string")
        if not isinstance(self.kind, str) or self.kind not in _VALID_KINDS:
            raise ValueError(
                f"invalid stage kind {self.kind!r}; expected one of {sorted(_VALID_KINDS)}"
            )
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError(f"stage {self.name!r} model_id must be a non-empty string")
        if not isinstance(self.model_kwargs, dict):
            raise ValueError(f"stage {self.name!r} model_kwargs must be a mapping")


@dataclass
class PipelineConfig:
    """Pipeline topology: the stages and the connectors that wire them.

    This is the *what* — the graph the runtime will execute. Deployment
    choices live separately on ``DeployConfig`` so the same topology can
    be retargeted at different runtimes without edits.

    Attributes:
        stages: Stages in declaration order. Names must be unique across
            the list. Multiple ``ar`` stages are allowed because each
            logical stage owns its own autoregressive state.
        connectors: Directed edges between stages; each endpoint must
            reference a name in ``stages``. Optional — a pipeline with
            no connectors runs its stages independently.

    Validation runs in ``__post_init__`` so downstream code can assume a
    well-formed topology without re-checking.
    """

    stages: list[StageConfig]
    # ponytail: not yet consumed by runtime -- execution follows stages order.
    # Remove if no consumer by the next milestone, or wire when the orchestrator
    # schedules by edge rather than by list position.
    connectors: list[ConnectorSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        known_names = set(names)
        for connector in self.connectors:
            if connector.source not in known_names or connector.target not in known_names:
                raise ValueError(
                    f"connector references unknown stage: {connector.source!r} -> {connector.target!r}"
                )


@dataclass
class DeployConfig:
    """Runtime-lifecycle settings, decoupled from pipeline topology.

    The same ``PipelineConfig`` can be deployed under different
    ``DeployConfig`` profiles — CPU smoke tests, multi-GPU serving,
    eager vs lazy loading — without editing the topology itself.

    Attributes:
        device: Device spec for the runtime, e.g. ``"cuda"``, ``"cpu"``,
            ``"cuda:0"``, ``"mps"``. Free-form string; interpretation
            is up to the runtime adapter.
        lazy_load: When ``True`` (default), models are loaded on first
            use rather than at process start — trades startup latency
            for lower peak memory. Set ``True`` when working memory is
            tighter than startup time; ``False`` for predictable
            latency on the first request.
        max_active_stages: Upper bound on stages resident in memory
            concurrently. ``1`` means strictly sequential execution
            (each stage is swapped in, used, then evicted); higher
            values keep more stages warm.
    """

    device: str = "cuda"
    # ponytail: not yet consumed by runtime. Wire when the bundle loader
    # supports lazy materialization.
    lazy_load: bool = True
    # ponytail: not yet consumed by runtime. Wire when the orchestrator
    # schedules more than one stage concurrently.
    max_active_stages: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("deploy device must be a non-empty string")
        if not isinstance(self.lazy_load, bool):
            raise ValueError("deploy lazy_load must be a boolean")
        # `bool` is a subclass of `int` in Python, so a YAML `true`/`false`
        # accidentally fed into `max_active_stages` would pass an
        # `isinstance(..., int)` check unless we reject it explicitly.
        if not isinstance(self.max_active_stages, int) or isinstance(self.max_active_stages, bool):
            raise ValueError("deploy max_active_stages must be an integer")
        if self.max_active_stages < 1:
            raise ValueError("deploy max_active_stages must be at least 1")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _first(data: Mapping[str, Any], keys: tuple[str, ...], label: str, default: Any = None) -> Any:
    # Alias-tolerant lookup across synonymous YAML spellings (e.g.
    # `payload_type` vs `payload` vs `type`). Returns the first key that
    # is present in `data`; raises if none are present and no default.
    for key in keys:
        if key in data:
            return data[key]
    if default is None:
        raise ValueError(f"{label} requires one of: {', '.join(keys)}")
    return default


def load_config(path: str | Path) -> tuple[PipelineConfig, DeployConfig]:
    """Parse a YAML config file into a ``(PipelineConfig, DeployConfig)`` pair.

    The YAML may nest pipeline/deploy under their own top-level keys
    (preferred), under an alias, or flat at the top level — see the
    implementation comment below for the exact resolution order.

    Args:
        path: Filesystem path to a UTF-8 YAML file. ``str`` and
            ``pathlib.Path`` are both accepted.

    Returns:
        A ``(PipelineConfig, DeployConfig)`` tuple. All cross-field
        invariants (unique stage names, valid kinds, connector
        endpoints reference defined stages) are validated during
        construction. Multiple ``ar`` stages are allowed.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file is malformed, the YAML is structurally
            invalid, or any validation rule fails. Error messages are
            prefixed with the offending YAML path (e.g.
            ``pipeline.stages[2].kind``) to ease debugging.
    """
    # Pipeline/deploy resolution order (first match wins):
    #   1. `pipeline:` / `deploy:` nested blocks (preferred form)
    #   2. `deployment:` block (alias for `deploy:`)
    #   3. Flat top-level keys (`stages:` / `connectors:` / `device:` / ...)
    # This keeps existing flat-authored configs working while encouraging
    # the nested form going forward.
    raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}, "config")
    pipeline_data = _mapping(raw.get("pipeline", raw), "pipeline")
    deploy_value = raw.get("deploy", raw.get("deployment"))
    if deploy_value is None:
        deploy_value = {
            key: raw[key] for key in ("device", "lazy_load", "max_active_stages") if key in raw
        }
    deploy_data = _mapping(deploy_value, "deploy")

    stage_values = pipeline_data.get("stages", [])
    if not isinstance(stage_values, list) or not stage_values:
        raise ValueError("pipeline.stages must be a non-empty list")
    stages = []
    for index, value in enumerate(stage_values):
        item = _mapping(value, f"pipeline.stages[{index}]")
        label = f"pipeline.stages[{index}]"
        kwargs = item.get("model_kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise ValueError(f"{label}.model_kwargs must be a mapping")
        stages.append(
            StageConfig(
                name=_required_string(item, "name", label),
                kind=_required_string(item, "kind", label),
                model_id=_required_string(item, "model_id", label),
                model_kwargs=dict(kwargs),
            )
        )

    connector_values = pipeline_data.get("connectors", [])
    if not isinstance(connector_values, list):
        raise ValueError("pipeline.connectors must be a list")
    connectors = []
    for index, value in enumerate(connector_values):
        item = _mapping(value, f"pipeline.connectors[{index}]")
        label = f"pipeline.connectors[{index}]"
        connectors.append(
            ConnectorSpec(
                source=_first(item, ("source", "from", "from_stage", "stage_from"), label),
                target=_first(item, ("target", "to", "to_stage", "stage_to"), label),
                payload_type=_first(
                    item, ("payload_type", "payload", "type", "kind"), label, default="auto"
                ),
            )
        )

    return (
        PipelineConfig(stages=stages, connectors=connectors),
        DeployConfig(
            device=deploy_data.get("device", "cuda"),
            lazy_load=deploy_data.get("lazy_load", True),
            max_active_stages=deploy_data.get("max_active_stages", 1),
        ),
    )
