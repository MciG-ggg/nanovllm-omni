"""OmniBase: shared constructor and lazy engine setup for Omni / AsyncOmni."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.params import OmniEngineArgs
from ..config.registry import (
    OMNI_PIPELINES,
    DeployConfig,
    PipelineConfig,
    load_deploy_config,
    resolve_pipeline_config,
)

if TYPE_CHECKING:
    from ..engine.executor import PipelineExecutor


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _name_match_candidate(model: str) -> str:
    """basename of the model id/path, lowercased with separators stripped.

    vllm-omni uses this same trick for the path-substring fallback (e.g.
    ``cosyvoice3`` beats ``cosyvoice`` by length). Mirrors their helper so
    existing local-dir inference matches what vllm-omni would do. Dots
    are stripped too so version-suffixed names like ``Mimir-1.6B-Instruct``
    still match against the un-dotted handle ``mimir_1_6b_instruct``.
    """
    name = Path(model.rstrip("/")).name or model
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


def _load_pretrained_config(model: str, trust_remote_code: bool) -> Any | None:
    """Load a HF ``PretrainedConfig``, or return None if transformers is
    unavailable / the config cannot be loaded.

    Shared by L1 and L6 of ``try_infer_model_type`` so the import guard and
    the ``from_pretrained`` swallow only live in one place.
    """
    try:
        from transformers import PretrainedConfig  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return PretrainedConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception:
        return None


def try_infer_model_type(
    model: str,
    trust_remote_code: bool = True,
) -> str | None:
    """Auto-detect the registered ``PipelineConfig.name`` for a model id/path.

    Cascade mirrors ``vllm_omni.config.config_factory.try_infer_model_type``:

    1. ``PretrainedConfig.from_pretrained`` -> ``model_type`` (skipped if
       transformers is unavailable; not in the base install).
    2. Read the directory's ``config.json`` ``model_type`` field.
    3. Fall back to ``config.json["architecture"]`` (singular), used by some
       non-HF pipelines (VoxCPM2-style).
    4. Fall back to ``model_index.json._class_name`` matched against
       ``PipelineConfig.diffusers_class_name`` (none today; placeholder).
    5. Path basename substring match against every registered key, longest
       wins (covers CosyVoice3-style snapshots that ship empty config.json).
    6. ``hf_config.architectures`` intersected with ``PipelineConfig.hf_architectures``;
       gated by the pipeline's ``hf_config_predicate`` when set.

    Returns the inferred model_type string, or ``None`` if no candidate
    matches.
    """
    model_dir = Path(model) if Path(model).is_dir() else None

    # L1: transformers PretrainedConfig (skipped if not installed)
    cfg = _load_pretrained_config(model, trust_remote_code)
    if cfg is not None and getattr(cfg, "model_type", None):
        return cfg.model_type

    # L2 / L3: config.json
    if model_dir is not None:
        data = _read_json(model_dir / "config.json")
        if data is not None:
            for key in ("model_type", "type", "architecture"):
                raw = data.get(key)
                if isinstance(raw, str) and raw:
                    return raw

    # L5: path basename substring match (longest registry key wins)
    candidate = _name_match_candidate(model)
    best: str | None = None
    best_len = 0
    for registered_key in OMNI_PIPELINES:
        norm = registered_key.lower().replace("-", "").replace("_", "")
        if norm and norm in candidate and len(norm) > best_len:
            best = registered_key
            best_len = len(norm)
    if best is not None:
        return best

    # L6: hf_architectures match (needs transformers)
    cfg = _load_pretrained_config(model, trust_remote_code)
    if cfg is not None:
        archs = set(getattr(cfg, "architectures", []) or [])
        if archs:
            for _key, registered in OMNI_PIPELINES.items():
                if isinstance(registered, PipelineConfig):
                    if not registered.hf_architectures:
                        continue
                    if archs.intersection(registered.hf_architectures):
                        predicate = registered.hf_config_predicate
                        if predicate is not None:
                            try:
                                if not predicate(cfg):
                                    continue
                            except Exception:
                                continue
                        return registered.name

    return None


def _pipeline_from_local_dir(model_dir: Path) -> PipelineConfig | None:
    """Local-dir fallback: ask ``try_infer_model_type`` and MiniMind-O default."""
    model_type = try_infer_model_type(str(model_dir))
    if model_type:
        found = resolve_pipeline_config(model_type)
        if found is not None:
            return found
    return resolve_pipeline_config("minimind_o")


class OmniBase:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)
        self._bundle: Any = None
        self._pipeline: PipelineConfig | None = None
        self._deploy: DeployConfig | None = None
        self._executor: PipelineExecutor | None = None

    def _resolve_pipeline(self) -> PipelineConfig:
        if self._pipeline is None:
            extra = self.engine_args.extra or {}
            explicit = extra.get("pipeline")
            pipeline = resolve_pipeline_config(explicit) if explicit else None
            if pipeline is None:
                pipeline = resolve_pipeline_config(self.model)
            if pipeline is None and Path(self.model).is_dir():
                pipeline = _pipeline_from_local_dir(Path(self.model))
            if pipeline is None:
                raise ValueError(
                    f"No pipeline registered for model {self.model!r}. "
                    f"Known models: see nanovllm_omni.config.OMNI_PIPELINES."
                )
            self._pipeline = pipeline
        return self._pipeline

    def _compute_final_stage_id(self, modalities: list[str] | None = None) -> int:
        """Pick the terminal stage whose ``final_output_type`` matches.

        Mirrors ``vllm_omni.entrypoints.utils.get_final_stage_id_for_e2e``:
        scan stages in reverse order for the first one with
        ``final_output_type in (modalities or [all stages' types])``. When
        ``modalities`` is None, every terminal stage is eligible and the
        last one wins.
        """
        pipeline = self._resolve_pipeline()
        if modalities is None:
            requested = {
                s.final_output_type
                for s in pipeline.stages
                if s.is_terminal and s.final_output_type
            }
        else:
            requested = set(modalities)
        last = len(pipeline.stages) - 1
        for sid in range(last, -1, -1):
            stage = pipeline.stages[sid]
            if stage.is_terminal and stage.final_output_type in requested:
                return sid
        return last

    def _final_output_type(self, modalities: list[str] | None = None) -> str:
        """Convenience: ``final_output_type`` of the resolved terminal stage.

        Kept for backward compatibility -- new code should call
        ``_compute_final_stage_id`` directly so the caller can route
        multi-modal outputs.
        """
        pipeline = self._resolve_pipeline()
        sid = self._compute_final_stage_id(modalities)
        stage = pipeline.stages[sid]
        return stage.final_output_type or "audio"

    def _resolve_deploy(self) -> DeployConfig:
        if self._deploy is None:
            pipeline = self._resolve_pipeline()
            deploy_path = (
                self.engine_args.extra.get("deploy_config_path") if self.engine_args.extra else None
            )
            if deploy_path is None:
                deploy_path = Path.cwd() / "deploy" / pipeline.default_deploy_config_name
            else:
                deploy_path = Path(deploy_path)
            self._deploy = load_deploy_config(deploy_path)
        return self._deploy

    def _ensure_executor(self) -> PipelineExecutor:
        if self._executor is None:
            from ..engine.executor import PipelineExecutor

            pipeline = self._resolve_pipeline()
            deploy = self._resolve_deploy()
            max_concurrent = int((self.engine_args.extra or {}).get("max_concurrent", 1))
            self._executor = PipelineExecutor(
                pipeline=pipeline,
                deploy=deploy,
                args=self.engine_args,
                max_concurrent=max_concurrent,
            )
        return self._executor

    def _ensure_bundle(self) -> Any:
        """Backward-compat shim for callers that still read the loaded
        MiniMind-O bundle directly. New code should use the runner instead.
        """
        if self._bundle is None:
            from ..models.minimind_omni import load_minimind_omni_bundle

            extra = dict(self.engine_args.extra or {})
            mimi_model_id = extra.pop("mimi_model_id", None) or extra.pop("mimi", None)
            kwargs: dict[str, Any] = {
                "trust_remote_code": self.engine_args.trust_remote_code,
                "dtype": self.engine_args.dtype,
            }
            if mimi_model_id:
                kwargs["mimi_model_id"] = mimi_model_id
            self._bundle = load_minimind_omni_bundle(
                model_id=self.model,
                device=self.engine_args.device,
                **kwargs,
            )
        return self._bundle
