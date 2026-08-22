"""OmniBase: shared constructor and lazy engine setup for Omni / AsyncOmni."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config_registry import (
    DeployConfig,
    PipelineConfig,
    load_deploy_config,
    resolve_pipeline_config,
)
from ..engine_args import OmniEngineArgs

if TYPE_CHECKING:
    from ..engine.executor import PipelineExecutor


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
            pipeline = resolve_pipeline_config(self.model)
            # Local MiniMind-O snapshots do not have a registry string alias;
            # treat an existing model directory as the built-in MiniMind-O
            # family while preserving normal registry lookup for HF handles.
            if pipeline is None and Path(self.model).is_dir():
                pipeline = resolve_pipeline_config("minimind_o")
            if pipeline is None:
                raise ValueError(
                    f"No pipeline registered for model {self.model!r}. "
                    f"Known models: see nanovllm_omni.config_registry.OMNI_PIPELINES."
                )
            self._pipeline = pipeline
        return self._pipeline

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
            kwargs: dict[str, Any] = {}
            if mimi_model_id:
                kwargs["mimi_model_id"] = mimi_model_id
            self._bundle = load_minimind_omni_bundle(
                model_id=self.model,
                device=self.engine_args.device,
                **kwargs,
            )
        return self._bundle
