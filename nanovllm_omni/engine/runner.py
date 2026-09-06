"""PipelineRunner: synchronous single-replica pipeline runner.

Runs one request through ``PipelineConfig.stages`` sequentially. Knows
nothing about concurrency, async, HTTP, or multi-replica routing. The
companion class ``PipelineExecutor`` (in ``engine/executor.py``) wraps
this runner for async and multi-request use.

Design basis: 10-round grill session. The reference's StagePool and
Orchestrator are designed for multi-replica routing and cross-stage
request lifecycle management; neither is needed for this project's
single-process, single-GPU scope.

Phase 2 (TK-016): stage factories and ``process_input`` hooks are
dotted-path strings on ``StageConfig``; this runner resolves them via
``resolve_stage_factory`` once per pipeline on first ``run``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    merge_pipeline_deploy,
    resolve_stage_factory,
)


class PipelineRunner:
    """Synchronous single-replica pipeline runner.

    A runner is built once per pipeline, then ``run`` is called once per
    request. Stage instances are constructed lazily on the first ``run``
    call so that import-time side effects (model loading) are deferred.
    """

    def __init__(
        self,
        pipeline: PipelineConfig,
        deploy: DeployConfig,
        args: OmniEngineArgs,
    ) -> None:
        self.pipeline = pipeline
        self.deploy = deploy
        self.deploy.validate_pipeline_kind(pipeline)
        self.args = args
        self._merged = merge_pipeline_deploy(pipeline, deploy)
        self._deploy_stages = {stage.name: stage for stage in deploy.stages}
        self._stage_instances: list[Any] | None = None

    def _stage_args(self, stage_name: str) -> OmniEngineArgs:
        """Overlay stage-local engine knobs without mutating shared args."""
        stage_deploy = self._deploy_stages.get(stage_name)
        if stage_deploy is None:
            return self.args
        args = copy.copy(self.args)
        args.extra = dict(self.args.extra or {})
        for field_name in (
            "max_num_batched_tokens",
            "max_num_seqs",
            "gpu_memory_utilization",
            "enforce_eager",
            "device",
        ):
            value = getattr(stage_deploy, field_name)
            if value is not None:
                setattr(args, field_name, value)
        if stage_deploy.devices is not None:
            args.extra["devices"] = stage_deploy.devices
        return args

    def _ensure_stages(self) -> list[Any]:
        if self._stage_instances is None:
            self._stage_instances = [
                resolve_stage_factory(stage.factory)(
                    self.deploy,
                    self._stage_args(stage.name),
                )
                for stage in self.pipeline.stages
            ]
        return self._stage_instances

    def _stage_sampling(
        self,
        per_stage_defaults: dict[str, Any],
        request: SamplingParams | None,
    ) -> SamplingParams:
        """Build a SamplingParams for one stage by merging the deploy YAML
        defaults with the caller's request override. The caller's fields win
        at the top level; deploy defaults always land in ``extra``. When the
        caller passes ``None``, deploy defaults also populate the top-level
        SamplingParams fields (temperature / top_p / max_tokens / …).
        """
        extras = dict(per_stage_defaults)
        if request is None:
            # Deploy YAML is untyped; cast strictly-typed fields so
            # SamplingParams' frozen validation accepts them.
            _casts = {
                "temperature": float,
                "top_p": float,
                "top_k": int,
                "max_tokens": int,
                "n": int,
            }
            kwargs: dict[str, Any] = {
                name: cast_(per_stage_defaults[name])
                for name, cast_ in _casts.items()
                if name in per_stage_defaults
            }
            if "seed" in per_stage_defaults:
                # int | None: pass through unchanged so YAML `seed: null` works.
                kwargs["seed"] = per_stage_defaults["seed"]
            if "stop" in per_stage_defaults:
                stop = per_stage_defaults["stop"]
                kwargs["stop"] = list(stop) if stop is not None else None
            return SamplingParams(extra=extras, **kwargs)

        return dataclasses.replace(
            request,
            extra={**extras, **(request.extra or {})},
        )

    def run(self, prompt: str, sampling: SamplingParams | None = None) -> Any:
        """Run one request through the pipeline.

        ``prompt`` is the initial payload. ``sampling`` is the per-request
        override; per-stage deploy defaults are merged underneath.
        """
        stages = self._ensure_stages()
        payload: Any = prompt
        for _, ((stage_cfg, stage_defaults), instance) in enumerate(
            zip(self._merged, stages, strict=True)
        ):
            if stage_cfg.process_input is not None:
                payload = resolve_stage_factory(stage_cfg.process_input)(payload, prompt)
            stage_sampling = self._stage_sampling(stage_defaults, sampling)
            payload = instance(payload, stage_sampling)
        return payload
