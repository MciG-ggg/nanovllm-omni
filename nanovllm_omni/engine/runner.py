"""PipelineRunner: synchronous single-replica pipeline runner.

Runs one request through ``PipelineConfig.stages`` sequentially. Knows
nothing about concurrency, async, HTTP, or multi-replica routing. The
companion class ``PipelineExecutor`` (in ``engine/executor.py``) wraps
this runner for async and multi-request use.

Design basis: 10-round grill session. vllm-omni's StagePool and
Orchestrator are designed for multi-replica routing and cross-stage
request lifecycle management; neither is needed for nanovllm-omni's
single-process, single-GPU scope. See /docs/aligned_interfaces.md.

Phase 2 (TK-016): stage factories and ``process_input`` hooks are
dotted-path strings on ``StageConfig``; this runner resolves them via
``resolve_stage_factory`` once per pipeline on first ``run``.
"""

from __future__ import annotations

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
        self.args = args
        self._merged = merge_pipeline_deploy(pipeline, deploy)
        self._stage_instances: list[Any] | None = None

    def _ensure_stages(self) -> list[Any]:
        if self._stage_instances is None:
            self._stage_instances = [
                resolve_stage_factory(stage.factory)(self.deploy, self.args)
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
            kwargs: dict[str, Any] = {}
            if "temperature" in per_stage_defaults:
                kwargs["temperature"] = float(per_stage_defaults["temperature"])
            if "top_p" in per_stage_defaults:
                kwargs["top_p"] = float(per_stage_defaults["top_p"])
            if "top_k" in per_stage_defaults:
                kwargs["top_k"] = int(per_stage_defaults["top_k"])
            if "max_tokens" in per_stage_defaults:
                kwargs["max_tokens"] = int(per_stage_defaults["max_tokens"])
            if "seed" in per_stage_defaults:
                kwargs["seed"] = per_stage_defaults["seed"]
            if "n" in per_stage_defaults:
                kwargs["n"] = int(per_stage_defaults["n"])
            if "stop" in per_stage_defaults:
                stop = per_stage_defaults["stop"]
                kwargs["stop"] = list(stop) if stop is not None else None
            return SamplingParams(extra=extras, **kwargs)

        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=list(request.stop) if request.stop else None,
            seed=request.seed,
            n=request.n,
            extra={**extras, **(request.extra or {})},
        )

    def run(self, prompt: str, sampling: SamplingParams | None = None) -> Any:
        """Run one request through the pipeline.

        ``prompt`` is the initial payload. ``sampling`` is the per-request
        override; per-stage deploy defaults are merged underneath.
        """
        stages = self._ensure_stages()
        payload: Any = prompt
        for (stage_cfg, stage_defaults), instance in zip(self._merged, stages, strict=True):
            if stage_cfg.process_input is not None:
                payload = resolve_stage_factory(stage_cfg.process_input)(payload, prompt)
            stage_sampling = self._stage_sampling(stage_defaults, sampling)
            payload = instance(payload, stage_sampling)
        return payload
