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

import dataclasses
from typing import Any

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    merge_pipeline_deploy,
    resolve_stage_factory,
)
from nanovllm_omni.engine.load_balancer import RoundRobinBalancer


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
        # TK-007 per-stage replica routing. Single-device default is 1 replica
        # per stage, so ``select`` always returns 0 and execution is unchanged;
        # the route is recorded on ``last_routes`` for the per-request trace.
        self._balancer = RoundRobinBalancer()
        self.last_routes: list[tuple[int, int]] = []

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
        override; per-stage deploy defaults are merged underneath. The per-
        stage replica route is recorded on ``last_routes`` (TK-007).
        """
        stages = self._ensure_stages()
        routes: list[tuple[int, int]] = []
        payload: Any = prompt
        for stage_id, ((stage_cfg, stage_defaults), instance) in enumerate(
            zip(self._merged, stages, strict=True)
        ):
            routes.append((stage_id, self._balancer.select(stage_id, stage_cfg.num_replicas)))
            if stage_cfg.process_input is not None:
                payload = resolve_stage_factory(stage_cfg.process_input)(payload, prompt)
            stage_sampling = self._stage_sampling(stage_defaults, sampling)
            payload = instance(payload, stage_sampling)
        self.last_routes = routes
        return payload
