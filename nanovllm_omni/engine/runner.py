"""PipelineRunner: synchronous single-replica pipeline engine.

A request is data, not control flow. ``add_request`` puts it in the queue
of the first stage; ``step`` lets one stage serve the requests it is
holding and pushes whatever finished into the next stage's queue. That is
the shape of ``vllm_omni``'s orchestrator — requests move forward stage by
stage and each stage owns its own queue — with the replica pools and
cross-process transport removed, because a single GPU in one process has
one stage running at a time.

The scheduling policy is round-robin over the stages that hold a request:
a stage that keeps receiving work cannot starve the stages after it. The
reference polls every stage each round and serves whichever is ready; on
one GPU that becomes a rotation.

Stage factories and ``process_input`` hooks are tuple registrations on
``StageConfig``; this runner resolves them once per pipeline on first
``step``. Registered model classes are resolved at the same point and
injected into the stage wrapper without any model-name branch here.

The stage side of the contract lives in ``engine/stage_engine.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import (
    DeployConfig,
    OmniModelRegistry,
    PipelineConfig,
    StageConfig,
    merge_pipeline_deploy,
    resolve_stage_factory,
)
from nanovllm_omni.engine.stage_engine import StageRequest
from nanovllm_omni.outputs import adapt_terminal_output


@dataclass(frozen=True)
class FinishedRequest:
    """One request that reached the terminal stage."""

    request_id: str
    payload: Any
    stage_ms: dict[str, float]


@dataclass
class _InFlight:
    """Mutable per-request state inside the runner.

    The payload itself lives in the queue of whichever stage holds the
    request; only the bookkeeping stays here.
    """

    request_id: str
    prompt: Any
    sampling: SamplingParams | None
    stage_ms: dict[str, float] = field(default_factory=dict)


class PipelineRunner:
    """Synchronous single-replica pipeline engine.

    Built once per pipeline. ``add_request`` / ``step`` are the engine
    loop; ``run`` is the one-shot convenience built on top. Stage
    instances are constructed lazily on the first ``step`` so that
    import-time side effects (model loading) are deferred.

    Not reentrant: ``step`` is the only writer to in-flight state, and
    ``PipelineExecutor`` holds a lock across every call into the runner.
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
        self._deploy_stages = {stage.name: stage for stage in deploy.stages}
        self._stage_instances: list[Any] | None = None
        self._in_flight: dict[str, _InFlight] = {}
        self._next_stage = 0

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
            instances = []
            for stage in self.pipeline.stages:
                factory = resolve_stage_factory(stage.stage_factory)
                kwargs: dict[str, Any] = {}
                if stage.model_architecture is not None:
                    kwargs["model_class"] = OmniModelRegistry.resolve(stage.model_architecture)
                instance = factory(
                    self.deploy,
                    self._stage_args(stage.name),
                    **kwargs,
                )
                instances.append(instance)
            self._stage_instances = instances
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

    def _submit(self, stage_index: int, request: _InFlight, payload: Any) -> None:
        """Run the stage's bridge hook, then queue the request for that stage."""
        stage_cfg, stage_defaults = self._merged[stage_index]
        if stage_cfg.process_input is not None:
            payload = resolve_stage_factory(stage_cfg.process_input)(payload, request.prompt)
        self._ensure_stages()[stage_index].add_request(
            StageRequest(
                request_id=request.request_id,
                payload=payload,
                sampling=self._stage_sampling(stage_defaults, request.sampling),
            )
        )

    def retire_all(self) -> None:
        """Drop every in-flight request.

        A step that raises must not leave the engine stuck: ``run`` drains
        until nothing is in flight, so a request no stage holds would spin
        forever. Today only the failing caller's own batch is in flight,
        because the executor serialises callers.

        Clears ``_in_flight`` first and reads the already-built stages, so
        the handler cannot fail on its own way out.
        """
        self._in_flight.clear()
        for stage in self._stage_instances or ():
            stage.drop_pending()

    def add_request(
        self,
        prompt: Any,
        sampling: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        """Put one request in flight; returns its id.

        ``prompt`` is the initial payload. ``sampling`` is the per-request
        override; per-stage deploy defaults are merged underneath.
        """
        request_id = request_id or uuid.uuid4().hex
        if request_id in self._in_flight:
            raise ValueError(f"request {request_id!r} is already in flight")
        request = _InFlight(request_id=request_id, prompt=prompt, sampling=sampling)
        self._in_flight[request_id] = request
        try:
            self._submit(0, request, prompt)
        except BaseException:
            # Loading the first stage can fail (missing weights, OOM); an
            # in-flight request no stage holds would hang the next ``run``.
            self._in_flight.pop(request_id, None)
            raise
        return request_id

    def has_unfinished(self) -> bool:
        """Whether any request is still in flight."""
        return bool(self._in_flight)

    def step(self) -> list[FinishedRequest]:
        """Let one stage serve the requests it is holding.

        Stages are served round-robin, starting where the last step left
        off, so a stage that keeps receiving work cannot starve the stages
        after it. Returns the requests that reached the terminal stage
        during this step (usually none).
        """
        if not self._in_flight:
            return []
        stages = self._ensure_stages()
        count = len(self._merged)
        for offset in range(count):
            index = (self._next_stage + offset) % count
            stage = stages[index]
            if not stage.has_pending():
                continue
            self._next_stage = (index + 1) % count
            stage_cfg = self._merged[index][0]
            t0 = time.perf_counter()
            try:
                outputs = stage.step()
                # Every request the stage served waited for the whole batch.
                batch_ms = (time.perf_counter() - t0) * 1000.0
                # ``_advance`` is inside the guard as well: its
                # ``process_input`` hook and the next stage's construction are
                # model code and can raise mid-route.
                return self._advance(index, stage_cfg, outputs, batch_ms)
            except BaseException:
                self.retire_all()
                raise
        return []

    def _advance(
        self,
        stage_index: int,
        stage_cfg: StageConfig,
        outputs: list[tuple[str, Any]],
        batch_ms: float,
    ) -> list[FinishedRequest]:
        """Record a stage's outputs and push each request one stage forward."""
        finished = []
        for request_id, payload in outputs:
            request = self._in_flight[request_id]
            if stage_cfg.is_terminal and stage_cfg.final_output_type is not None:
                payload = adapt_terminal_output(payload, stage_cfg.final_output_type)
            request.stage_ms[stage_cfg.name] = batch_ms
            if stage_index + 1 == len(self._merged):
                del self._in_flight[request_id]
                finished.append(
                    FinishedRequest(
                        request_id=request_id,
                        payload=payload,
                        stage_ms=request.stage_ms,
                    )
                )
            else:
                self._submit(stage_index + 1, request, payload)
        return finished

    def run(
        self, prompt: Any, sampling: SamplingParams | None = None
    ) -> tuple[Any, dict[str, float]]:
        """Add one request and drain it; returns ``(payload, stage_ms)``.

        The one-shot form every single-request caller uses (``Omni``,
        bench). Batched callers use ``add_request`` + ``step``.
        """
        request_id = self.add_request(prompt, sampling)
        while self.has_unfinished():
            for finished in self.step():
                if finished.request_id == request_id:
                    return finished.payload, finished.stage_ms
        raise RuntimeError(f"request {request_id!r} never finished")

    def run_payload(self, prompt: Any, sampling: SamplingParams | None = None) -> Any:
        """Run one request, return only the payload (executor/entrypoint seam)."""
        payload, _ = self.run(prompt, sampling)
        return payload
