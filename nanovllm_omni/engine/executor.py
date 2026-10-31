"""PipelineExecutor: async wrapper around PipelineRunner.

Dispatches synchronous runner calls to a thread pool executor so that
multiple HTTP requests or ``AsyncOmni`` consumers can share a single GPU.
``max_concurrent=1`` by default (single GPU).

``run_many`` is the batched entry: it puts every request in flight before
draining, which is what lets one stage see several requests in one step.
Draining one request at a time (``run``) can never batch.

Design basis: 10-round grill session Q3 = (i). Sanity layer for HTTP and
AsyncOmni; the underlying model loading is synchronous.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator  # noqa: F401 (re-exported for callers)
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import DeployConfig, PipelineConfig
from nanovllm_omni.engine.runner import PipelineRunner


class PipelineExecutor:
    """Async wrapper around a single ``PipelineRunner``."""

    def __init__(
        self,
        pipeline: PipelineConfig,
        deploy: DeployConfig,
        args: OmniEngineArgs,
        max_concurrent: int = 1,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._runner = PipelineRunner(pipeline, deploy, args)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="pipeline-runner",
        )
        # The runner keeps mutable in-flight state and is not reentrant, so
        # every entry point takes this lock. ``max_concurrent`` bounds how
        # many callers may queue, not how many run at once.
        self._lock = threading.Lock()

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def run(
        self,
        prompt: str,
        sampling: SamplingParams | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Run one request on the calling thread (``Omni`` path)."""
        with self._lock:
            return self._runner.run(prompt, sampling)

    def run_many(
        self,
        prompts: list[Any],
        samplings: list[SamplingParams | None],
    ) -> list[tuple[Any, dict[str, float]]]:
        """Run several requests together, in submission order.

        Every request goes in flight before the first step, so a stage
        that batches sees them as one batch. If putting one of them in
        flight raises, the ones already queued are retired rather than
        left to be drained and discarded by the next caller.
        """
        with self._lock:
            request_ids: list[str] = []
            try:
                for prompt, sampling in zip(prompts, samplings, strict=True):
                    request_ids.append(self._runner.add_request(prompt, sampling))
            except BaseException:
                self._runner.retire_all()
                raise
            results: dict[str, tuple[Any, dict[str, float]]] = {}
            while self._runner.has_unfinished():
                for finished in self._runner.step():
                    results[finished.request_id] = (finished.payload, finished.stage_ms)
            return [results[request_id] for request_id in request_ids]

    async def submit(
        self,
        prompt: str,
        sampling: SamplingParams | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Submit one request and await ``(payload, stage_ms)``."""
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self.run,
                prompt,
                sampling,
            )

    async def submit_many(
        self,
        prompts: list[Any],
        samplings: list[SamplingParams | None],
    ) -> list[tuple[Any, dict[str, float]]]:
        """Submit several requests together and await all of them."""
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self.run_many,
                prompts,
                samplings,
            )
