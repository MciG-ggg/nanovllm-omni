"""PipelineExecutor: async wrapper around PipelineRunner.

Dispatches synchronous ``PipelineRunner.run`` calls to a thread pool
executor so that multiple HTTP requests or ``AsyncOmni`` consumers can
share a single GPU. ``max_concurrent=1`` by default (single GPU).

Design basis: 10-round grill session Q3 = (i). Sanity layer for HTTP and
AsyncOmni; the underlying model loading is synchronous.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
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

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    async def submit(
        self,
        prompt: str,
        sampling: SamplingParams | None = None,
    ) -> Any:
        """Submit one request and await the result."""
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._runner.run,
                prompt,
                sampling,
            )

    async def stream(
        self,
        prompts_with_sampling: AsyncIterator[tuple[str, SamplingParams]],
    ) -> AsyncIterator[Any]:
        """Submit a stream of (prompt, sampling) pairs and yield results in
        submission order. Each call to ``submit`` is awaited sequentially;
        concurrent submission is the caller's responsibility (yield pairs in
        parallel to interleave).
        """
        async for prompt, sampling in prompts_with_sampling:
            yield await self.submit(prompt, sampling)
