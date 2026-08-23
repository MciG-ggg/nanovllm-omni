"""AsyncOmni: async-iterator aligned entry point.

Returns an ``AsyncIterator[OmniRequestOutput]`` that submits each prompt
to ``PipelineExecutor.submit`` and awaits the result. The underlying
``PipelineRunner`` is synchronous; the executor wraps it in
``run_in_executor`` so the event loop is not blocked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..engine_args import SamplingParams
from ..outputs import OmniRequestOutput
from .omni import Omni


class AsyncOmni(Omni):
    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> AsyncIterator[OmniRequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        executor = self._ensure_executor()
        for prompt in prompts:
            payload = await executor.submit(prompt, sampling_params)
            yield OmniRequestOutput.from_pipeline(
                payload,
                final_output_type=self._final_output_type(),
            )
