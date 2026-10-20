"""AsyncOmni: async-iterator aligned entry point.

Returns an ``AsyncIterator[OmniRequestOutput]`` that submits each prompt
to ``PipelineExecutor.submit`` and awaits the result. The underlying
``PipelineRunner`` is synchronous; the executor wraps it in
``run_in_executor`` so the event loop is not blocked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..config.params import SamplingParams
from ..outputs import OmniRequestOutput
from .omni import Omni, _split_prompt


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
            # Same split as the sync path: a dict prompt carries modal
            # payload (image, ...) that rides along in sampling.extra.
            text, modal = _split_prompt(prompt)
            params = sampling_params
            if modal:
                import dataclasses

                extra = dict(getattr(sampling_params, "extra", None) or {})
                extra.update(modal)
                params = (
                    dataclasses.replace(sampling_params, extra=extra)
                    if sampling_params is not None
                    else SamplingParams(extra=extra)
                )
            payload = await executor.submit(text, params)
            yield OmniRequestOutput.from_pipeline(
                payload,
                final_output_type=self._final_output_type(),
            )
