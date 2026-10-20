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
        *,
        sampling_params_list: SamplingParams | list[SamplingParams] | None = None,
    ) -> AsyncIterator[OmniRequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is not None and sampling_params_list is not None:
            raise ValueError("pass sampling_params or sampling_params_list, not both")
        if sampling_params_list is None:
            per_prompt = [sampling_params] * len(prompts)
        elif isinstance(sampling_params_list, SamplingParams):
            per_prompt = [sampling_params_list] * len(prompts)
        else:
            per_prompt = list(sampling_params_list)
        executor = self._ensure_executor()
        for idx, prompt in enumerate(prompts):
            params = per_prompt[idx] if idx < len(per_prompt) else None
            # Same split as the sync path: a dict prompt carries modal
            # payload (image, ...) that rides along in sampling.extra.
            text, modal = _split_prompt(prompt)
            if modal:
                import dataclasses

                extra = dict(getattr(params, "extra", None) or {})
                extra.update(modal)
                params = (
                    dataclasses.replace(params, extra=extra)
                    if params is not None
                    else SamplingParams(extra=extra)
                )
            payload = await executor.submit(text, params)
            yield OmniRequestOutput.from_pipeline(
                payload,
                final_output_type=self._final_output_type(),
            )
