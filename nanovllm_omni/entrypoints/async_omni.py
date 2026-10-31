"""AsyncOmni: async-iterator aligned entry point.

Returns an ``AsyncIterator[OmniRequestOutput]`` that submits each prompt
to ``PipelineExecutor.submit`` and awaits the result. The underlying
``PipelineRunner`` is synchronous; the executor wraps it in
``run_in_executor`` so the event loop is not blocked.

``AsyncOmni`` shares ``OmniBase`` with ``Omni`` but does not subclass
``Omni``: the reference keeps them as siblings because their ``generate``
signatures differ (per-request ``sampling_params`` here, per-prompt
``sampling_params_list`` there). Inheriting one from the other would make
one of the two signatures a lie about substitutability.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from ..config.params import OmniPromptType, SamplingParams
from ..outputs import OmniRequestOutput
from .base import OmniBase
from .omni import _as_list, _make_output, _merge_modal


class AsyncOmni(OmniBase):
    async def generate(
        self,
        prompts: OmniPromptType | Iterable[OmniPromptType],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
        *,
        sampling_params_list: SamplingParams | list[SamplingParams] | None = None,
    ) -> AsyncIterator[OmniRequestOutput]:
        if sampling_params is not None and sampling_params_list is not None:
            raise ValueError("pass sampling_params or sampling_params_list, not both")
        prompts_list = _as_list(prompts)
        if sampling_params_list is None:
            per_prompt = [sampling_params] * len(prompts_list)
        elif isinstance(sampling_params_list, SamplingParams):
            per_prompt = [sampling_params_list] * len(prompts_list)
        else:
            per_prompt = list(sampling_params_list)
        executor = self._ensure_executor()
        final_output_type = self._final_output_type()
        prepared = [
            _merge_modal(per_prompt[idx] if idx < len(per_prompt) else None, prompt)
            for idx, prompt in enumerate(prompts_list)
        ]
        results = await executor.submit_many(
            [text for text, _ in prepared],
            [params for _, params in prepared],
        )
        for payload, stage_ms in results:
            yield _make_output(payload, stage_ms, final_output_type)


__all__ = ["AsyncOmni"]
