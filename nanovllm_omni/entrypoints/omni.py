"""Omni: synchronous aligned-Python entry point.

Drives a request through the configured pipeline via PipelineRunner.
Per-stage defaults come from the deploy YAML through merge_pipeline_deploy;
the caller-supplied SamplingParams overrides them per request.
"""

from __future__ import annotations

from ..config.params import SamplingParams
from ..outputs import OmniRequestOutput
from .base import OmniBase


class Omni(OmniBase):
    def _one(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> OmniRequestOutput:
        executor = self._ensure_executor()
        payload = executor._runner.run(prompt, sampling_params)
        return OmniRequestOutput.from_pipeline(
            payload,
            final_output_type=self._final_output_type(),
        )

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> list[OmniRequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        results: list[OmniRequestOutput] = []
        for prompt in prompts:
            results.append(self._one(prompt, sampling_params))
        return results
