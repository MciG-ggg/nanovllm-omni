"""Omni: synchronous aligned-Python entry point.

Drives a request through the configured pipeline via PipelineRunner.
Per-stage defaults come from the deploy YAML through merge_pipeline_deploy;
the caller-supplied SamplingParams overrides them per request.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Generator, Iterable
from typing import Any

from ..config.params import OmniPromptType, SamplingParams
from ..outputs import OmniRequestOutput
from .base import OmniBase


def _as_list(prompts: OmniPromptType | Iterable[OmniPromptType]) -> list[OmniPromptType]:
    if isinstance(prompts, (str, dict)):
        return [prompts]
    return list(prompts)


def _split_prompt(prompt: OmniPromptType) -> tuple[str, dict[str, Any] | None]:
    """Extract text + modal payload from a single prompt.

    A str prompt is text-only. A dict prompt has a ``prompt`` key (text)
    and may carry modal payloads (``image``, ...) that ride along in
    ``sampling.extra`` so stages can consume them without a new seam --
    smolvla already reads ``extra["image"]`` (HWC uint8 RGB or PIL bytes).
    """
    if not isinstance(prompt, dict):
        return prompt, None
    text = prompt.get("prompt", "")
    modal = {key: value for key, value in prompt.items() if key != "prompt" and value is not None}
    return str(text), (modal or None)


class Omni(OmniBase):
    def _one(
        self,
        prompt: OmniPromptType,
        sampling_params: SamplingParams | None = None,
    ) -> OmniRequestOutput:
        executor = self._ensure_executor()
        text, modal = _split_prompt(prompt)
        if modal:
            extra = dict(getattr(sampling_params, "extra", None) or {})
            extra.update(modal)
            sampling_params = (
                dataclasses.replace(
                    sampling_params,
                    extra=extra,
                )
                if sampling_params is not None
                else SamplingParams(extra=extra)
            )
        payload = executor._runner.run(text, sampling_params)
        return OmniRequestOutput.from_pipeline(
            payload,
            final_output_type=self._final_output_type(),
        )

    def _collect_kwargs(
        self,
        sampling_params_list: Iterable[SamplingParams] | None,
    ) -> list[SamplingParams | None]:
        """Materialize per-prompt sampling overrides.

        Accepts either a sequence (``sampling_params_list``, vllm-omni shape)
        or a single ``sampling_params``; returns one entry per prompt with
        ``None`` meaning "rely on deploy defaults".
        """
        if sampling_params_list is None:
            return [None]
        return list(sampling_params_list)

    def generate(
        self,
        prompts: OmniPromptType | Iterable[OmniPromptType],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,  # accepted for vllm-omni parity; no-op corpus is small
        *,
        sampling_params_list: Iterable[SamplingParams] | None = None,
        py_generator: bool = False,
    ) -> Any:
        """Generate for one or more prompts.

        vllm-omni parity: returns ``list[OmniRequestOutput]`` by default, or
        a lazy ``Generator[OmniRequestOutput]`` with ``py_generator=True``
        (each execution happens on iteration, in submission order). The single
        ``sampling_params`` applies to every prompt; ``sampling_params_list``
        gives per-prompt overrides (must match prompt count when non-empty).
        """
        prompts_list = _as_list(prompts)
        params_list = self._collect_kwargs(sampling_params_list or [sampling_params])
        if len(params_list) < len(prompts_list):
            params_list = params_list * len(prompts_list) if len(params_list) == 1 else params_list

        def _iter() -> Generator[OmniRequestOutput, None, None]:
            for idx, prompt in enumerate(prompts_list):
                sp = params_list[idx] if idx < len(params_list) else sampling_params
                yield self._one(prompt, sp)

        if py_generator:
            return _iter()
        return list(_iter())


__all__ = ["Omni"]
