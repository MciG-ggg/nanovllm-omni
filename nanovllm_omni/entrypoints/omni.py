from ..engine_args import SamplingParams
from ..models.minimind_omni.stages import generate_audio
from ..outputs import OmniRequestOutput
from .base import OmniBase


class Omni(OmniBase):
    def _one(self, prompt, sampling_params=None):
        sp = sampling_params or SamplingParams()
        if not isinstance(sp, SamplingParams):
            # Tolerate plain namespace / dict-like callers.
            temperature = float(getattr(sp, "temperature", 0.7))
            top_p = float(getattr(sp, "top_p", 0.9))
            max_tokens = int(getattr(sp, "max_tokens", 16))
            extra = dict(getattr(sp, "extra", {}) or {})
        else:
            temperature = sp.temperature
            top_p = sp.top_p
            max_tokens = sp.max_tokens
            extra = dict(sp.extra or {})

        # Prefer thinker-stage defaults when caller left SamplingParams at stock values.
        if temperature == 1.0 and "temperature" not in extra:
            temperature = 0.7
        if top_p == 1.0:
            top_p = 0.9

        audio = generate_audio(
            self._ensure_bundle(),
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            open_thinking=bool(extra.get("open_thinking", False)),
        )
        return OmniRequestOutput.from_pipeline(audio)

    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str):
            prompts = [prompts]
        return [self._one(prompt, sampling_params) for prompt in prompts]
