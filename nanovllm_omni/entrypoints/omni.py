from .base import OmniBase


class Omni(OmniBase):
    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str):
            prompts = [prompts]
        return [self._one(prompt, sampling_params) for prompt in prompts]
