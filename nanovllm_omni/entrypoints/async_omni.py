from .base import OmniBase


class AsyncOmni(OmniBase):
    async def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str):
            prompts = [prompts]
        for prompt in prompts:
            yield self._one(prompt, sampling_params)
