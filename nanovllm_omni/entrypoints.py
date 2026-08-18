from .engine_args import SamplingParams, OmniEngineArgs
from .outputs import OmniRequestOutput

class OmniBase:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)
    def _one(self, prompt, sampling_params=None):
        return OmniRequestOutput(outputs=None, multimodal_output=None)

class Omni(OmniBase):
    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        return [self._one(p, sampling_params) for p in prompts]

class AsyncOmni(OmniBase):
    async def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        for prompt in prompts:
            yield self._one(prompt, sampling_params)
