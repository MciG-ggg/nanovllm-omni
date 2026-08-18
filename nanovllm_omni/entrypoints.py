from .engine_args import SamplingParams, OmniEngineArgs
from .outputs import OmniRequestOutput
from .config_registry import *
from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle
from nanovllm_omni.runtime.pipeline import Pipeline
from nanovllm_omni.runtime.orchestrator import Orchestrator
import asyncio

class OmniBase:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)
        self.bundle = load_minimind_omni_bundle(model, device=self.engine_args.device)
        self.pipeline = Pipeline((self.bundle.thinker, self.bundle.talker, self.bundle.code2wav))
    def _one(self, prompt, sampling_params=None):
        result = Orchestrator().submit(self.pipeline, prompt)
        return OmniRequestOutput.from_pipeline(result)

class Omni(OmniBase):
    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        return [self._one(p, sampling_params) for p in prompts]

class AsyncOmni(OmniBase):
    async def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        for prompt in prompts:
            yield self._one(prompt, sampling_params)
