from ..engine_args import SamplingParams, OmniEngineArgs
from ..outputs import OmniRequestOutput
from ..models.minimind_omni import load_minimind_omni_bundle
from ..runtime.pipeline import Pipeline
from ..runtime.orchestrator import Orchestrator

class OmniBase:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)
        self.bundle = load_minimind_omni_bundle(model, device=getattr(self.engine_args, "device", "cuda"))
        self.pipeline = Pipeline((self.bundle.thinker, self.bundle.talker, self.bundle.code2wav))
    def _one(self, prompt, sampling_params=None):
        return OmniRequestOutput.from_pipeline(Orchestrator().submit(self.pipeline, prompt))

class Omni(OmniBase):
    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        return [self._one(p, sampling_params) for p in prompts]

class AsyncOmni(OmniBase):
    async def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if isinstance(prompts, str): prompts = [prompts]
        for prompt in prompts:
            yield self._one(prompt, sampling_params)
