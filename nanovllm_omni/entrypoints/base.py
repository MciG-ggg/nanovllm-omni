from ..engine_args import OmniEngineArgs
from ..outputs import OmniRequestOutput


class OmniBase:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.engine_args = OmniEngineArgs(model=model, **kwargs)

    def _one(self, prompt, sampling_params=None):
        return OmniRequestOutput(outputs=None, multimodal_output=None)
