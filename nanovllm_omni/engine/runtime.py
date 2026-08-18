"""Small local runtime wrapping the legacy ordered pipeline."""
from nanovllm_omni.runtime.pipeline import Pipeline
from nanovllm_omni.runtime.orchestrator import Orchestrator

class OmniEngine:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.orchestrator = Orchestrator()
    def generate_one(self, prompt):
        return self.orchestrator.submit(self.pipeline, prompt)
