"""Minimal engine wrapper for aligned public API."""


class OmniEngine:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def generate_one(self, prompt):
        if self.pipeline is None:
            return None
        return self.pipeline(prompt) if callable(self.pipeline) else None
