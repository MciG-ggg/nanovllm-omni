"""Minimal engine wrapper for aligned public API."""

from __future__ import annotations

from typing import Any

from nanovllm_omni.models.minimind_omni.stages import MinimindBundle, generate_audio
from nanovllm_omni.outputs import AudioPayload


class OmniEngine:
    def __init__(self, pipeline=None, bundle: MinimindBundle | None = None):
        self.pipeline = pipeline
        self.bundle = bundle

    def generate_one(self, prompt: str, **kwargs: Any) -> AudioPayload | Any:
        if self.bundle is not None:
            return generate_audio(self.bundle, prompt, **kwargs)
        if self.pipeline is None:
            return None
        return self.pipeline(prompt) if callable(self.pipeline) else None
