"""Ordered in-memory pipeline used by the public model-free submission seam."""

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.payloads import AudioPayload
from nanovllm_omni.stage import Stage


@dataclass
class Pipeline:
    """Execute explicitly ordered stages and retain their typed outputs."""

    stages: tuple[Stage[Any, Any], ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("pipeline requires at least one stage")

    def run(self, request: str) -> tuple[AudioPayload, tuple[tuple[str, Any], ...]]:
        """Run one request through every stage, returning final audio and a trace."""
        payload: Any = request
        outputs: list[tuple[str, Any]] = []
        for stage in self.stages:
            payload = stage.execute(payload)
            outputs.append((stage.name, payload))
        if not isinstance(payload, AudioPayload):
            raise TypeError("pipeline final output must be an AudioPayload")
        return payload, tuple(outputs)
