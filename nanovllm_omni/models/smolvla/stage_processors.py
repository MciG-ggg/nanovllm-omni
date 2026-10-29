"""Bridge payloads between vlm and action stages.

Modeled after ``minimind_omni/stage_processors.py`` (ADR-028).
Frozen dataclasses + pure functions; no model calls.

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §3 ADR-028.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VlmStageOutput:
    """Output of the vlm (AR) stage: robot state + the KV cache in metadata.

    The prefix KV cache and pad masks travel in ``metadata`` because the
    action stage consumes them as one dict; they are not per-request
    tensor fields on this payload.
    """

    robot_state: Any  # torch.Tensor [state_dim] normalized proprio
    attention_mask: Any | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionInputPayload:
    """Input to the action (flow matching) stage."""

    robot_state: Any
    chunk_len: int
    action_dim: int
    attention_mask: Any | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def vlm2action(payload: Any, prompt: str = "") -> ActionInputPayload:
    """Bridge function: VlmStageOutput → ActionInputPayload.

    Pure function: shape validation + conversion, no tokenize, no model call.
    Registered as ``process_input`` in the pipeline topology.
    """
    if not isinstance(payload, VlmStageOutput):
        raise TypeError(f"vlm2action expects VlmStageOutput, got {type(payload).__name__}")
    metadata = dict(payload.metadata)
    metadata["instruction"] = prompt
    return ActionInputPayload(
        robot_state=payload.robot_state,
        chunk_len=metadata.get("chunk_len", 10),
        action_dim=metadata.get("action_dim", 7),
        attention_mask=payload.attention_mask,
        request_id=payload.request_id,
        metadata=metadata,
    )


__all__ = [
    "ActionInputPayload",
    "VlmStageOutput",
    "vlm2action",
]
