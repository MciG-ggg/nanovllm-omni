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
    """Output of the vlm (AR) stage: backbone hidden states + robot state."""

    prefix_states: Any  # torch.Tensor [T, H] backbone hidden states
    robot_state: Any  # torch.Tensor [state_dim] normalized proprio
    attention_mask: Any | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionInputPayload:
    """Input to the action (flow matching) stage."""

    prefix_states: Any
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
        prefix_states=payload.prefix_states,
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
