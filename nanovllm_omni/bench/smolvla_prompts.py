"""Profile input set for smolvla (6 inputs x 20 runs each, thorough).

Each input is one (state, instruction) pair from a fixed LIBERO demo
episode. ``state`` carries proprioception + vision; ``instruction`` is
the high-level task description (constant across the episode).

# TODO(populate): pick a LIBERO task and a fixed demo episode, then
# extract 6 (state, instruction) pairs from it. Until populated, the
# tuple is empty so the bench short-circuits cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SmolVLAInput:
    """One smolvla profile input.

    ``state`` shape is lerobot-observation-specific (proprio + vision
    keys); concrete shape comes from the LIBERO demo chosen above.
    """

    id: str
    state: Any
    instruction: str
    action_chunk_size: int = 50
    num_flow_steps: int = 10


SMOLVLA_INPUTS: tuple[SmolVLAInput, ...] = (
    # TODO: replace with real (state, instruction) pairs from a fixed
    # LIBERO demo episode. See top-of-file note.
)
