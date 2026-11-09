"""The MiniMind talker decode recipe, shared by every backend.

Framework-free on purpose: the MLX runner imports this module, and
importing torch on the Mac track would pull a second array framework
into the process.

The recipe is the vendor's ``stream_generate`` schedule. Eight codebooks,
each lagging one position; codebook ``i`` starts sampling real codes once
the step reaches ``i``; a frame is emitted once all eight are active.
"""

from __future__ import annotations

NUM_CODEBOOKS = 8
AUDIO_PAD_TOKEN = 2049
AUDIO_STOP_TOKEN = 2050
AUDIO_SPK_TOKEN = 2051
AUDIO_VENDOR_TEMPERATURE = 0.2
AUDIO_REPETITION_PENALTY = 1.05


def can_emit_frame(stop_positions: list[int | None], step: int) -> bool:
    """Whether the emission gate opens at ``step`` or at any later step.

    The gate needs all eight codebooks active, ``idx[i] = step - 7 + i``.
    ``idx`` only grows, so a codebook that stays inactive now stays inactive
    forever: a closed gate is permanent, which is what lets the decode loop
    stop early without changing the emitted frames.
    """
    lag = NUM_CODEBOOKS - 1
    return all(
        stop_positions[i] is None or step - lag + i < stop_positions[i]
        for i in range(NUM_CODEBOOKS)
    )


__all__ = [
    "AUDIO_PAD_TOKEN",
    "AUDIO_REPETITION_PENALTY",
    "AUDIO_SPK_TOKEN",
    "AUDIO_STOP_TOKEN",
    "AUDIO_VENDOR_TEMPERATURE",
    "NUM_CODEBOOKS",
    "can_emit_frame",
]
