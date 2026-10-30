"""Emission gate for the MiniMind-O talker decode loop.

The talker emits a frame at ``step >= 7`` only when all eight codebooks are
still active. ``idx[i] = step - 7 + i`` only grows, so a closed gate never
reopens; that permanence is what makes the early exit output-identical.
"""

from nanovllm_omni.models.minimind_omni.stage import _can_emit_frame


def test_gate_closes_when_a_stopped_codebook_is_reached() -> None:
    stop_positions = [None, None, None, 5, None, None, None, None]

    assert _can_emit_frame(stop_positions, step=8) is True
    assert _can_emit_frame(stop_positions, step=9) is False


def test_gate_stays_closed_for_every_later_step() -> None:
    stop_positions = [None, None, None, 5, None, None, None, None]

    assert all(_can_emit_frame(stop_positions, step=step) is False for step in range(9, 120))


def test_gate_ignores_codebooks_that_never_stop() -> None:
    assert _can_emit_frame([None] * 8, step=119) is True
