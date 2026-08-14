"""Issue #11: typed-bridge helpers and watchdog caps.

The real MiniMind-O Talker runs ``self.talker`` (TalkerModule) end-to-end,
so this file no longer pretends to exercise a fake Talker without
weights. We keep the pure typed-payload helpers (``frames_from_bridges``,
``apply_talker_watchdog``) and the default constants so other tests can
import them.
"""

from __future__ import annotations

from nanovllm_omni.models.minimind_omni import (
    MinimindTalker,
    apply_talker_watchdog,
    frames_from_bridges,
)
from nanovllm_omni.payloads import (
    TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
    BridgePayload,
    TensorPayload,
    TokenPayload,
)


def _bridge(codes: tuple[int, ...] = (), *, forced: bool = False, step: int = 0) -> BridgePayload:
    meta = {"forced": "true", "step": str(step)} if forced else {}
    return BridgePayload(
        tokens=TokenPayload(token_ids=(1,), text="" if forced else "hi", metadata=meta),
        hidden_states=TensorPayload(values=(1.0,), shape=(1, 1)),
        audio_codes=codes,
    )


def test_frames_from_bridges_splits_codebook_chunks() -> None:
    bridges = (
        _bridge(tuple(range(8))),
        _bridge(tuple(range(8, 16))),
        _bridge(),  # empty forced pad
    )
    frames = frames_from_bridges(bridges, codebooks=8)
    assert frames == [list(range(8)), list(range(8, 16))]


def test_overflow_frames_on_last_bridge_are_recovered() -> None:
    # Two frames flattened onto the final bridge (simulates pack overflow).
    flat = tuple(range(16))
    bridges = (_bridge(), _bridge(flat, forced=True))
    assert frames_from_bridges(bridges, codebooks=8) == [list(range(8)), list(range(8, 16))]


def test_watchdog_caps_post_bridge_tail() -> None:
    frames = [[i] * 8 for i in range(20)]
    capped = apply_talker_watchdog(frames, thinker_bridge_count=5, max_steps_after_last=3)
    assert len(capped) == 8  # 5 bridge-conditioned + 3 tail


def test_watchdog_negative_disables() -> None:
    frames = [[i] * 8 for i in range(50)]
    assert apply_talker_watchdog(frames, 1, max_steps_after_last=-1) == frames


def test_talker_default_watchdog_constant() -> None:
    assert TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN == 192
    assert MinimindTalker.max_steps_after_last_thinker_token == 192
