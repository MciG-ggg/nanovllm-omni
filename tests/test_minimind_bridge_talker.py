"""Issue #11: Talker reads bridges only; watchdog caps post-bridge tail.

No torch / HF weights — pure typed-payload contract.
"""

from __future__ import annotations

from nanovllm_omni.models.minimind_omni import (
    MIMI_CODEBOOKS,
    MinimindTalker,
    apply_talker_watchdog,
    frames_from_bridges,
)
from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
    THINKER_FORCED_PADDING_DEFAULT,
    BridgePayload,
    CodecTokenPayload,
    TensorPayload,
    ThinkerRun,
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


def test_talker_reads_bridges_without_handle() -> None:
    """No shared-handle stash: Talker(handle=None) still works from ThinkerRun."""
    frame0 = tuple(range(1, 9))
    bridges = tuple(
        _bridge(frame0 if i == 0 else (), forced=(i > 0), step=max(0, i - 1))
        for i in range(THINKER_FORCED_PADDING_DEFAULT + 1)
    )
    run = ThinkerRun(
        bridges=bridges,
        visible_tokens=bridges[0].tokens,
        eos_token_id=0,
        forced_padding_count=THINKER_FORCED_PADDING_DEFAULT,
    )
    codec = MinimindTalker(handle=None).execute(run)
    assert isinstance(codec, CodecTokenPayload)
    assert codec.codebooks == MIMI_CODEBOOKS
    assert len(codec.active_mask) == 1
    assert codec.active_mask[0][0] is True
    # First codebook keeps real id; later codebooks inactive → padding on frame 0.
    assert codec.token_ids[0] == 1
    for k in range(1, MIMI_CODEBOOKS):
        assert codec.token_ids[k] == AUDIO_PADDING_TOKEN_ID
