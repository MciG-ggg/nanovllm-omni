"""The talker runs only through the content plus the 8-step codebook tail."""

import pytest


def test_num_steps_is_capped_at_content_plus_tail() -> None:
    torch = pytest.importorskip("torch")
    from nanovllm_omni.models.minimind_omni.stage_processors import (
        ThinkerStageOutput,
        thinker2talker,
    )

    output = ThinkerStageOutput(
        bridge_states=torch.zeros(20, 4),
        prompt_token_ids=(1, 2),
        output_token_ids=tuple(range(18)),
        text_token_ids=(1, 2) + tuple(range(18)),
        metadata={"content_len": 4},
    )

    payload = thinker2talker(output)

    assert payload.metadata["start_pos"] == 2
    assert payload.metadata["num_steps"] == 12


def test_vendor_tail_covers_the_teacher_forced_stop() -> None:
    from nanovllm_omni.models.minimind_omni.stage import _TALKER_TAIL

    assert _TALKER_TAIL >= 56
