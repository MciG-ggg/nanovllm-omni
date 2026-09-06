"""Focused local Talker MTP contract tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402
from tests._talker_fixtures import make_fake_bundle  # noqa: E402


def _mtp_inputs(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(1, batch_size + 1, dtype=torch.long),
        torch.zeros(batch_size, 8),
        torch.arange(batch_size * 8, dtype=torch.float32).reshape(batch_size, 8),
        torch.ones(batch_size, 8),
    )


def test_talker_mtp_greedy_uses_real_adapter_heads() -> None:
    talker = wrap_talker(make_fake_bundle(audio_vocab_size=16))
    input_ids, input_embeds, hidden, text_step = _mtp_inputs()

    output = talker.talker_mtp(
        input_ids,
        input_embeds,
        hidden,
        text_step,
        do_sample=False,
        active_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    expected_residual = torch.stack(
        [logits.argmax(dim=-1) for logits in talker.lm_head(hidden)[1:]], dim=1
    )

    assert output.shape == (1, 8)
    assert output.dtype == torch.long
    assert torch.equal(output[:, 0], input_ids)
    assert torch.equal(output[:, 1:], expected_residual)


def test_talker_mtp_supports_batch_two() -> None:
    talker = wrap_talker(make_fake_bundle(audio_vocab_size=16))
    input_ids, input_embeds, hidden, text_step = _mtp_inputs(batch_size=2)
    output = talker.talker_mtp(
        input_ids,
        input_embeds,
        hidden,
        text_step,
        do_sample=False,
        active_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    assert output.shape == (2, 8)
    assert torch.equal(output[:, 0], input_ids)


def test_talker_mtp_rejects_broadcasting_active_mask() -> None:
    talker = wrap_talker(make_fake_bundle())
    args = _mtp_inputs(batch_size=2)
    with pytest.raises(ValueError, match="active_mask must have shape"):
        talker.talker_mtp(*args, active_mask=torch.ones(8, dtype=torch.bool))
    with pytest.raises(ValueError, match="active_mask must have shape"):
        talker.talker_mtp(*args, active_mask=torch.ones(2, 7, dtype=torch.bool))


def test_talker_mtp_replaces_inactive_positions_with_pad() -> None:
    talker = wrap_talker(make_fake_bundle(audio_pad_token=15, audio_vocab_size=16))
    args = _mtp_inputs()
    active_mask = torch.tensor([[False, True, False, True, True, True, True, True]])
    output = talker.talker_mtp(*args, do_sample=False, active_mask=active_mask)
    assert output[0, 0].item() == 15
    assert output[0, 2].item() == 15
    assert (output[0, active_mask[0]] != 15).all()


def test_talker_mtp_seeded_sampling_is_reproducible() -> None:
    talker = wrap_talker(make_fake_bundle(audio_vocab_size=16))
    args = _mtp_inputs()
    active_mask = torch.ones(1, 8, dtype=torch.bool)
    first = torch.Generator().manual_seed(1234)
    second = torch.Generator().manual_seed(1234)
    output_a = talker.talker_mtp(
        *args,
        active_mask=active_mask,
        temperature=1.0,
        top_k=0,
        generator=first,
    )
    output_b = talker.talker_mtp(
        *args,
        active_mask=active_mask,
        temperature=1.0,
        top_k=0,
        generator=second,
    )
    assert torch.equal(output_a, output_b)


def test_code_layer_masks_cover_delayed_activation_steps() -> None:
    talker = wrap_talker(make_fake_bundle())
    assert talker._code_layer_masks[0].sum().item() == 0  # step 0: no audio
    assert talker._code_layer_masks[1].sum().item() == 1  # step 1: layer 0
    assert talker._code_layer_masks[8].sum().item() == 8  # all residual layers


def test_ready_diagonal_frames_are_ordered_and_emit_once() -> None:
    talker = wrap_talker(make_fake_bundle(num_code_layers=4, audio_vocab_size=100))
    history = torch.tensor(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        dtype=torch.long,
    )
    current = torch.tensor(
        [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]],
        dtype=torch.long,
    )
    frames = talker._ready_diagonal_audio_frames(history, current, emitted_frames=1)
    assert frames is not None
    assert torch.equal(
        frames,
        torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], dtype=torch.long),
    )
    assert talker._ready_diagonal_audio_frames(history, current, emitted_frames=4) is None


def test_ready_diagonal_frames_reject_stop_and_pad_boundaries() -> None:
    talker = wrap_talker(
        make_fake_bundle(
            num_code_layers=4,
            audio_vocab_size=16,
            audio_pad_token=14,
            audio_stop_token=15,
        )
    )
    current = torch.tensor(
        [[1, 1, 1, 15], [2, 2, 2, 2], [3, 3, 14, 3], [4, 4, 4, 4]],
        dtype=torch.long,
    )
    frames = talker._ready_diagonal_audio_frames(None, current, emitted_frames=0)
    assert frames is None


def test_on_requests_finished_cleans_phase_two_state() -> None:
    talker = wrap_talker(make_fake_bundle())
    talker._stop_pending_by_req["req-1"] = True
    talker._steps_after_last_thinker_by_req["req-1"] = 3
    talker.on_requests_finished({"req-1"})
    assert "req-1" not in talker._stop_pending_by_req
    assert "req-1" not in talker._steps_after_last_thinker_by_req
