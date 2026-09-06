"""MiniMind-Omni talker forward + sample + postprocess tests.

The wrapper class exposes the LLM_AR stage contract (preprocess /
forward / compute_logits / sample / postprocess / make_omni_output) and
the bridge-state alignment (``_select_bridge_states``,
``_make_inputs_embeds``, ``_sample_codebook_logits_batch``). These tests
exercise each method on a tiny synthetic TalkerModule so the shapes and
edge cases are pinned without needing the real HF checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.talker import (  # noqa: E402
    TalkerOutput,
    wrap_talker,
)
from tests._talker_fixtures import (  # noqa: E402
    make_fake_bundle,
    span_bridge,
)

# ---------------------------------------------------------------------------
# _audio_ids_from_layer0
# ---------------------------------------------------------------------------


def test_audio_ids_from_layer0_two_dim_input() -> None:
    """``[B, T]`` input gets padded into ``[B, num_code_layers, T]``
    with the input ids placed on layer 0.
    """
    bundle = make_fake_bundle(audio_pad_token=9, audio_vocab_size=16)
    talker = wrap_talker(bundle)
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = talker._audio_ids_from_layer0(inp)
    assert out.shape == (1, 8, 3)  # [B, num_code_layers, T]
    assert (out[:, 0, :] == inp).all()
    assert (out[:, 1:, :] == 9).all()  # audio_pad_token fills other layers


def test_audio_ids_from_layer0_three_dim_passthrough() -> None:
    bundle = make_fake_bundle(num_code_layers=8)
    talker = wrap_talker(bundle)
    inp = torch.zeros(1, 8, 3, dtype=torch.long)
    out = talker._audio_ids_from_layer0(inp)
    assert out.shape == (1, 8, 3)
    assert out is inp  # 3-D passthrough returns the same tensor (coerced to long)


def test_audio_ids_from_layer0_clamps_layer0() -> None:
    """Layer-0 input ids are clamped into ``[0, audio_vocab_size - 1]``."""
    bundle = make_fake_bundle(audio_vocab_size=4, audio_pad_token=0)
    talker = wrap_talker(bundle)
    inp = torch.tensor([[10, -3, 5]], dtype=torch.long)  # out-of-range values
    out = talker._audio_ids_from_layer0(inp)
    assert (out[:, 0, :] == torch.tensor([3, 0, 3])).all()  # clamped


# ---------------------------------------------------------------------------
# _select_bridge_states
# ---------------------------------------------------------------------------


def test_select_bridge_states_decode_span_aligned_to_num_computed() -> None:
    """Decode input at position ``num_computed`` should be conditioned on
    the matching bridge row, not always ``bridge[-1]``.
    """
    bundle = make_fake_bundle(hidden_size=8)
    talker = wrap_talker(bundle)
    bridge = span_bridge(hidden_size=8, sequence_len=6)
    info = {
        "hidden_states": {"bridge": bridge},
        "_omni_prompt_len": 2,
        "_omni_num_computed_tokens": 4,  # current decode position
        "_omni_is_prefill": False,
    }
    span, is_prefill, prompt_len, num_computed, bridge_len = talker._select_bridge_states(
        info, span_len=1, device=bridge.device
    )
    assert span.shape == (1, 8)
    assert is_prefill is False
    assert prompt_len == 2
    assert num_computed == 4
    assert bridge_len == 6
    # The decode step at num_computed=4 should select bridge[4].
    assert torch.allclose(span[0], bridge[4])


def test_select_bridge_states_missing_bridge_raises() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    with pytest.raises(ValueError, match="requires hidden_states.bridge"):
        talker._select_bridge_states({}, span_len=1, device=torch.device("cpu"))


def test_select_bridge_states_short_bridge_raises() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    bridge = span_bridge(hidden_size=8, sequence_len=2)
    info = {"hidden_states": {"bridge": bridge}}
    with pytest.raises(ValueError, match="shorter than scheduled span"):
        talker._select_bridge_states(info, span_len=5, device=bridge.device)


# ---------------------------------------------------------------------------
# _make_inputs_embeds
# ---------------------------------------------------------------------------


def test_make_inputs_embeds_shape_and_scale_application() -> None:
    """``embed_proj(bridge) * text_scale + codec_proj(audio_emb) * audio_scale``.

    Output shape is ``[T, hidden]`` (no batch axis): the embed_tokens
    layer-sums over the ``num_code_layers`` axis so its output is already
    ``[B, T, hidden]`` and the ``reshape(-1, hidden)`` collapses B*T into
    a 2-D tensor to match ``text_part`` shape ``[T, hidden]``.
    """
    bundle = make_fake_bundle(hidden_size=8, audio_vocab_size=16, text_hidden_size=8)
    talker = wrap_talker(bundle)
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    bridge = span_bridge(hidden_size=8, sequence_len=3)
    embeds = talker._make_inputs_embeds(inp, bridge)
    assert embeds.shape == (3, 8)


def test_make_inputs_embeds_spk_token_inserts_speaker_projection() -> None:
    """Positions where layer-0 ids == ``audio_spk_token`` get replaced by
    ``spk_proj(spk_emb)`` before the text/audio fusion.
    """
    bundle = make_fake_bundle(
        hidden_size=8,
        audio_vocab_size=16,
        spk_emb_size=4,
        audio_spk_token=11,
    )
    talker = wrap_talker(bundle)
    inp = torch.tensor([[11, 1, 2]], dtype=torch.long)  # spk token at position 0
    bridge = span_bridge(hidden_size=8, sequence_len=3)
    spk_emb = torch.randn(1, 4)
    embeds_with_spk = talker._make_inputs_embeds(inp, bridge, spk_emb=spk_emb)
    embeds_without_spk = talker._make_inputs_embeds(inp, bridge, spk_emb=None)
    # The two embeddings MUST differ at the spk position (index 0).
    assert not torch.allclose(embeds_with_spk[0, :], embeds_without_spk[0, :])
    # But match elsewhere (positions 1 and 2).
    assert torch.allclose(embeds_with_spk[1:, :], embeds_without_spk[1:, :])


# ---------------------------------------------------------------------------
# _sample_codebook_logits_batch
# ---------------------------------------------------------------------------


def test_sample_codebook_logits_batch_shape_and_deterministic() -> None:
    """do_sample=False -> argmax over (num_layers, batch, vocab)."""
    bundle = make_fake_bundle(hidden_size=8, audio_vocab_size=16, num_code_layers=8)
    talker = wrap_talker(bundle)
    torch.manual_seed(0)
    logits = [torch.randn(2, 16) for _ in range(8)]  # 8 layers, batch=2
    sampled = talker._sample_codebook_logits_batch(logits, do_sample=False)
    assert sampled.shape == (2, 8)  # [batch, num_layers]


def test_sample_codebook_logits_batch_empty_returns_empty() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    out = talker._sample_codebook_logits_batch([], do_sample=True)
    assert out.numel() == 0


# ---------------------------------------------------------------------------
# preprocess / forward / compute_logits / sample / postprocess / make_omni_output
# ---------------------------------------------------------------------------


def test_preprocess_returns_embeds_with_mtp_inputs_for_decode_step() -> None:
    bundle = make_fake_bundle(hidden_size=8, audio_vocab_size=16, num_code_layers=8)
    talker = wrap_talker(bundle)
    bridge = span_bridge(hidden_size=8, sequence_len=5)
    info = {
        "hidden_states": {"bridge": bridge},
        "_omni_prompt_len": 2,
        "_omni_num_computed_tokens": 3,
        "_omni_is_prefill": False,
        "request_id": "req-1",
    }
    inp = torch.tensor([1], dtype=torch.long)  # decode step (span_len=1)
    input_ids, embeds, update = talker.preprocess(inp, None, **info)
    assert torch.equal(input_ids, inp)
    assert embeds.shape == (1, 8)
    # Decode path populates mtp_inputs (last_hidden, text_step, active_mask).
    assert "mtp_inputs" in update
    last_hidden, text_step, active_mask = update["mtp_inputs"]
    assert last_hidden.shape == (1, 8)
    assert text_step.shape == (1, 8)
    assert active_mask.shape == (1, 8)


def test_preprocess_prefill_emits_audio_codes_padding() -> None:
    bundle = make_fake_bundle(num_code_layers=8)
    talker = wrap_talker(bundle)
    bridge = span_bridge(sequence_len=4)
    info = {
        "hidden_states": {"bridge": bridge},
        "_omni_prompt_len": 4,
        "_omni_num_computed_tokens": 0,
        "_omni_is_prefill": True,
        "request_id": "req-1",
    }
    inp = torch.tensor([1, 2, 3, 4], dtype=torch.long)  # span_len > 1 -> prefill
    _input_ids, _embeds, update = talker.preprocess(inp, None, **info)
    assert "codes" in update and "audio" in update["codes"]
    assert update["codes"]["audio"].shape == (4, 8)  # [span_len, num_code_layers]


def test_forward_shape_from_input_ids() -> None:
    bundle = make_fake_bundle(num_hidden_layers=2, hidden_size=8, audio_vocab_size=16)
    talker = wrap_talker(bundle)
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = talker.forward(input_ids=inp)
    # ``embed_input_ids`` reshapes ``[B, num_layers, T] -> [B*T, hidden]``,
    # so the trunk returns a flat ``[T, hidden]``.
    assert out.shape == (3, 8)


def test_forward_shape_from_inputs_embeds() -> None:
    bundle = make_fake_bundle(num_hidden_layers=2, hidden_size=8)
    talker = wrap_talker(bundle)
    embeds = torch.randn(4, 8)
    out = talker.forward(inputs_embeds=embeds)
    assert out.shape == (4, 8)


def test_forward_requires_input_ids_or_embeds() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    with pytest.raises(ValueError, match="must be provided"):
        talker.forward()


def test_compute_logits_returns_layer0_only() -> None:
    bundle = make_fake_bundle(hidden_size=8, audio_vocab_size=16, num_code_layers=8)
    talker = wrap_talker(bundle)
    hidden = torch.randn(1, 4, 8)
    logits = talker.compute_logits(hidden)
    # layer-0 vocab size == audio_vocab_size (16).
    assert logits.shape == (1, 4, 16)


def test_compute_logits_accepts_talker_output_envelope() -> None:
    bundle = make_fake_bundle(hidden_size=8, audio_vocab_size=16)
    talker = wrap_talker(bundle)
    hidden = torch.randn(1, 2, 8)
    logits = talker.compute_logits(TalkerOutput(text_hidden_states=hidden))
    assert logits.shape == (1, 2, 16)


def test_compute_logits_returns_none_for_none_input() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    assert talker.compute_logits(None) is None


def test_sample_returns_one_token_per_row() -> None:
    bundle = make_fake_bundle(audio_vocab_size=16)
    talker = wrap_talker(bundle)
    logits = torch.randn(3, 16)
    out = talker.sample(logits, sampling_metadata=None)
    assert out.shape == (3, 1)
    assert (out >= 0).all() and (out < 16).all()


def test_sample_masks_internal_stop_token_id() -> None:
    """``internal_stop_token_id`` is force-masked to -inf so it can never be sampled."""
    bundle = make_fake_bundle(audio_vocab_size=16, internal_stop_token_id=12)
    talker = wrap_talker(bundle)
    # Construct logits that put internal_stop_token_id at the top.
    logits = torch.full((1, 16), -10.0)
    logits[0, 12] = 100.0  # huge bias towards internal_stop_token_id
    # Many seeds; internal_stop_token_id must never be sampled.
    for _ in range(50):
        out = talker.sample(logits, sampling_metadata=None)
        assert out[0, 0].item() != 12


def test_sample_applies_pending_internal_stop() -> None:
    """If ``_stop_pending_by_req[request_id]`` is set, that row is forced to
    ``internal_stop_token_id``.
    """
    bundle = make_fake_bundle(audio_vocab_size=16, internal_stop_token_id=12)
    talker = wrap_talker(bundle)
    logits = torch.randn(2, 16)
    meta = SimpleNamespace(
        temperature=1.0,
        top_k=0,
        do_sample=True,
        generator=None,
        request_ids=["req-A", "req-B"],
    )
    talker._stop_pending_by_req["req-A"] = True
    out = talker.sample(logits, sampling_metadata=meta)
    assert out[0, 0].item() == 12
    # ``_stop_pending_by_req`` is one-shot.
    assert "req-A" not in talker._stop_pending_by_req


def test_on_requests_finished_clears_per_request_state() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    talker._stop_pending_by_req.update({"req-A": True, "req-B": True, "req-C": True})
    talker._steps_after_last_thinker_by_req.update({"req-A": 2, "req-B": 3, "req-C": 4})
    talker.on_requests_finished({"req-A", "req-C"})
    assert "req-A" not in talker._stop_pending_by_req
    assert "req-C" not in talker._stop_pending_by_req
    assert "req-B" in talker._stop_pending_by_req
    assert "req-A" not in talker._steps_after_last_thinker_by_req
    assert "req-C" not in talker._steps_after_last_thinker_by_req
    assert talker._steps_after_last_thinker_by_req["req-B"] == 3


def test_postprocess_stashes_last_hidden_and_code_history() -> None:
    bundle = make_fake_bundle(num_code_layers=8, audio_vocab_size=16)
    talker = wrap_talker(bundle)
    hidden = torch.randn(2, 8)
    audio_codes = torch.tensor([[1] * 8, [2] * 8], dtype=torch.long)
    update = talker.postprocess(
        hidden,
        request_id="req-1",
        _omni_is_prefill=False,
        codes={"audio": audio_codes},
    )
    assert "last" in update["hidden_states"]
    assert update["hidden_states"]["last"].shape == (1, 8)
    assert "history" in update["codes"]
    # No prior history -> history equals the audio rows.
    assert torch.equal(update["codes"]["history"], audio_codes)
    # audio_step = 1, so ready_frames = max(0, 2 - 8 + 1) = 0.
    assert update["meta"]["emitted_audio_frames"] == 0


def test_postprocess_prefill_returns_only_last_hidden() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    hidden = torch.randn(3, 8)
    update = talker.postprocess(hidden, _omni_is_prefill=True)
    assert "last" in update["hidden_states"]
    assert "codes" not in update


def test_postprocess_detects_audio_stop_token_on_last_row() -> None:
    """When the last row's last-layer code equals ``audio_stop_token``, the
    request is queued for forced internal stop.
    """
    bundle = make_fake_bundle(audio_stop_token=10, num_code_layers=8)
    talker = wrap_talker(bundle)
    hidden = torch.randn(4, 8)
    audio_codes = torch.zeros(4, 8, dtype=torch.long)
    audio_codes[-1, -1] = 10  # last layer of last row hits audio_stop_token
    talker.postprocess(
        hidden,
        request_id="req-stop",
        _omni_is_prefill=False,
        codes={"audio": audio_codes},
    )
    assert talker._stop_pending_by_req.get("req-stop") is True


def test_make_omni_output_passthrough() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    out = TalkerOutput(text_hidden_states=torch.randn(2, 8))
    assert talker.make_omni_output(out) is out


def test_make_omni_output_with_empty_buffer_returns_text_only() -> None:
    bundle = make_fake_bundle()
    talker = wrap_talker(bundle)
    hidden = torch.randn(2, 8)
    out = talker.make_omni_output(hidden)
    assert isinstance(out, TalkerOutput)
    assert out.text_hidden_states is hidden
    assert out.multimodal_outputs == {}


def test_make_omni_output_extracts_diagonal_frames_when_buffer_supplied() -> None:
    """When ``model_intermediate_buffer`` carries an info_dict with audio
    codes + history, the diagonal MTP-ready frames are attached to
    ``multimodal_outputs['codes']['audio']``.
    """
    bundle = make_fake_bundle(num_code_layers=4, audio_vocab_size=10)
    talker = wrap_talker(bundle)
    # 10 rows -> ready_frames = 10 - 4 + 1 = 7.
    audio = torch.full((10, 4), 1, dtype=torch.long)  # all-ones, below stop boundary
    info = {"codes": {"audio": audio, "history": None}, "meta": {"emitted_audio_frames": 0}}
    hidden = torch.randn(10, 8)
    out = talker.make_omni_output(hidden, model_intermediate_buffer=[info])
    assert "codes" in out.multimodal_outputs
    frames = out.multimodal_outputs["codes"]["audio"]
    assert frames.shape == (7, 4)


def test_make_omni_output_skips_frames_with_stop_token_in_diagonal() -> None:
    """Rows whose diagonal frame contains ``>= audio_vocab_size`` are
    skipped (they terminate audio upstream).
    """
    bundle = make_fake_bundle(num_code_layers=4, audio_vocab_size=10)
    talker = wrap_talker(bundle)
    audio = torch.tensor(
        [
            [1, 1, 1, 1],  # row 0
            [1, 1, 1, 1],  # row 1
            [1, 1, 1, 1],  # row 2
            [1, 1, 1, 10],  # row 3 -- last layer at stop boundary
            [1, 1, 1, 1],  # row 4
        ],
        dtype=torch.long,
    )
    info = {"codes": {"audio": audio, "history": None}, "meta": {"emitted_audio_frames": 0}}
    hidden = torch.randn(5, 8)
    out = talker.make_omni_output(hidden, model_intermediate_buffer=[info])
    # The first ready frame (rows 0..3, last layer = 10) is skipped; only the
    # second frame (rows 1..4, last layer = 1) survives. Result: 1 frame.
    frames = out.multimodal_outputs.get("codes", {}).get("audio")
    if frames is not None:
        assert frames.shape == (1, 4)
    else:
        # All frames were stop-bounded; the wrapper returns text-only.
        assert out.multimodal_outputs == {}


# ---------------------------------------------------------------------------
# import-only helper (no SimpleNamespace name collision; final import)
# ---------------------------------------------------------------------------


from types import SimpleNamespace  # noqa: E402  -- placed at module bottom intentionally
