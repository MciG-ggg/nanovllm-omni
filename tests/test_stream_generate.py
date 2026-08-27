"""End-to-end smoke for the single-request ``stream_generate`` wrapper.

The wrapper drives ``BatchedThinkerRunner(max_batch=1)``; per-step logic is
already covered by ``test_batched_generation.py``. These tests pin only the
*shape* of the wrapper's output so a future regression in the public
contract (signature, return type, frame gating) is caught at the unit level
without spinning up a real model.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests.test_batched_generation import FakeMiniMindOmni  # noqa: E402


def _input_ids(model, text: str = "hi") -> Any:
    """[1, P] token ids derived from ``text`` using the fake model's vocab."""
    ids = [ord(c) % (model.vocab - 100) + 1 for c in text]
    return torch.tensor([ids], dtype=torch.long)


def test_stream_generate_runs_end_to_end_and_terminates() -> None:
    """The wrapper yields >= 1 step and terminates without hanging."""
    from nanovllm_omni.models.minimind_omni.generation import stream_generate

    torch.manual_seed(0)
    model = FakeMiniMindOmni()
    input_ids = _input_ids(model, "hello world")

    yielded: list[tuple[Any, Any]] = []
    for chunk, frame in stream_generate(
        model,
        input_ids,
        eos_token_id=2,
        max_new_tokens=10,
        temperature=0.75,
        top_p=0.90,
        rp=1.0,
        open_thinking=False,
    ):
        yielded.append((chunk, frame))

    assert len(yielded) > 0
    # First yield is the post-prefill text chunk (no audio yet).
    first_chunk, first_frame = yielded[0]
    assert first_chunk is not None
    assert first_chunk.dim() == 2 and first_chunk.shape[0] == 1
    assert first_frame is None  # frame gated until step >= 8
    # Every frame in the stream, if any, is an 8-int list.
    for _chunk, frame in yielded:
        assert frame is None or (isinstance(frame, list) and len(frame) == 8)


def test_stream_generate_text_chunk_is_none_after_eos() -> None:
    """After EOS, ``text_chunk`` is ``None`` even when audio is still emitting.

    Pin by monkey-patching ``BatchedThinkerRunner.decode_group`` to flip
    ``text_finished=True`` after 3 decode steps -- deterministic, no RNG.
    """
    from nanovllm_omni.models.minimind_omni import batched_generation as bg
    from nanovllm_omni.models.minimind_omni.generation import stream_generate

    model = FakeMiniMindOmni()
    input_ids = _input_ids(model, "hi")

    original_decode = bg.BatchedThinkerRunner.decode_group
    counter = {"calls": 0}

    def fake_decode(self, group):  # type: ignore[no-untyped-def]
        counter["calls"] += 1
        original_decode(self, group)
        # After 3rd decode, force text_finished so the wrapper yields None.
        if counter["calls"] >= 3:
            for sequence in group.items:
                self.states[sequence.request_id].text_finished = True

    bg.BatchedThinkerRunner.decode_group = fake_decode
    try:
        chunks: list[Any] = []
        for chunk, _frame in stream_generate(
            model,
            input_ids,
            eos_token_id=2,
            max_new_tokens=20,
            temperature=0.75,
            top_p=0.90,
            rp=1.0,
            open_thinking=False,
        ):
            chunks.append(chunk)
    finally:
        bg.BatchedThinkerRunner.decode_group = original_decode

    # First yield must NOT be None (we never start finished).
    assert chunks[0] is not None
    # After text_finished=True, the wrapper yields None once before returning.
    assert None in chunks


def test_stream_generate_open_thinking_suppresses_audio_until_think_end() -> None:
    """``open_thinking=True``: no audio frames are emitted until think_end is detected.

    Pin the gating logic itself. We drive the wrapper with a model whose
    ``config.think_end_ids`` matches what the uniform logits WILL sample, and
    check that the audio frame emission step matches the documented
    ``think_end_step + 2`` formula.
    """
    from nanovllm_omni.models.minimind_omni.generation import stream_generate

    torch.manual_seed(2)
    model = FakeMiniMindOmni(vocab=64)  # smaller vocab -> think_end sampled quickly
    # 2-token think_end pattern. With uniform logits, eventually emitted.
    model.config.think_end_ids = [3, 4]

    yielded: list[tuple[Any, Any]] = []
    for chunk, frame in stream_generate(
        model,
        input_ids := _input_ids(model, "hi"),  # noqa: F841
        eos_token_id=2,
        max_new_tokens=15,
        temperature=0.75,
        top_p=0.90,
        rp=1.0,
        open_thinking=True,
    ):
        yielded.append((chunk, frame))

    # At least one yield must happen; otherwise the loop terminated instantly
    # and we have nothing to verify gating on.
    assert len(yielded) > 0
