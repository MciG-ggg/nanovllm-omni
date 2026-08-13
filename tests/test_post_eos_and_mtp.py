"""External-behavior tests for GitHub issue #3 (post-EOS + MTP semantics).

Each test asserts behavior observable through the public pipeline seam
(``Orchestrator.submit`` / ``Pipeline.run``) plus the typed payloads the
seam emits. Internal counters, the per-request state object, and any
private orchestrator attributes are not inspected.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from nanovllm_omni.orchestrator import Orchestrator
from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    THINKER_FORCED_PADDING_DEFAULT,
    AudioPayload,
    CodecTokenPayload,
    ThinkerRun,
)
from nanovllm_omni.pipeline import Pipeline
from nanovllm_omni.stage import FakeCode2Wav, FakeTalker, FakeThinker

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def make_pipeline(
    forced_padding_count: int = THINKER_FORCED_PADDING_DEFAULT,
    codebooks: int = 4,
) -> Pipeline:
    return Pipeline(
        (
            FakeThinker(forced_padding_count=forced_padding_count),
            FakeTalker(codebooks=codebooks),
            FakeCode2Wav(),
        )
    )


class _CapturingTalker:
    """Records the ThinkerRun it receives so a test can inspect the bridge stream."""

    name = "talker"

    def __init__(self) -> None:
        self.received_runs: list[ThinkerRun] = []

    def execute(self, payload: ThinkerRun) -> CodecTokenPayload:  # type: ignore[override]
        self.received_runs.append(payload)
        return CodecTokenPayload(
            token_ids=(0,),
            codebooks=1,
            active_mask=((True,),),
        )


class _RaisingTalker:
    """Raises on every call so failure-path tests can exercise cleanup."""

    name = "talker"

    def execute(self, payload: Any) -> CodecTokenPayload:  # type: ignore[override]
        raise RuntimeError("simulated talker failure")


# --------------------------------------------------------------------------- #
# AC #1 + #2: forced padding count is configurable and every step contributes #
# --------------------------------------------------------------------------- #


class TestThinkerForcedPadding:
    """The Thinker enters the post-EOS state and produces N forced bridges."""

    def test_thinker_runs_default_forced_padding_count(self) -> None:
        _, trace = make_pipeline().run("hello")

        thinker_run = trace[0][1]
        assert isinstance(thinker_run, ThinkerRun)
        assert thinker_run.forced_padding_count == THINKER_FORCED_PADDING_DEFAULT
        assert len(thinker_run.bridges) == THINKER_FORCED_PADDING_DEFAULT + 1

    def test_forced_padding_count_is_configurable(self) -> None:
        pipeline = make_pipeline(forced_padding_count=8)
        _, trace = pipeline.run("hello")

        thinker_run = trace[0][1]
        assert thinker_run.forced_padding_count == 8
        assert len(thinker_run.bridges) == 9

    def test_every_forced_step_emits_a_bridge_to_the_talker(self) -> None:
        """AC #2: every forced step contributes a bridge for the Talker."""
        capturing = _CapturingTalker()
        pipeline = Pipeline(
            (
                FakeThinker(forced_padding_count=6),
                capturing,
                FakeCode2Wav(),
            )
        )

        Orchestrator().submit(pipeline, "hello")

        assert len(capturing.received_runs) == 1
        bridges = capturing.received_runs[0].bridges
        # 1 visible + 6 forced = 7 bridges, all distinct payloads.
        assert len(bridges) == 7
        unique = {id(b) for b in bridges}
        assert len(unique) == 7

    def test_visible_step_is_the_first_bridge(self) -> None:
        capturing = _CapturingTalker()
        pipeline = Pipeline((FakeThinker(forced_padding_count=4), capturing, FakeCode2Wav()))

        Orchestrator().submit(pipeline, "hello omni")

        bridges = capturing.received_runs[0].bridges
        assert bridges[0].tokens.text == "hello omni"
        assert bridges[0].tokens.metadata == {}
        for forced in bridges[1:]:
            assert forced.tokens.metadata.get("forced") == "true"


# --------------------------------------------------------------------------- #
# AC #3: per-request isolation + cleanup                                      #
# --------------------------------------------------------------------------- #


class TestPerRequestIsolationAndCleanup:
    """Post-EOS state is isolated across requests and cleared on every exit."""

    def test_sequential_submissions_produce_independent_bridge_streams(self) -> None:
        """Two sequential submits with different prompts carry distinct bridges."""
        orchestrator = Orchestrator()
        capturing = _CapturingTalker()
        pipeline = Pipeline((FakeThinker(forced_padding_count=4), capturing, FakeCode2Wav()))

        orchestrator.submit(pipeline, "alpha")
        orchestrator.submit(pipeline, "beta")

        assert len(capturing.received_runs) == 2
        first, second = capturing.received_runs
        # Distinct text propagates into the visible bridge of each request.
        assert first.bridges[0].tokens.text == "alpha"
        assert second.bridges[0].tokens.text == "beta"
        # Each request owns independent bridge *objects* — no shared references.
        # (Forced bridges are padding, so values may coincide; identity must not.)
        first_bridge_ids = {id(b) for b in first.bridges}
        second_bridge_ids = {id(b) for b in second.bridges}
        assert first_bridge_ids.isdisjoint(second_bridge_ids)
        # The ThinkerRun envelopes are likewise distinct objects.
        assert first is not second

    def test_concurrent_submissions_each_get_their_own_bridges(self) -> None:
        """Concurrent submits with the same prompt still produce independent runs."""
        orchestrator = Orchestrator()
        capturing = _CapturingTalker()
        pipeline = Pipeline((FakeThinker(forced_padding_count=4), capturing, FakeCode2Wav()))

        results: list[AudioPayload] = []
        errors: list[BaseException] = []

        def submit() -> None:
            try:
                results.append(orchestrator.submit(pipeline, "concurrent").audio)
            except BaseException as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(results) == 4
        assert len(capturing.received_runs) == 4
        # Each ThinkerRun is an independent object carrying its own bridges.
        ids = {id(run) for run in capturing.received_runs}
        assert len(ids) == 4

    def test_state_is_cleared_after_successful_completion(self) -> None:
        orchestrator = Orchestrator()
        orchestrator.submit(make_pipeline(), "hello")

        assert orchestrator.active_request_count() == 0

    def test_state_is_cleared_after_mid_pipeline_failure(self) -> None:
        orchestrator = Orchestrator()
        pipeline = Pipeline((FakeThinker(), _RaisingTalker(), FakeCode2Wav()))

        with pytest.raises(RuntimeError):
            orchestrator.submit(pipeline, "hello")

        assert orchestrator.active_request_count() == 0


# --------------------------------------------------------------------------- #
# AC #4: Talker MTP active mask + audio padding token                          #
# --------------------------------------------------------------------------- #


class TestTalkerMTPMask:
    """The Talker fills inactive codebook positions with the audio padding token."""

    def test_active_mask_saturates_at_codebooks_per_frame(self) -> None:
        _, trace = make_pipeline(forced_padding_count=3, codebooks=4).run("hello")
        codec = trace[1][1]
        assert isinstance(codec, CodecTokenPayload)
        assert codec.codebooks == 4
        # 1 visible + 3 forced = 4 frames; mask is 4x4.
        assert len(codec.active_mask) == 4
        assert all(len(frame) == 4 for frame in codec.active_mask)

    def test_active_mask_pattern_is_delayed_mtp(self) -> None:
        """At frame t, codebook k is active iff k <= t (delayed activation)."""
        _, trace = make_pipeline(forced_padding_count=5, codebooks=3).run("hello")
        codec = trace[1][1]
        frames = len(codec.active_mask)

        for t in range(frames):
            for k in range(3):
                assert codec.active_mask[t][k] == (
                    k <= t
                ), f"frame={t} codebook={k}: expected {k <= t}"

    def test_inactive_codebook_positions_carry_the_audio_padding_token(self) -> None:
        _, trace = make_pipeline(forced_padding_count=5, codebooks=3).run("hello")
        codec = trace[1][1]

        for frame_idx, frame in enumerate(codec.active_mask):
            for codebook_idx, active in enumerate(frame):
                idx = frame_idx * codec.codebooks + codebook_idx
                if active:
                    assert codec.token_ids[idx] != AUDIO_PADDING_TOKEN_ID, (
                        f"active position frame={frame_idx} codebook={codebook_idx} "
                        f"must not be the padding token"
                    )
                else:
                    assert codec.token_ids[idx] == AUDIO_PADDING_TOKEN_ID, (
                        f"inactive position frame={frame_idx} codebook={codebook_idx} "
                        f"must be the padding token"
                    )

    def test_first_codebook_is_active_every_frame(self) -> None:
        """The 'primary' codebook is never padded: it predicts at every step."""
        _, trace = make_pipeline(forced_padding_count=8, codebooks=4).run("hello")
        codec = trace[1][1]

        for _frame_idx, frame in enumerate(codec.active_mask):
            assert frame[0] is True


# --------------------------------------------------------------------------- #
# AC #5 + #6: external-behavior tests, mock audio still runnable              #
# --------------------------------------------------------------------------- #


class TestPublicPipelineSeam:
    """The mock audio path stays runnable through the public pipeline seam."""

    def test_submit_returns_playable_audio_through_public_seam(self) -> None:
        result = Orchestrator().submit(make_pipeline(), "hello omni")

        assert isinstance(result.audio, AudioPayload)
        assert result.audio.sample_rate == 8_000
        assert result.audio.metadata == {"format": "pcm_s16le", "source": "fake-code2wav"}
        assert len(result.audio.samples) == 800
        assert result.stage_names == ("thinker", "talker", "code2wav")

    def test_full_request_path_observable_via_pipeline_trace(self) -> None:
        """Visible EOS precedes the full forced-padding run; downstream finishes."""
        _, trace = make_pipeline(forced_padding_count=5).run("hello")

        thinker_run, codec, audio = trace[0][1], trace[1][1], trace[2][1]
        assert isinstance(thinker_run, ThinkerRun)
        assert isinstance(codec, CodecTokenPayload)
        assert isinstance(audio, AudioPayload)
        assert len(thinker_run.bridges) == 6
        assert len(codec.active_mask) == 6

    def test_thinker_bridges_carry_per_step_distinct_hidden_states(self) -> None:
        """Every forced step has a distinct hidden state — observable, not a counter."""
        _, trace = make_pipeline(forced_padding_count=4).run("hi")
        bridges = trace[0][1].bridges

        distinct_values = {tuple(b.hidden_states.values) for b in bridges}
        # Visible bridge contributes one shape, each forced step another.
        assert len(distinct_values) == len(bridges)
