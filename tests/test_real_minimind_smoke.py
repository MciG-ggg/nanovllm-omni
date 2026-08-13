"""Manual real-weight smoke test for GitHub issue #4.

Skipped by default in CI. Run with::

    pytest -m smoke tests/test_real_minimind_smoke.py

Requires ``torch``, ``transformers``, ``huggingface_hub``, ``soundfile``
(installable via ``pip install nanovllm-omni[minimind]``), network access
to the Hugging Face Hub, and roughly 1 GB of disk for the downloaded
``jingyaogong/minimind-3o`` checkpoint. Verifies AC #2-#7 against the
public pipeline seam.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.models import load_minimind_omni_bundle
from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    AudioPayload,
    CodecTokenPayload,
    ThinkerRun,
)
from nanovllm_omni.runtime import Orchestrator, Pipeline
from nanovllm_omni.stage import Stage

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        True,
        reason="manual real-weight smoke test; download + load skipped in CI",
    ),
]


def _bundle():
    return load_minimind_omni_bundle()


def _pipeline(bundle):
    return Pipeline((bundle.thinker, bundle.talker, bundle.code2wav))


def test_real_minimind_pipeline_runs_one_request() -> None:
    """AC #6: real-weight smoke test executes one request, checks output shape."""
    result = Orchestrator().submit(_pipeline(_bundle()), "Tell me a short joke.")

    assert isinstance(result.audio, AudioPayload)
    assert result.audio.sample_rate == 24_000
    assert result.audio.metadata == {
        "format": "pcm_s16le",
        "source": "minimind-omni",
    }
    assert len(result.audio.samples) > 0
    assert result.stage_names == ("thinker", "talker", "code2wav")


def test_real_minimind_thinker_emits_thinker_run() -> None:
    """AC #2 + #3: Thinker stage emits a typed ThinkerRun with one bridge."""
    _, trace = _pipeline(_bundle()).run("Hello.")

    thinker_output = trace[0][1]
    assert isinstance(thinker_output, ThinkerRun)
    assert thinker_output.forced_padding_count == 0
    assert len(thinker_output.bridges) == 1
    assert thinker_output.bridges[0].tokens.text  # non-empty visible text


def test_real_minimind_talker_emits_mtp_mask() -> None:
    """AC #3 + #4: Talker stage applies the MTP delayed active mask."""
    _, trace = _pipeline(_bundle()).run("Hello.")

    codec = trace[1][1]
    assert isinstance(codec, CodecTokenPayload)
    assert codec.codebooks == 8
    assert codec.sample_rate == 24_000

    for t, frame in enumerate(codec.active_mask):
        for k, active in enumerate(frame):
            assert active == (k <= t), f"frame={t} codebook={k}: expected {k <= t}"

    # Inactive positions carry the audio padding token.
    for t, frame in enumerate(codec.active_mask):
        for k, active in enumerate(frame):
            idx = t * codec.codebooks + k
            if not active:
                assert codec.token_ids[idx] == AUDIO_PADDING_TOKEN_ID


def test_real_minimind_audio_decodes_to_wav_bytes() -> None:
    """AC #4: Code2Wav decodes Mimi codec tokens into a playable WAV payload."""
    result = Orchestrator().submit(_pipeline(_bundle()), "Hello.")

    wav = result.audio.wav_bytes()
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_real_minimind_failure_cleanup_releases_request() -> None:
    """AC #7: failure cleanup leaves no in-flight request state."""
    orchestrator = Orchestrator()

    class _Boom(Stage[object, object]):
        name = "boom"

        def execute(self, _payload: object) -> object:
            raise RuntimeError("simulated mid-pipeline failure")

    pipeline = Pipeline((_bundle().thinker, _Boom(), _bundle().code2wav))

    with pytest.raises(RuntimeError):
        orchestrator.submit(pipeline, "Hello.")

    assert orchestrator.active_request_count() == 0
