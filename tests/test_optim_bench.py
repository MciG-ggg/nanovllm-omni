"""Tests for the ``nanovllm_omni.optim.bench`` harness (TK-011).

3 unit tests + 3 @pytest.mark.smoke tests, per GitHub issue #23.
Unit tests run on CPU with a stub bundle so CI does not need the
MiniMind-O weights.
"""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes (CPU-only; mirror MiniMind-O's surface enough for the helpers)
# ---------------------------------------------------------------------------


@dataclass
class _FakeTensor:
    """Stand-in for the float audio tensor returned by ``mimi.decode``."""

    data: list[float]

    def squeeze(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self):  # noqa: D401 - matches torch API
        import numpy as np

        return np.asarray(self.data, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.data),)


class _FakeAudioOut:
    def __init__(self, samples: list[float]) -> None:
        self.audio_values = _FakeTensor(samples)


class _FakeMimi:
    def decode(self, codes: Any) -> _FakeAudioOut:  # noqa: D401
        return _FakeAudioOut([0.1, -0.2, 0.3, -0.4] * 24)


class _FakeModel:
    """Stub that mimics MiniMind's streaming generate (text_ids, audio_frame)."""

    eos_token_id = 0

    def generate(self, input_ids: Any, eos_token_id: Any, **kwargs: Any):  # noqa: D401
        yield ([1, 2, 3], [10, 11, 12, 13, 14, 15, 16, 17])
        yield ([1, 2, 3, 4], [18, 19, 20, 21, 22, 23, 24, 25])


class _TokenizerOutput(dict):
    """Dict with a ``.data`` attribute that mirrors HuggingFace's BatchEncoding."""

    @property
    def data(self) -> _TokenizerOutput:
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        open_thinking: bool = False,
    ) -> str:
        # Accept either a messages list (preferred) or a pre-templated string.
        if isinstance(messages, list):
            return "\n".join(m["content"] for m in messages)
        return str(messages)

    def __call__(self, text: str) -> _TokenizerOutput:
        return _TokenizerOutput(input_ids=[1, 2, 3, 4, 5])


@dataclass
class _FakeBundle:
    model: Any
    tokenizer: Any
    mimi: Any
    device: str = "cpu"
    model_id: str = "fake/model"


def _bundle() -> _FakeBundle:
    return _FakeBundle(model=_FakeModel(), tokenizer=_FakeTokenizer(), mimi=_FakeMimi())


def _short_prompt():
    from nanovllm_omni.optim.bench import BenchPrompt

    return BenchPrompt(id="short_test", text="你好。")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_stage_times_non_negative():
    """Each ``StageTimes`` field is ``>= 0`` after one ``run_one`` call."""
    from nanovllm_omni.optim.bench import run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    assert r.times.tokenize_ms >= 0
    assert r.times.generate_ms >= 0
    assert r.times.decode_ms >= 0
    assert r.times.wav_ms >= 0


def test_total_equals_sum():
    """``total_ms == sum(tokenize, generate, decode, wav)`` to within 0.01 ms."""
    from nanovllm_omni.optim.bench import run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    expected = r.times.tokenize_ms + r.times.generate_ms + r.times.decode_ms + r.times.wav_ms
    assert r.times.total_ms == pytest.approx(expected, abs=0.01)


def test_run_one_returns_valid_wav():
    """``run_one`` produces an ``AudioPayload`` whose ``wav_bytes()`` round-trips."""
    from nanovllm_omni.optim.bench import run_one
    from nanovllm_omni.outputs import AudioPayload

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    assert r.audio_bytes[:4] == b"RIFF"
    assert r.audio_bytes[8:12] == b"WAVE"
    payload = AudioPayload(data=r.audio_bytes, sample_rate=24000)
    wav = payload.wav_bytes()
    with wave.open(io.BytesIO(wav), "rb") as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == 24000
        assert fh.getnframes() > 0


# ---------------------------------------------------------------------------
# Smoke tests (require GPU + real model weights; skipped with ``-m "not smoke"``)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_seed_determinism():
    """Two ``run_one`` calls with the same ``(prompt, seed)`` produce identical audio bytes."""
    from nanovllm_omni.models.minimind_omni import create_bundle
    from nanovllm_omni.optim.bench import BenchPrompt, run_one

    bundle = create_bundle(
        model_id="/home/mcig/minimind-3o",
        mimi_model_id="/home/mcig/mimi",
        device="cuda",
    )
    prompt = BenchPrompt(id="det_test", text="你好。")

    r1 = run_one(bundle, prompt, seed=42, max_tokens=8, open_thinking=False)
    r2 = run_one(bundle, prompt, seed=42, max_tokens=8, open_thinking=False)
    assert r1.audio_bytes == r2.audio_bytes


@pytest.mark.smoke
def test_prompts_complete():
    """All six ``BENCH_PROMPTS`` complete one ``run_one`` without raising."""
    from nanovllm_omni.models.minimind_omni import create_bundle
    from nanovllm_omni.optim.bench import BENCH_PROMPTS, run_one

    bundle = create_bundle(
        model_id="/home/mcig/minimind-3o",
        mimi_model_id="/home/mcig/mimi",
        device="cuda",
    )
    for p in BENCH_PROMPTS:
        r = run_one(bundle, p, max_tokens=4)
        assert r.audio_bytes[:4] == b"RIFF"
        assert r.times.total_ms > 0


@pytest.mark.smoke
def test_timer_envelope():
    """Sum of stage timers is within 5% of a single ``perf_counter`` envelope."""
    from nanovllm_omni.models.minimind_omni import create_bundle
    from nanovllm_omni.optim.bench import BenchPrompt, run_one

    bundle = create_bundle(
        model_id="/home/mcig/minimind-3o",
        mimi_model_id="/home/mcig/mimi",
        device="cuda",
    )
    prompt = BenchPrompt(id="env_test", text="hello")

    t0 = time.perf_counter()
    r = run_one(bundle, prompt, max_tokens=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sum_ms = r.times.total_ms
    assert sum_ms > 0
    # Sum of per-stage timers should be <= the wall-clock envelope and
    # within 5% below it (small overhead is allowed for the perf_counter
    # call sequence itself).
    ratio = sum_ms / elapsed_ms
    assert 0.95 <= ratio <= 1.0, f"timer envelope violated: ratio={ratio:.3f}"
