"""Tests for the ``nanovllm_omni.engine.bench`` harness (TK-011).

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

# Skip the whole file when torch is missing. AGENTS.md scopes the dev
# extra to pytest+ruff+black; heavy ML deps (torch, transformers, ...)
# live in the ``minimind`` extra and are NOT installed in CI. Local
# runs that exercise the bench / compile / cuda-graph paths should
# `uv pip install torch` (or ``uv sync --group minimind``) first.
torch = pytest.importorskip("torch")

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
    from nanovllm_omni.engine.bench import BenchPrompt

    return BenchPrompt(id="short_test", text="你好。")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_stage_times_non_negative():
    """Each ``StageTimes`` field is ``>= 0`` after one ``run_one`` call."""
    from nanovllm_omni.engine.bench import run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    assert r.times.tokenize_ms >= 0
    assert r.times.generate_ms >= 0
    assert r.times.decode_ms >= 0
    assert r.times.wav_ms >= 0


def test_total_equals_sum():
    """``total_ms == sum(tokenize, generate, decode, wav)`` to within 0.01 ms."""
    from nanovllm_omni.engine.bench import run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    expected = r.times.tokenize_ms + r.times.generate_ms + r.times.decode_ms + r.times.wav_ms
    assert r.times.total_ms == pytest.approx(expected, abs=0.01)


def test_run_one_returns_valid_wav():
    """``run_one`` produces an ``AudioPayload`` whose ``wav_bytes()`` round-trips."""
    from nanovllm_omni.engine.bench import run_one
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


def test_run_one_full_times_e2e_and_derives_frames():
    """``run_one_full`` routes through ``Omni.generate`` and derives frames from the WAV.

    Uses a stub ``omni`` so CI does not need the MiniMind-O weights.
    """
    from nanovllm_omni.engine.bench import run_one_full
    from nanovllm_omni.outputs import AudioPayload, MultimodalPayload

    # 2400 16-bit samples @ 24 kHz -> 0.1 s (4800 bytes raw PCM; header added
    # by AudioPayload.wav_bytes). 4800 / 3840 bytes-per-frame = 1 Mimi frame.
    raw_pcm = b"\x00\x00" * 2400
    wav = AudioPayload(data=raw_pcm, sample_rate=24000).wav_bytes()

    class _FakeOmni:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _text, _sampling_params):
            self.calls += 1
            out = type(
                "Out",
                (),
                {
                    "multimodal_output": MultimodalPayload.from_dict(
                        {"audio": AudioPayload(data=wav, sample_rate=24000)}
                    )
                },
            )()
            return [out]

    omni = _FakeOmni()
    r = run_one_full(omni, _short_prompt(), max_tokens=4)
    assert omni.calls == 1
    assert r.times.generate_ms > 0.0
    assert r.times.total_ms == r.times.generate_ms  # E2E is the only timed stage
    assert r.audio_bytes[:4] == b"RIFF"
    # 2400 samples / 1920 samples-per-frame = 1 frame (Mimi 24 kHz mono).
    assert r.frames >= 1


def test_run_one_full_records_per_stage_breakdown():
    """``run_one_full`` reads ``PipelineRunner._last_stage_timings`` into ``stage_ms``.

    The fake omni carries an executor -> runner chain that exposes a
    populated ``_last_stage_timings``; ``run_one_full`` must extract it,
    map ``thinker`` onto ``generate_ms`` and ``code2wav`` onto
    ``decode_ms``, and surface the full breakdown via ``stage_ms``.
    """
    import json

    from nanovllm_omni.engine.bench import run_one_full
    from nanovllm_omni.outputs import AudioPayload, MultimodalPayload

    raw_pcm = b"\x00\x00" * 2400
    wav = AudioPayload(data=raw_pcm, sample_rate=24000).wav_bytes()

    class _FakeRunner:
        _last_stage_timings: list[tuple[str, float]] = [
            ("thinker", 100.5),
            ("talker", 200.25),
            ("code2wav", 50.125),
        ]

    class _FakeExecutor:
        _runner = _FakeRunner()

    class _FakeOmni:
        def __init__(self) -> None:
            self._executor = _FakeExecutor()

        def generate(self, _text, _sampling_params):
            out = type(
                "Out",
                (),
                {
                    "multimodal_output": MultimodalPayload.from_dict(
                        {"audio": AudioPayload(data=wav, sample_rate=24000)}
                    )
                },
            )()
            return [out]

    omni = _FakeOmni()
    r = run_one_full(omni, _short_prompt(), max_tokens=4)

    # Full breakdown surfaced as a dict.
    assert r.stage_ms == {"thinker": 100.5, "talker": 200.25, "code2wav": 50.125}
    # Per-stage mapping onto existing StageTimes fields.
    assert r.times.generate_ms == 100.5
    assert r.times.decode_ms == 50.125
    assert r.times.wav_ms == 0.0
    # CSV row carries the new column; round-trips through json.
    row = r.as_csv_row()
    assert "stage_ms" in row
    assert row["stage_ms"] == json.dumps(
        {"thinker": 100.5, "talker": 200.25, "code2wav": 50.125}, sort_keys=True
    )
    # Existing columns must still be present and parseable (no regression).
    for col in (
        "tokenize_ms",
        "generate_ms",
        "decode_ms",
        "wav_ms",
        "total_ms",
        "frames",
        "vram_mb",
        "generate_cuda_ms",
        "decode_cuda_ms",
        "cpu_dispatch_ms",
        "generate_per_step_ms",
    ):
        assert col in row


def test_run_one_full_without_executor_keeps_wall_clock_generate():
    """When the stub has no executor, ``run_one_full`` keeps the wall-clock
    ``generate_ms`` (no per-stage mapping) and ``stage_ms`` stays None.
    Guards the existing ``test_run_one_full_times_e2e_and_derives_frames``
    contract.
    """
    from nanovllm_omni.engine.bench import run_one_full
    from nanovllm_omni.outputs import AudioPayload, MultimodalPayload

    raw_pcm = b"\x00\x00" * 2400
    wav = AudioPayload(data=raw_pcm, sample_rate=24000).wav_bytes()

    class _BareFakeOmni:
        def generate(self, _text, _sampling_params):
            out = type(
                "Out",
                (),
                {
                    "multimodal_output": MultimodalPayload.from_dict(
                        {"audio": AudioPayload(data=wav, sample_rate=24000)}
                    )
                },
            )()
            return [out]

    r = run_one_full(_BareFakeOmni(), _short_prompt(), max_tokens=4)
    assert r.stage_ms is None
    assert r.times.total_ms == r.times.generate_ms
    assert r.as_csv_row()["stage_ms"] == ""


def test_stage_times_has_cuda_fields_and_overhead():
    """StageTimes tracks per-stage GPU time and derives CPU dispatch overhead."""
    from nanovllm_omni.engine.bench import StageTimes, run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    # On the CPU stub bundle the cuda fields are 0 (CUDA unavailable).
    assert r.times.generate_cuda_ms >= 0
    assert r.times.decode_cuda_ms >= 0
    # cpu_dispatch_ms is wall minus cuda; must be non-negative.
    assert r.times.cpu_dispatch_ms >= 0

    # Direct construction: wall > cuda -> positive dispatch overhead.
    s = StageTimes(
        tokenize_ms=1.0,
        generate_ms=100.0,
        decode_ms=20.0,
        wav_ms=0.5,
        generate_cuda_ms=90.0,
        decode_cuda_ms=15.0,
    )
    assert s.cpu_dispatch_ms == pytest.approx(15.0)
    assert s.total_cuda_ms == pytest.approx(105.0)


def test_run_result_as_csv_row_has_detail_columns():
    """RunResult.as_csv_row emits the four per-stage detail columns."""
    from nanovllm_omni.engine.bench import run_one

    r = run_one(_bundle(), _short_prompt(), max_tokens=4)
    row = r.as_csv_row()
    # CUDA *can* be available on the host even with a CPU stub bundle; we just
    # assert the fields are present and parseable. On a real GPU the values
    # become meaningful (cuda_ms > 0).
    for col in ("generate_cuda_ms", "decode_cuda_ms", "cpu_dispatch_ms", "generate_per_step_ms"):
        assert col in row
        assert float(row[col]) >= 0
    # Per-step is wall / frames; we got 2 frames from the stub.
    assert float(row["generate_per_step_ms"]) > 0


def test_parse_kineto_trace_groups_kernels_under_stage_events():
    """parse_kineto_trace returns one StageProfile per record_function stage."""
    import json
    import tempfile

    from nanovllm_omni.engine.bench.trace import parse_kineto_trace

    events = [
        # Stage events (user_annotation, ph=X)
        {
            "name": "generate",
            "ph": "X",
            "cat": "user_annotation",
            "ts": 0,
            "dur": 1000,
            "tid": 1,
            "pid": 0,
            "args": {"id": 1},
        },
        {
            "name": "decode",
            "ph": "X",
            "cat": "user_annotation",
            "ts": 1100,
            "dur": 200,
            "tid": 1,
            "pid": 0,
            "args": {"id": 2},
        },
        # Children of generate (CUDA kernels).
        {
            "name": "aten::addmm",
            "ph": "X",
            "cat": "kernel",
            "ts": 10,
            "dur": 100,
            "tid": 0,
            "pid": 1,
            "args": {"kernel": "sgemm", "grid": [128, 1, 1], "block": [256, 1, 1]},
        },
        {
            "name": "aten::addmm",
            "ph": "X",
            "cat": "kernel",
            "ts": 200,
            "dur": 200,
            "tid": 0,
            "pid": 1,
            "args": {"kernel": "sgemm"},
        },
        {
            "name": "aten::softmax",
            "ph": "X",
            "cat": "kernel",
            "ts": 500,
            "dur": 50,
            "tid": 0,
            "pid": 1,
            "args": {"kernel": "softmax"},
        },
        # generate.step sub-events: 3 iterations.
        {
            "name": "generate.step",
            "ph": "X",
            "cat": "user_annotation",
            "ts": 20,
            "dur": 90,
            "tid": 1,
            "pid": 0,
            "args": {"id": 10},
        },
        {
            "name": "generate.step",
            "ph": "X",
            "cat": "user_annotation",
            "ts": 220,
            "dur": 180,
            "tid": 1,
            "pid": 0,
            "args": {"id": 11},
        },
        {
            "name": "generate.step",
            "ph": "X",
            "cat": "user_annotation",
            "ts": 510,
            "dur": 40,
            "tid": 1,
            "pid": 0,
            "args": {"id": 12},
        },
        # Child of decode (one kernel).
        {
            "name": "aten::conv2d",
            "ph": "X",
            "cat": "kernel",
            "ts": 1150,
            "dur": 80,
            "tid": 0,
            "pid": 1,
            "args": {"kernel": "conv2d"},
        },
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump({"traceEvents": events}, fh)
        path = fh.name

    profile = parse_kineto_trace(path)
    assert len(profile.stages) == 2

    gen = profile.by_stage("generate")
    assert gen is not None
    assert gen.wall_us == 1000
    assert gen.num_steps == 3
    assert gen.kernel_count == 3
    assert gen.total_kernel_us == 350
    top = gen.top_kernels[0]
    assert top.name == "aten::addmm"
    assert top.total_us == 300
    assert top.count == 2

    dec = profile.by_stage("decode")
    assert dec is not None
    assert dec.wall_us == 200
    assert dec.kernel_count == 1
    assert dec.num_steps == 0


def test_trace_profile_markdown_renders_table():
    from nanovllm_omni.engine.bench.trace import (
        KernelStat,
        StageProfile,
        TraceProfile,
        trace_profile_markdown,
    )

    tp = TraceProfile(
        stages=(
            StageProfile(
                name="generate",
                wall_us=1000.0,
                kernel_count=3,
                total_kernel_us=350.0,
                top_kernels=(
                    KernelStat(name="aten::addmm", total_us=300.0, count=2),
                    KernelStat(name="aten::softmax", total_us=50.0, count=1),
                ),
                num_steps=3,
            ),
            StageProfile(
                name="decode",
                wall_us=200.0,
                kernel_count=1,
                total_kernel_us=80.0,
                top_kernels=(KernelStat(name="aten::conv2d", total_us=80.0, count=1),),
            ),
        )
    )
    md = trace_profile_markdown(tp)
    assert "| stage |" in md
    assert "| generate |" in md
    assert "| decode |" in md
    assert "aten::addmm" in md


# ---------------------------------------------------------------------------
# Smoke tests (require GPU + real model weights; skipped with ``-m "not smoke"``)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_seed_determinism():
    """Two ``run_one`` calls with the same ``(prompt, seed)`` produce identical audio bytes."""
    from nanovllm_omni.engine.bench import BenchPrompt, run_one
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    bundle = load_minimind_omni_bundle(
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
    from nanovllm_omni.engine.bench import BENCH_PROMPTS, run_one
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    bundle = load_minimind_omni_bundle(
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
    from nanovllm_omni.engine.bench import BenchPrompt, run_one
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    bundle = load_minimind_omni_bundle(
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
