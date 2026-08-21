"""Tests for the ``nanovllm_omni.optim.bench`` harness (TK-011).

All unit tests run on CPU with a stub bundle so CI does not need the
MiniMind-O weights. The one GPU-required test is marked ``smoke`` and
skipped by the AGENTS.md CI invocation ``python -m pytest -m "not smoke"``.
"""

from __future__ import annotations

import csv
import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes
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
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        open_thinking: bool = False,
    ) -> str:
        return messages[0]["content"]

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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_stage_times_total_sums_four_parts():
    from nanovllm_omni.optim.bench import StageTimes

    s = StageTimes(tokenize=0.10, generate=2.50, decode=0.30, wav=0.02)
    assert s.total == pytest.approx(2.92)
    d = s.as_dict()
    assert set(d) == {"tokenize_ms", "generate_ms", "decode_ms", "wav_ms", "total_ms"}
    assert d["total_ms"] == pytest.approx(2920.0)


def test_run_one_returns_run_result_with_valid_wav():
    from nanovllm_omni.optim.bench import RunResult, run_one

    r = run_one(_bundle(), "你好。", max_tokens=4)
    assert isinstance(r, RunResult)
    assert r.stages.total > 0
    assert r.wav_bytes[:4] == b"RIFF"
    assert r.wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(r.wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


def test_run_n_returns_n_times_prompts_results_with_warmup_ignored():
    from nanovllm_omni.optim.bench import run_n

    prompts = ("a", "b")
    results = run_n(_bundle(), prompts, n=3, warmup=1, max_tokens=2)
    assert len(results) == 3 * len(prompts)
    seen_prompts = {r.prompt for r in results}
    assert seen_prompts == set(prompts)
    assert all(r.wav_bytes[:4] == b"RIFF" for r in results)


def test_markdown_table_emits_header_and_rows():
    from nanovllm_omni.optim.bench import markdown_table

    rows = [
        {
            "prompt": "hi",
            "tokenize_ms": 1.0,
            "generate_ms": 100.0,
            "decode_ms": 5.0,
            "wav_ms": 0.5,
            "total_ms": 106.5,
            "n_tokens": 16,
            "n_samples": 96,
            "max_mem_bytes": 0,
            "wav_bytes": 192,
        },
    ]
    out = markdown_table(rows)
    lines = out.strip().split("\n")
    assert lines[0].startswith("| prompt |")
    assert "---" in lines[1]
    assert "| hi |" in lines[2]
    assert "| 100.00 |" in lines[2]


def test_write_csv_creates_file_with_expected_columns(tmp_path: Path):
    from nanovllm_omni.optim.bench import write_csv

    rows = [
        {
            "prompt": "p",
            "tokenize_ms": 1.0,
            "generate_ms": 10.0,
            "decode_ms": 1.0,
            "wav_ms": 0.1,
            "total_ms": 12.1,
            "n_tokens": 8,
            "n_samples": 96,
            "max_mem_bytes": 0,
            "wav_bytes": 192,
        },
    ]
    out = write_csv(rows, tmp_path / "subdir" / "out.csv")
    assert out.exists()
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header[0] == "prompt"
        assert "total_ms" in header
        body = next(reader)
        assert body[0] == "p"
        assert body[header.index("wav_bytes")] == "192"


# ---------------------------------------------------------------------------
# Smoke (requires GPU + real model weights; skipped with ``-m "not smoke"``).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bench_cli_time_smoke_runs_one_iteration(tmp_path: Path):
    """Drive ``python -m nanovllm_omni.optim.bench time --runs=1`` end-to-end.

    Loads real MiniMind-O weights; skipped by default via ``-m "not smoke"``.
    """
    import subprocess
    import sys

    out_csv = tmp_path / "smoke.csv"
    cmd = [
        sys.executable,
        "-m",
        "nanovllm_omni.optim.bench",
        "time",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--prompts",
        "0",
        "--max-tokens",
        "4",
        "--out",
        str(out_csv),
        "--device",
        "cuda",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["total_ms"]) > 0
    assert int(rows[0]["wav_bytes"]) > 0
