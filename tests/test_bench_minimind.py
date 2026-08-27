"""Tests for nanovllm_omni.optim.bench.bench_minimind (TK-012).

The script's CPU-only path is exercised end-to-end (no real model load,
no GPU). The GPU path is verified by mocking ``run_n`` + the bundle
factory so the file is small, fast, and CI-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from nanovllm_omni.optim.bench import bench_minimind
from nanovllm_omni.optim.bench.bench_minimind import (
    CSV_COLUMNS,
    DEFAULT_N,
    FIXED_MAX_TOKENS,
    FIXED_PROMPT,
    _markdown_table,
    _row_from_summary,
    _summarize,
)


@dataclass
class _FakeRunResult:
    """Subset of ``RunResult`` that ``_summarize`` reads."""

    times: _FakeStageTimes
    frames: int
    seed: int = 0
    prompt_id: str = "fake"
    run_idx: int = 0
    vram_peak_mb: float = 0.0
    audio_bytes: bytes = b""


@dataclass
class _FakeStageTimes:
    tokenize_ms: float = 0.0
    generate_ms: float = 0.0
    decode_ms: float = 0.0
    wav_ms: float = 0.0
    generate_cuda_ms: float = 0.0
    decode_cuda_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tokenize_ms + self.generate_ms + self.decode_ms + self.wav_ms


def test_fixed_prompt_and_token_budget_locked() -> None:
    """The bench scenario is locked: do not drift the prompt or token count."""
    assert FIXED_PROMPT.id == "ttfb_fixed"
    assert FIXED_PROMPT.text == "Hello, how are you?"
    assert FIXED_MAX_TOKENS == 120  # 120 / 12.5 Hz = 9.6 s audio budget; cap at 5 s
    assert DEFAULT_N == 50


def test_summarize_empty_returns_zeros() -> None:
    """``_summarize([])`` -> all zeros (no request, no timings)."""
    assert _summarize([]) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_summarize_computes_ttfb_and_total() -> None:
    """TTFB = tokenize + per-frame generate; total = StageTimes.total_ms;
    VRAM peak is the max across the run."""
    fake_results = [
        _FakeRunResult(
            times=_FakeStageTimes(tokenize_ms=10.0, generate_ms=240.0),
            frames=120,
            vram_peak_mb=1800.0,
        ),
        _FakeRunResult(
            times=_FakeStageTimes(tokenize_ms=12.0, generate_ms=300.0),
            frames=120,
            vram_peak_mb=1820.0,
        ),
    ]
    ttfb_p50, ttfb_p99, total_p50, total_p99, vram_peak = _summarize(fake_results)
    # TTFB samples: [10 + 240/120 = 12.0, 12 + 300/120 = 14.5]
    assert ttfb_p50 == 13.25  # median of [12.0, 14.5]
    # total: 250.0, 312.0
    assert total_p50 == 281.0  # median
    assert vram_peak == 1820.0


def test_csv_row_from_summary_has_all_columns() -> None:
    """The CSV row from ``_row_from_summary`` carries every locked column."""
    row = _row_from_summary("gpu-test", "deadbeef", 50, (12.0, 14.5, 281.0, 312.0, 1820.0))
    for col in CSV_COLUMNS:
        assert col in row
    assert row["gpu"] == "gpu-test"
    assert row["commit"] == "deadbeef"
    assert row["n_requests"] == 50
    assert row["ttfb_p50_ms"] == "12.000"


def test_markdown_table_format() -> None:
    """Markdown table header + body carries the locked columns."""
    row = _row_from_summary(
        "RTX 3050 (4 GB, driver 123.4)", "abc1234", 50, (12, 14, 281, 312, 1820)
    )
    table = _markdown_table(row)
    assert "| gpu | commit | n | ttfb_p50_ms |" in table
    assert "RTX 3050" in table
    assert "abc1234" in table


def test_main_cpu_path_writes_csv(tmp_path: Path) -> None:
    """On a CPU host the script still writes a (zero-valued) CSV row."""
    csv_path = tmp_path / "bench.csv"
    with mock.patch("nanovllm_omni.optim.bench.bench_minimind.gpu_label", return_value="cpu"):
        rc = bench_minimind.main(["--n", "5", "--out", str(csv_path)])
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    assert contents[0].strip() == ",".join(CSV_COLUMNS)
    # One data row + header
    assert len(contents) == 2


def test_main_gpu_path_uses_run_n(tmp_path: Path) -> None:
    """On a GPU host, ``run_n`` is called once with the locked scenario."""
    csv_path = tmp_path / "bench.csv"
    fake_bundle = object()
    # >= 2 results: `_summarize` feeds statistics.quantiles(n=100), which on
    # Python <= 3.12 raises "must have at least two data points" on a single
    # sample (3.13 loosened method="inclusive"). Two rows keep the test
    # green across the CI matrix (3.11/3.12).
    fake_results = [
        _FakeRunResult(
            times=_FakeStageTimes(tokenize_ms=10.0, generate_ms=240.0),
            frames=120,
            vram_peak_mb=1800.0,
        ),
        _FakeRunResult(
            times=_FakeStageTimes(tokenize_ms=12.0, generate_ms=300.0),
            frames=120,
            vram_peak_mb=1820.0,
        ),
    ]
    fake_create_bundle = mock.MagicMock(return_value=fake_bundle)
    fake_run_n = mock.MagicMock(return_value=fake_results)
    import sys

    fake_model_module = mock.MagicMock()
    fake_model_module.create_bundle = fake_create_bundle

    # run_n is imported at module top (`from .runner import run_n`), so we
    # patch the name on bench_minimind directly; the fake model module goes
    # into sys.modules (scoped so it is restored on exit -- it leaked before
    # and poisoned every later lazy `minimind_omni` import in the suite).
    with (
        mock.patch.object(bench_minimind, "run_n", new=fake_run_n),
        mock.patch.dict(
            sys.modules,
            {
                "nanovllm_omni.models.minimind_omni": fake_model_module,
            },
        ),
        mock.patch("nanovllm_omni.optim.bench.bench_minimind.gpu_label", return_value="RTX 3050"),
    ):
        rc = bench_minimind.main(["--n", "3", "--warmup", "1", "--out", str(csv_path)])
    assert rc == 0
    # Locked scenario must reach run_n intact.
    assert fake_run_n.call_count == 1
    kwargs = fake_run_n.call_args.kwargs
    assert kwargs["n"] == 3
    assert kwargs["warmup"] == 1
    assert kwargs["max_tokens"] == FIXED_MAX_TOKENS
    assert kwargs["seed"] == 42
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    # The prompt arg is the fixed one (positional).
    assert fake_run_n.call_args.args[1].text == FIXED_PROMPT.text
    assert csv_path.exists()
