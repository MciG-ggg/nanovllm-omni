"""Tests for nanovllm_omni.bench.bench_sd_turbo (Phase 1 baseline skeleton).

CPU-only path is exercised end-to-end. GPU path is TODO-wired in
bench_sd_turbo.py -- the locked input fixture + summary helpers are
verified here so a future wired GPU path inherits a known-good base.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from nanovllm_omni.bench import bench_sd_turbo
from nanovllm_omni.bench.bench_sd_turbo import (
    CSV_COLUMNS,
    SD_TURBO_INPUTS,
    _row,
    _summarize,
)


def test_input_fixture_locked() -> None:
    """Phase 1 input set is locked: don't drift the prompt without an ADR."""
    assert len(SD_TURBO_INPUTS) == 1
    inp = SD_TURBO_INPUTS[0]
    assert inp.id == "sd_turbo_01"
    assert inp.seed == 42
    # sd-turbo is fp16-only by recipe (adversarial distillation needs
    # the smaller mantissa to land in the right weight scale).
    assert inp.precision == "float16"
    assert inp.num_inference_steps == 1
    assert inp.guidance_scale == 0.0


def test_summarize_two_or_more_runs() -> None:
    """_summarize returns (p50, p99, peak); needs >= 2 walls for quantiles."""
    p50, p99, peak = _summarize([100.0, 200.0, 300.0])
    assert p50 == 200.0  # median
    assert peak == 300.0
    # p99 just needs to be in range; not asserting exact value due to
    # inclusive-method rounding differences across Python versions.
    assert 200.0 <= p99 <= 300.0


def test_summarize_single_run_returns_zero_p99() -> None:
    """One wall: p50 == walls[0], p99 == 0.0 (quantiles need >= 2 samples)."""
    p50, p99, peak = _summarize([42.0])
    assert p50 == 42.0
    assert p99 == 0.0
    assert peak == 42.0


def test_csv_row_has_all_columns() -> None:
    """Every locked column is populated by _row."""
    inp = SD_TURBO_INPUTS[0]
    row = _row(inp, [10.0, 20.0, 30.0], "gpu-test", "deadbeef", 1500.0)
    for col in CSV_COLUMNS:
        assert col in row
    assert row["gpu"] == "gpu-test"
    assert row["commit"] == "deadbeef"
    assert row["input_id"] == inp.id
    assert row["seed"] == 42
    assert row["n_runs"] == 3
    assert row["vram_peak_mb"] == "1500.000"


def test_main_cpu_path_writes_csv(tmp_path: Path) -> None:
    """On a CPU host the script still writes a (zero-valued) CSV row."""
    csv_path = tmp_path / "bench.csv"
    with mock.patch("nanovllm_omni.bench.bench_sd_turbo.gpu_label", return_value="cpu"):
        rc = bench_sd_turbo.main(["--runs", "5", "--out", str(csv_path)])
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    assert contents[0].strip() == ",".join(CSV_COLUMNS)
    # Header + one data row (SD_TURBO_INPUTS has 1 input).
    assert len(contents) == 2
