"""Tests for nanovllm_omni.bench.bench_smolvla (Phase 1 baseline).

CPU-only path is exercised end-to-end. GPU path is TODO-wired in
bench_smolvla.py -- the locked input fixture + summary helpers are
verified here so a future wired GPU path inherits a known-good base.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from nanovllm_omni.bench import bench_smolvla
from nanovllm_omni.bench.bench_smolvla import (
    CSV_COLUMNS,
    SMOLVLA_INPUTS,
    _row,
    _summarize,
)


def test_input_fixture_locked() -> None:
    """Phase 1 input set is locked: don't drift without an ADR."""
    assert len(SMOLVLA_INPUTS) == 1
    inp = SMOLVLA_INPUTS[0]
    assert inp.id == "smolvla_01"
    # 8-dim state matches SmolVLA observation.state normalizer.
    assert len(inp.state) == 8
    assert inp.num_inference_steps == 10
    assert inp.instruction.startswith("pick up")
    # Synthetic image is 256x256x3 uint8.
    assert inp.image.shape == (256, 256, 3)


def test_summarize_two_or_more_runs() -> None:
    """_summarize returns (p50, p99, peak); needs >= 2 walls for quantiles."""
    p50, p99, peak = _summarize([100.0, 200.0, 300.0])
    assert p50 == 200.0  # median
    assert peak == 300.0
    assert 200.0 <= p99 <= 300.0


def test_summarize_single_run_returns_zero_p99() -> None:
    """One wall: p50 == walls[0], p99 == 0.0."""
    p50, p99, peak = _summarize([42.0])
    assert p50 == 42.0
    assert p99 == 0.0
    assert peak == 42.0


def test_csv_row_has_all_columns() -> None:
    """Every locked column is populated by _row."""
    inp = SMOLVLA_INPUTS[0]
    row = _row(inp, [10.0, 20.0, 30.0], "gpu-test", "deadbeef", 1500.0)
    for col in CSV_COLUMNS:
        assert col in row
    assert row["gpu"] == "gpu-test"
    assert row["commit"] == "deadbeef"
    assert row["input_id"] == inp.id
    assert row["num_inference_steps"] == inp.num_inference_steps
    assert row["n_runs"] == 3
    assert row["vram_peak_mb"] == "1500.000"


def test_main_cpu_path_writes_csv(tmp_path: Path) -> None:
    """On a CPU host the script writes a (zero-valued) CSV row per input."""
    csv_path = tmp_path / "bench.csv"
    with mock.patch("nanovllm_omni.bench.bench_smolvla.gpu_label", return_value="cpu"):
        rc = bench_smolvla.main(["--runs", "20", "--out", str(csv_path)])
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    assert contents[0].strip() == ",".join(CSV_COLUMNS)
    # Header + 1 data row (SMOLVLA_INPUTS has 1 input until LIBERO episode lands).
    assert len(contents) == 2
