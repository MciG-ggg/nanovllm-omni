"""Tests for nanovllm_omni.bench.bench_smolvlm (Phase 1 baseline).

CPU-only path is exercised end-to-end. GPU path is TODO-wired in
bench_smolvlm.py -- the locked input fixture + summary helpers are
verified here so a future wired GPU path inherits a known-good base.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from nanovllm_omni.bench import bench_smolvlm
from nanovllm_omni.bench.bench_smolvlm import (
    CSV_COLUMNS,
    SMOLVLM_INPUTS,
    _row,
    _summarize,
)


def test_input_fixture_locked() -> None:
    """Phase 1 input set is locked: don't drift without an ADR."""
    assert len(SMOLVLM_INPUTS) == 3
    ids = [inp.id for inp in SMOLVLM_INPUTS]
    assert ids == ["smolvlm_short", "smolvlm_medium", "smolvlm_long"]
    # max_new_tokens locked at 64 across all 3 inputs so the profile
    # isolates prompt-length effect, not decode-budget effect.
    for inp in SMOLVLM_INPUTS:
        assert inp.max_new_tokens == 64
        assert inp.temperature == 0.0  # greedy


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
    inp = SMOLVLM_INPUTS[0]
    row = _row(inp, [10.0, 20.0, 30.0], "gpu-test", "deadbeef", 1500.0)
    for col in CSV_COLUMNS:
        assert col in row
    assert row["gpu"] == "gpu-test"
    assert row["commit"] == "deadbeef"
    assert row["input_id"] == inp.id
    assert row["max_new_tokens"] == inp.max_new_tokens
    assert row["n_runs"] == 3
    assert row["vram_peak_mb"] == "1500.000"


def test_main_cpu_path_writes_csv(tmp_path: Path) -> None:
    """On a CPU host the script writes a (zero-valued) CSV row per input."""
    csv_path = tmp_path / "bench.csv"
    with mock.patch("nanovllm_omni.bench.bench_smolvlm.gpu_label", return_value="cpu"):
        rc = bench_smolvlm.main(["--runs", "10", "--out", str(csv_path)])
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    assert contents[0].strip() == ",".join(CSV_COLUMNS)
    # Header + 3 data rows (SMOLVLM_INPUTS has 3 inputs).
    assert len(contents) == 4
