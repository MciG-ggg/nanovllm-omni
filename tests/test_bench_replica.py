"""Tests for nanovllm_omni.optim.bench.bench_replica (TK-013).

CPU path + helper functions are exercised end-to-end. The GPU path is
verified by mocking ``run_batched_generate`` so the file is fast and
CI-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from nanovllm_omni.optim.bench import bench_replica
from nanovllm_omni.optim.bench.bench_replica import (
    CSV_COLUMNS,
    DEFAULT_PROMPTS_PER_REPLICA,
    DEFAULT_REPLICAS,
    _markdown_table,
    _summarize_row,
)


def test_default_replicas_and_prompts_locked() -> None:
    """Sweep defaults locked at (1, 2, 4) and 8 prompts per replica."""
    assert DEFAULT_REPLICAS == (1, 2, 4)
    assert DEFAULT_PROMPTS_PER_REPLICA == 8


def test_summarize_row_has_all_csv_columns() -> None:
    row = _summarize_row(
        replicas=2,
        gpu="RTX 3050",
        commit="abc1234",
        batch=16,
        latencies=[200.0, 210.0, 220.0],
        throughput=10.0,
        vram_peak_mb=1800.0,
    )
    for col in CSV_COLUMNS:
        assert col in row
    assert row["replicas"] == 2
    assert row["batch"] == 16
    assert row["throughput_ops_per_sec"] == "10.000"
    assert row["latency_p50_ms"] == "210.000"
    assert row["vram_peak_mb"] == "1800.000"


def test_markdown_table_format() -> None:
    """Markdown table header + body carries all locked columns."""
    rows = [
        _summarize_row(
            replicas=1,
            gpu="RTX 3050",
            commit="abc1234",
            batch=8,
            latencies=[300.0],
            throughput=2.5,
            vram_peak_mb=1800.0,
        ),
        _summarize_row(
            replicas=2,
            gpu="RTX 3050",
            commit="abc1234",
            batch=16,
            latencies=[280.0, 290.0],
            throughput=5.0,
            vram_peak_mb=1820.0,
        ),
    ]
    table = _markdown_table(rows)
    assert "| replicas | gpu | commit | batch | throughput_ops_per_sec |" in table
    assert "| 1 | RTX 3050 |" in table
    assert "| 2 | RTX 3050 |" in table


def test_main_cpu_path_writes_csv_per_replica(tmp_path: Path) -> None:
    """On a CPU host the script still writes one CSV row per replica (zero-valued)."""
    csv_path = tmp_path / "bench.csv"
    with mock.patch("nanovllm_omni.optim.bench.bench_replica.gpu_label", return_value="cpu"):
        rc = bench_replica.main(
            [
                "--replicas",
                "1",
                "2",
                "--prompts-per-replica",
                "3",
                "--out",
                str(csv_path),
            ]
        )
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    # Header + 2 data rows (one per replica)
    assert contents[0].strip() == ",".join(CSV_COLUMNS)
    assert len(contents) == 3
    # Each data row carries the requested batch (= replicas * prompts_per_replica)
    rows = [line.split(",") for line in contents[1:]]
    assert rows[0][0] == "1"
    assert rows[1][0] == "2"
    assert rows[0][3] == "3"
    assert rows[1][3] == "6"


def test_main_gpu_path_invokes_run_batched_per_replica(tmp_path: Path) -> None:
    """On a GPU host, ``run_batched_generate`` is called once per replica
    with the locked scenario."""
    csv_path = tmp_path / "bench.csv"
    fake_bundle = object()
    fake_payloads = [object(), object()]
    fake_batched = mock.MagicMock(return_value=fake_payloads)
    fake_create_bundle = mock.MagicMock(return_value=fake_bundle)

    fake_engine_module = mock.MagicMock()
    fake_engine_module.run_batched_generate = fake_batched
    fake_model_module = mock.MagicMock()
    fake_model_module.create_bundle = fake_create_bundle

    with (
        mock.patch.dict(
            sys.modules,
            {
                "nanovllm_omni.engine.batched_runner": fake_engine_module,
                "nanovllm_omni.models.minimind_omni": fake_model_module,
            },
        ),
        mock.patch("nanovllm_omni.optim.bench.bench_replica.gpu_label", return_value="RTX 3050"),
    ):
        rc = bench_replica.main(
            [
                "--replicas",
                "1",
                "2",
                "--prompts-per-replica",
                "2",
                "--max-new-tokens",
                "8",
                "--out",
                str(csv_path),
            ]
        )
    assert rc == 0
    # Two replicas -> two calls to the batched fn
    assert fake_batched.call_count == 2
    # First call: num_replicas=1, batch=2 prompts
    first = fake_batched.call_args_list[0]
    assert first.kwargs["num_replicas"] == 1
    assert len(first.args[1]) == 2  # prompts passed positionally (bundle, prompts, ...)
    # Second call: num_replicas=2, batch=4 prompts
    second = fake_batched.call_args_list[1]
    assert second.kwargs["num_replicas"] == 2
    assert len(second.args[1]) == 4
    # Locked token budget + sampling params reached the batched fn
    assert first.kwargs["max_new_tokens"] == 8
    assert first.kwargs["temperature"] == 0.7
    assert first.kwargs["top_p"] == 0.9
    assert csv_path.exists()
    contents = csv_path.read_text().splitlines()
    assert len(contents) == 3  # header + 2 data rows
