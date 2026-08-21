"""Benchmark harness for the four MiniMind-O stages (TK-011).

Public surface used by ``tests/test_optim_bench.py`` and the
``python -m nanovllm_omni.optim.bench`` CLI:

* :data:`BENCH_PROMPTS` / :class:`BenchPrompt` -- the fixed prompt dataclass + set.
* :class:`StageTimes` / :class:`RunResult` -- timing and result shapes.
* :func:`run_one` / :func:`run_n` -- drive the four helpers end-to-end.
* :func:`write_csv` / :func:`markdown_table` -- spec-compliant result reporters.
* :func:`markdown_table_detail` -- extended table with per-stage GPU/CPU split.
* :class:`TraceProfile` / :class:`StageProfile` / :class:`KernelStat` and
  :func:`parse_kineto_trace` -- parsed Kineto trace breakdown.
"""

from .prompts import BENCH_PROMPTS, BenchPrompt
from .report import (
    CSV_COLUMNS,
    markdown_table,
    markdown_table_detail,
    write_csv,
)
from .runner import RunResult, StageTimes, run_n, run_one
from .trace import (
    KernelStat,
    StageProfile,
    TraceProfile,
    parse_kineto_trace,
    trace_profile_markdown,
    trace_profile_top_kernels,
)

__all__ = [
    "BENCH_PROMPTS",
    "BenchPrompt",
    "CSV_COLUMNS",
    "KernelStat",
    "RunResult",
    "StageProfile",
    "StageTimes",
    "TraceProfile",
    "markdown_table",
    "markdown_table_detail",
    "parse_kineto_trace",
    "run_n",
    "run_one",
    "trace_profile_markdown",
    "trace_profile_top_kernels",
    "write_csv",
]
