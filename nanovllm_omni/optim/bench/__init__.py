"""Benchmark harness for the four MiniMind-O stages (TK-011).

Public surface used by ``tests/test_optim_bench.py`` and the
``python -m nanovllm_omni.optim.bench`` CLI:

* :data:`BENCH_PROMPTS` / :class:`BenchPrompt` -- the fixed prompt dataclass + set.
* :class:`StageTimes` / :class:`RunResult` -- timing and result shapes.
* :func:`run_one` / :func:`run_n` -- drive the four helpers end-to-end.
* :func:`write_csv` / :func:`markdown_table` -- result reporters.
"""

from .prompts import BENCH_PROMPTS, BenchPrompt
from .report import CSV_COLUMNS, markdown_table, write_csv
from .runner import RunResult, StageTimes, run_n, run_one

__all__ = [
    "BENCH_PROMPTS",
    "BenchPrompt",
    "CSV_COLUMNS",
    "RunResult",
    "StageTimes",
    "markdown_table",
    "run_n",
    "run_one",
    "write_csv",
]
