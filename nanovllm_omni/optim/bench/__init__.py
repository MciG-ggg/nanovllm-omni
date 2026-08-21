"""Benchmark harness for the four MiniMind-O stages (TK-011).

Public surface used by ``tests/test_optim_bench.py`` and the
``python -m nanovllm_omni.optim.bench`` CLI:

* :data:`BENCH_PROMPTS` -- the six fixed prompts.
* :class:`StageTimes` / :class:`RunResult` -- timing and result shapes.
* :func:`run_one` / :func:`run_n` -- drive the four helpers end-to-end.
"""

from .prompts import BENCH_PROMPTS
from .report import markdown_table, write_csv
from .runner import RunResult, StageTimes, run_n, run_one

__all__ = [
    "BENCH_PROMPTS",
    "RunResult",
    "StageTimes",
    "markdown_table",
    "run_n",
    "run_one",
    "write_csv",
]
