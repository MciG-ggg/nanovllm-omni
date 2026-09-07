"""Single entry point for stage annotations.

Wraps :func:`torch.profiler.record_function` and :func:`torch.cuda.nvtx.range`
under one context manager so that the same code path is visible to the
``torch.profiler`` Kineto trace **and** to the ``nsys`` NVTX summary.

Both calls are cheap and safe to issue unconditionally: ``record_function``
is a no-op when no profiler is active and ``nvtx.range`` is a thin wrapper
around the NVTX C API that is itself cheap when no ``nsys`` collector is
attached.  Naming is kept identical to the historical ``record_function``
labels so existing parsers (``parse_kineto_trace``) keep working.

Internal module — model code and benchmarks import :func:`stage` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Open a Kineto ``user_annotation`` and an NVTX range of the same name.

    The NVTX half is skipped on hosts without a working CUDA build (the
    ``torch.cuda.nvtx`` extension is a stub that raises on every call when
    no GPU runtime is available).  On a CUDA host the NVTX call is cheap
    enough to issue unconditionally: ``record_function`` is a no-op when
    no profiler is active, and ``nvtx.range`` only shows up in summaries
    when an ``nsys -t cuda,nvtx`` collector is wrapping the process.
    """
    import torch
    import torch.profiler as profiler

    with profiler.record_function(name):
        if torch.cuda.is_available():
            import torch.cuda.nvtx as nvtx

            with nvtx.range(name):
                yield
        else:
            yield
