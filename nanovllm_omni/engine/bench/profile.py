"""Thin wrappers for ``torch.profiler`` and the ``nsys`` CLI."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler


@contextmanager
def torch_profiler(out_dir: str | Path) -> Iterator[None]:
    """Capture torch profiler trace (Kineto) under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            on_trace_ready=tensorboard_trace_handler(str(out)),
        ),
        record_function("bench_outer"),
    ):
        yield


def nsys_command(
    output: str | Path,
    *,
    target_args: list[str],
    trace: str = "cuda,nvtx",
    extra: list[str] | None = None,
) -> list[str]:
    """Build an ``nsys profile`` argv to wrap a child Python invocation."""
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("nsys not on PATH. Install with: sudo apt install -y nsight-systems-cli")
    argv = [
        nsys,
        "profile",
        "-o",
        str(output),
        "-t",
        trace,
        "--force-overwrite=true",
        "-s",
        "none",
    ]
    if extra:
        argv.extend(extra)
    argv.extend(target_args)
    return argv


def run_nsys(output: str | Path, target_args: list[str], **kwargs: Any) -> int:
    """Spawn ``nsys profile`` around ``target_args`` and wait for it."""
    cmd = nsys_command(output, target_args=target_args, **kwargs)
    return subprocess.call(cmd)
