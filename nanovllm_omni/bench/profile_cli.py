"""Profiling subcommands for the benchmark CLI."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runner import run_n
from .trace import parse_kineto_trace, trace_profile_markdown, trace_profile_top_kernels


def profile_detail(
    args: Namespace,
    *,
    load_bundle: Callable[[Namespace], Any],
    resolve_prompts: Callable[[str | None], list[Any]],
    run_all: Callable[..., Any],
    kwargs_for: Callable[[Namespace], dict[str, Any]],
) -> int:
    """Capture a torch trace and a per-stage kernel breakdown."""
    from torch.profiler import ProfilerActivity, profile

    bundle = load_bundle(args)
    prompts = resolve_prompts(args.prompts)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = out_prefix.with_suffix(".trace.json")
    detail_path = out_prefix.with_suffix(".detail.md")

    if args.warmup:
        for prompt in prompts:
            run_n(bundle, prompt, n=0, warmup=args.warmup, **kwargs_for(args))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        run_all(bundle, prompts, n=args.runs, warmup=0, run_kwargs=kwargs_for(args))
    prof.export_chrome_trace(str(trace_path))

    profile_data = parse_kineto_trace(trace_path)
    summary = trace_profile_markdown(profile_data)
    detail = trace_profile_top_kernels(profile_data, per_stage=5)
    detail_path.write_text(
        f"# Per-stage kernel breakdown ({trace_path})\n\n{summary}\n\n{detail}\n",
        encoding="utf-8",
    )

    print(f"trace:        {trace_path}")
    print(f"detail md:    {detail_path}")
    print()
    print(summary)
    print()
    print(detail)
    return 0
