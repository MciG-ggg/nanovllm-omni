"""CLI for the TK-011 Session-1 benchmark harness.

Subcommands:

* ``time``            -- run N iterations; write CSV + print markdown table.
* ``trace-torch``     -- run once under torch.profiler; export Chrome trace.
* ``profile-detail``  -- run under torch.profiler; dump trace AND a parsed
  per-stage kernel breakdown (top kernels, kernel count, n_steps).
* ``trace-nsys``      -- re-invoke ``_nsys-inner`` under ``nsys profile``.
* ``_nsys-inner``     -- private inner command used by ``trace-nsys``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .prompts import BENCH_PROMPTS, BenchPrompt
from .report import markdown_table, markdown_table_detail, write_csv
from .runner import RunResult, run_n
from .trace import (
    parse_kineto_trace,
    trace_profile_markdown,
    trace_profile_top_kernels,
)


def _load_bundle(args: argparse.Namespace):
    from nanovllm_omni.models.minimind_omni import create_bundle

    kwargs: dict[str, str] = {}
    if args.mimi:
        kwargs["mimi_model_id"] = args.mimi
    bundle = create_bundle(model_id=args.model, device=args.device, **kwargs)
    return bundle


def _resolve_prompts(arg: str | None) -> list[BenchPrompt]:
    if arg is None or arg == "all":
        return list(BENCH_PROMPTS)
    by_id = {p.id: p for p in BENCH_PROMPTS}
    ids = [s.strip() for s in arg.split(",") if s.strip()]
    try:
        return [by_id[i] for i in ids]
    except KeyError as exc:
        valid = ", ".join(sorted(by_id))
        raise SystemExit(f"unknown prompt id: {exc.args[0]} (valid: {valid})") from exc


def _kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "open_thinking": args.open_thinking,
        "use_cuda_graph": args.use_cuda_graph,
        "seed": args.seed,
    }


def _run_all(
    bundle: object,
    prompts: list[BenchPrompt],
    *,
    n: int,
    warmup: int,
    run_kwargs: dict[str, object],
) -> list[RunResult]:
    results: list[RunResult] = []
    for prompt in prompts:
        results.extend(run_n(bundle, prompt, n=n, warmup=warmup, **run_kwargs))
    return results


def cmd_time(args: argparse.Namespace) -> int:
    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args.prompts)
    results = _run_all(bundle, prompts, n=args.runs, warmup=args.warmup, run_kwargs=_kwargs(args))
    out = write_csv(results, args.out)
    print(f"wrote {len(results)} rows to {out}")
    print()
    print("Spec markdown table (10 cols):")
    print(markdown_table(results))
    print("Detailed markdown table (GPU/CPU split + per-step):")
    print(markdown_table_detail(results))
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Length matrix: prompts x {--lengths} budget sweep (defect B verification).

    Each cell runs ``run_n(bundle, prompt, n=args.runs, warmup=args.warmup,
    max_tokens=L, use_cuda_graph=args.use_cuda_graph)`` and reports the
    median frames + generate_ms. Emits a markdown table; defect B fix is
    visible here as ``frames < L`` once content terminates naturally.
    """
    import statistics

    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args.prompts)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not lengths:
        raise SystemExit("--lengths must list at least one integer")

    rows: list[dict[str, object]] = []
    for prompt in prompts:
        for length in lengths:
            kwargs = _kwargs(args)
            kwargs["max_tokens"] = length
            for r in run_n(bundle, prompt, n=args.runs, warmup=args.warmup, **kwargs):
                rows.append(
                    {
                        "prompt_id": prompt.id,
                        "length": length,
                        "frames": r.frames,
                        "generate_ms": r.times.generate_ms,
                        "total_ms": r.times.total_ms,
                    }
                )

    # Per-cell aggregate
    cells: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        cells.setdefault((row["prompt_id"], row["length"]), []).append(row)

    cols = ("prompt_id", "length", "frames", "generate_median_ms", "total_median_ms")
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for prompt_id, length in sorted(cells):
        rs = cells[(prompt_id, length)]
        gen = [float(r["generate_ms"]) for r in rs]
        tot = [float(r["total_ms"]) for r in rs]
        frames = [int(r["frames"]) for r in rs]
        cells_row = (
            prompt_id,
            str(length),
            str(int(statistics.median(frames))),
            f"{statistics.median(gen):.2f}",
            f"{statistics.median(tot):.2f}",
        )
        lines.append("| " + " | ".join(cells_row) + " |")
    print("\n".join(lines) + "\n")
    return 0


def cmd_trace_torch(args: argparse.Namespace) -> int:
    from torch.profiler import ProfilerActivity, profile

    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args.prompts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        results = _run_all(bundle, prompts, n=1, warmup=0, run_kwargs=_kwargs(args))
    prof.export_chrome_trace(str(out_path))
    print(f"torch trace exported to {out_path}; {len(results)} result(s)")
    return 0


def cmd_profile_detail(args: argparse.Namespace) -> int:
    """Capture a torch.profiler trace AND a parsed per-stage kernel breakdown."""
    from torch.profiler import ProfilerActivity, profile

    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args.prompts)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = out_prefix.with_suffix(".trace.json")
    detail_path = out_prefix.with_suffix(".detail.md")

    # Warmup runs OUTSIDE the profiler so JIT compilation, CUDA kernel
    # autotune, and cuDNN benchmark heuristics do not bloat the trace.
    if args.warmup:
        for prompt in prompts:
            run_n(
                bundle,
                prompt,
                n=0,
                warmup=args.warmup,
                **_kwargs(args),
            )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        _run_all(
            bundle,
            prompts,
            n=args.runs,
            warmup=0,
            run_kwargs=_kwargs(args),
        )
    prof.export_chrome_trace(str(trace_path))

    profile_data = parse_kineto_trace(trace_path)
    summary = trace_profile_markdown(profile_data)
    detail = trace_profile_top_kernels(profile_data, per_stage=5)
    detail_md = f"# Per-stage kernel breakdown ({trace_path})\n\n" f"{summary}\n\n" f"{detail}\n"
    detail_path.write_text(detail_md, encoding="utf-8")

    print(f"trace:        {trace_path}")
    print(f"detail md:    {detail_path}")
    print()
    print(summary)
    print()
    print(detail)
    return 0


def cmd_trace_nsys(args: argparse.Namespace) -> int:
    nsys = shutil.which("nsys")
    if not nsys:
        print(
            "nsys not on PATH. Install with: " "sudo apt install -y nsight-systems-cli",
            file=sys.stderr,
        )
        return 1

    target = [
        sys.executable,
        "-m",
        "nanovllm_omni.optim.bench",
        "_nsys-inner",
        "--out",
        args.csv_out,
        "--model",
        args.model,
    ]
    if args.device:
        target.extend(["--device", args.device])
    if args.mimi:
        target.extend(["--mimi", args.mimi])
    if args.prompts:
        target.extend(["--prompts", args.prompts])
    if args.max_tokens is not None:
        target.extend(["--max-tokens", str(args.max_tokens)])
    target.extend(["--seed", str(args.seed)])

    cmd = [
        nsys,
        "profile",
        "-o",
        args.out,
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        *target,
    ]
    return subprocess.call(cmd)


def cmd_nsys_inner(args: argparse.Namespace) -> int:
    """Hidden inner command invoked by ``trace-nsys`` via the nsys wrapper."""
    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args.prompts)
    results = _run_all(bundle, prompts, n=1, warmup=0, run_kwargs=_kwargs(args))
    out = write_csv(results, args.out)
    print(f"wrote {len(results)} rows to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanovllm_omni.optim.bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="jingyaogong/minimind-3o")
    common.add_argument("--device", default=None)
    common.add_argument("--mimi", default=None)
    common.add_argument(
        "--prompts",
        default=None,
        help='Comma-separated prompt IDs ("short_01,medium_01"); default = all six.',
    )
    common.add_argument("--max-tokens", type=int, default=16)
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--top-p", type=float, default=0.9)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--open-thinking", action="store_true", help="Pass open_thinking=True to model.generate"
    )
    common.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Route decode through the CUDA-Graph fast path (run_generate "
        "use_cuda_graph=True; opt-in, CUDA-only).",
    )

    p_time = sub.add_parser("time", parents=[common], help="Time N runs and write CSV")
    p_time.add_argument("--runs", type=int, default=5)
    p_time.add_argument("--warmup", type=int, default=1)
    p_time.add_argument(
        "--out",
        default="docs/perf/session-1.csv",
        help="CSV output path (default: docs/perf/session-1.csv)",
    )
    p_time.set_defaults(func=cmd_time)

    p_matrix = sub.add_parser(
        "matrix",
        parents=[common],
        help="Length matrix: prompts x --lengths sweep (defect B verification)",
    )
    p_matrix.add_argument(
        "--lengths",
        default="8,16,32,64,120",
        help="Comma-separated budgets to sweep (default: 8,16,32,64,120)",
    )
    p_matrix.add_argument("--runs", type=int, default=3)
    p_matrix.add_argument("--warmup", type=int, default=1)
    p_matrix.set_defaults(func=cmd_matrix)

    p_torch = sub.add_parser(
        "trace-torch", parents=[common], help="Capture torch.profiler Chrome trace"
    )
    p_torch.add_argument("--out", required=True, help="Chrome trace JSON path")
    p_torch.set_defaults(func=cmd_trace_torch)

    p_detail = sub.add_parser(
        "profile-detail",
        parents=[common],
        help="Capture trace + emit per-stage kernel breakdown",
    )
    p_detail.add_argument(
        "--out", required=True, help="Output prefix (writes <prefix>.trace.json + .detail.md)"
    )
    p_detail.add_argument("--runs", type=int, default=1)
    # Default warmup=1 runs OUTSIDE the profiler so the trace captures only
    # the steady-state kernels (JIT / autotune / cuDNN benchmark are skipped).
    p_detail.add_argument("--warmup", type=int, default=1)
    p_detail.set_defaults(func=cmd_profile_detail)

    p_nsys = sub.add_parser("trace-nsys", parents=[common], help="Capture nsys profile")
    p_nsys.add_argument("--out", required=True, help="nsys output prefix")
    p_nsys.add_argument(
        "--csv-out", default="/tmp/bench-nsys.csv", help="CSV output of the inner run"
    )
    p_nsys.set_defaults(func=cmd_trace_nsys)

    p_inner = sub.add_parser(
        "_nsys-inner",
        parents=[common],
        help="INTERNAL: invoked by trace-nsys under the nsys wrapper",
    )
    p_inner.add_argument("--out", required=True)
    p_inner.set_defaults(func=cmd_nsys_inner)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
