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

    kwargs: dict[str, object] = {}
    if args.mimi:
        kwargs["mimi_model_id"] = args.mimi
    if getattr(args, "enforce_eager", False):
        kwargs["enforce_eager"] = True
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
        "use_thinker_cuda_graph": bool(args.use_thinker_cuda_graph),
        "seed": args.seed,
    }


def _graph_overrides(args: argparse.Namespace) -> dict[str, bool]:
    return {
        name: value
        for name, value in (
            ("use_thinker_cuda_graph", args.use_thinker_cuda_graph),
            ("use_talker_cuda_graph", args.use_talker_cuda_graph),
        )
        if value is not None
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
    if args.pipeline == "full" and args.use_thinker_cuda_graph is True:
        raise SystemExit(
            "full E2E thinker CUDA Graph is unavailable: it does not preserve "
            "the required post-EOS bridge-state contract"
        )
    prompts = _resolve_prompts(args.prompts)

    if args.pipeline == "full":
        import torch

        from nanovllm_omni import Omni
        from nanovllm_omni.optim.bench.runner import run_n_full

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = _kwargs(args)
        kwargs.pop("use_thinker_cuda_graph", None)
        kwargs.pop("open_thinking", None)  # Omni.generate has no open-thinking switch
        omni = Omni(
            model=args.model,
            device=device,
            dtype="float16" if device.startswith("cuda") else "float32",
            trust_remote_code=True,
            enforce_eager=bool(getattr(args, "enforce_eager", False)),
            pipeline="minimind_o",
            **_graph_overrides(args),
        )
        results: list[RunResult] = []
        for prompt in prompts:
            results.extend(run_n_full(omni, prompt, n=args.runs, warmup=args.warmup, **kwargs))
    else:
        bundle = _load_bundle(args)
        results = _run_all(
            bundle, prompts, n=args.runs, warmup=args.warmup, run_kwargs=_kwargs(args)
        )

    out = write_csv(results, args.out)
    print(f"wrote {len(results)} rows to {out}")
    print()
    print("Spec markdown table (10 cols):")
    print(markdown_table(results))
    print("Detailed markdown table (GPU/CPU split + per-step):")
    print(markdown_table_detail(results))
    return 0


def cmd_sweep_graphs(args: argparse.Namespace) -> int:
    """Run fusion x CUDA-Graph cells in isolated processes."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_cells = (
        ("none", False, False),
        ("thinker", True, False),
        ("talker", False, True),
        ("both", True, True),
    )
    failed: list[str] = []
    for eager in (True, False):
        fusion = "off" if eager else "on"
        for graph, thinker_graph, talker_graph in graph_cells:
            out = out_dir / f"fusion-{fusion}-graph-{graph}.csv"
            command = [
                sys.executable,
                "-m",
                "nanovllm_omni.optim.bench",
                "time",
                "--pipeline",
                "full",
                "--model",
                args.model,
                "--max-tokens",
                str(args.max_tokens),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--seed",
                str(args.seed),
                "--runs",
                str(args.runs),
                "--warmup",
                str(args.warmup),
                "--out",
                str(out),
                "--use-thinker-cuda-graph" if thinker_graph else "--no-use-thinker-cuda-graph",
                "--use-talker-cuda-graph" if talker_graph else "--no-use-talker-cuda-graph",
            ]
            if args.device:
                command.extend(("--device", args.device))
            if args.mimi:
                command.extend(("--mimi", args.mimi))
            if args.prompts:
                command.extend(("--prompts", args.prompts))
            if eager:
                command.append("--enforce-eager")
            print(f"running fusion={fusion}, graph={graph}: {' '.join(command)}")
            if subprocess.run(command).returncode:
                failed.append(f"fusion={fusion}, graph={graph}")
    if failed:
        print(f"failed cells: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Length matrix: prompts x {--lengths} budget sweep (defect B verification).

    Each cell runs ``run_n(bundle, prompt, n=args.runs, warmup=args.warmup,
    max_tokens=L, use_thinker_cuda_graph=args.use_thinker_cuda_graph)`` and reports the
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
        "--use-thinker-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the deploy setting for thinker CUDA Graph (default: use deploy YAML).",
    )
    common.add_argument(
        "--use-talker-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the deploy setting for talker CUDA Graph (default: use deploy YAML).",
    )
    common.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Skip the four attention fusion monkey-patches (SDPA decode, "
        "fused QKV/gate-up, fused RMSNorm, fused RoPE). Used by the bench "
        "harness to measure an apples-to-apples baseline. CUDA Graph is "
        "gated independently by --use-thinker-cuda-graph.",
    )
    common.add_argument(
        "--pipeline",
        choices=("thinker", "full"),
        default="thinker",
        help="measurement target: thinker single-stage decode (default, the "
        "historical bench) or the full three-stage E2E through Omni.generate.",
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

    p_sweep = sub.add_parser(
        "sweep-graphs",
        parents=[common],
        help="Run fusion x {none, thinker, talker, both} graph cells in fresh processes",
    )
    p_sweep.add_argument("--runs", type=int, default=20)
    p_sweep.add_argument("--warmup", type=int, default=1)
    p_sweep.add_argument("--out-dir", default="docs/perf/aligned/graph-sweep")
    p_sweep.set_defaults(func=cmd_sweep_graphs)

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
