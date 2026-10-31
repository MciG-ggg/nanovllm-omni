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
from typing import Any

from .prompts import BENCH_PROMPTS, BenchPrompt
from .report import markdown_table, markdown_table_detail, write_csv
from .runner import RunResult, run_n
from .trace import (
    parse_kineto_trace,
    trace_profile_markdown,
    trace_profile_top_kernels,
)


def _load_bundle(args: argparse.Namespace):
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    kwargs: dict[str, object] = {}
    if args.mimi:
        kwargs["mimi_model_id"] = args.mimi
    bundle = load_minimind_omni_bundle(model_id=args.model, device=args.device, **kwargs)
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
        "seed": args.seed,
    }


def _graph_overrides(args: argparse.Namespace) -> dict[str, bool]:
    # Only the thinker has a graph path (ADR-0005 / ADR-006): the talker
    # bypasses the fork ModelRunner, so it has no ``enforce_eager``.
    return {
        name: value
        for name, value in (("use_thinker_cuda_graph", args.use_thinker_cuda_graph),)
        if value is not None
    }


def _extra_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Stage knobs that travel through ``OmniEngineArgs.extra``."""
    capture_bs = getattr(args, "thinker_max_capture_bs", None)
    if capture_bs is None:
        return {}
    return {"thinker_max_capture_bs": capture_bs}


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


def _make_full_omni(args: argparse.Namespace):
    """Build the full three-stage ``Omni`` the E2E measurements run on."""
    import torch

    from nanovllm_omni import Omni

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Omni(
        model=args.model,
        device=device,
        dtype="float16" if device.startswith("cuda") else "float32",
        trust_remote_code=True,
        pipeline="minimind_o",
        extra=_extra_overrides(args),
        **_graph_overrides(args),
    )


def cmd_batch(args: argparse.Namespace) -> int:
    """Measure throughput as a function of how many requests share a batch.

    Every size is measured against the same prompts, and each request's
    audio is compared with its own single-request run: a batch may not
    change what a request produces.
    """
    import csv
    import statistics

    from nanovllm_omni.bench.runner import run_batch

    prompts = _resolve_prompts(args.prompts)
    omni = _make_full_omni(args)
    kwargs = {
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]

    solo_audio: dict[str, bytes] = {}
    for prompt in prompts:
        solo_audio[prompt.id] = run_batch(omni, [prompt], **kwargs)["audios"][0]

    rows: list[dict[str, Any]] = []
    for size in sizes:
        batch = [prompts[i % len(prompts)] for i in range(size)]
        for _ in range(max(args.warmup, 0)):
            run_batch(omni, batch, **kwargs)
        runs = [run_batch(omni, batch, **kwargs) for _ in range(args.runs)]
        wall_ms = statistics.median(r["wall_ms"] for r in runs)
        # A request's audio may differ from its solo run: a batched prefill
        # runs different kernels over a packed batch, so the bridge rows
        # differ in the last bits and the talker is sensitive to them. The
        # thinker's tokens do not differ. Count the matches instead of a
        # boolean, so one mismatch stays visible.
        matched = sum(
            audio == solo_audio[prompt.id]
            for prompt, audio in zip(batch, runs[-1]["audios"], strict=True)
        )
        rows.append(
            {
                "batch_size": size,
                "wall_ms": f"{wall_ms:.3f}",
                "per_request_ms": f"{wall_ms / size:.3f}",
                "req_per_s": f"{1000.0 * size / wall_ms:.4f}",
                "same_audio": f"{matched}/{size}",
                "vram_mb": f"{max(r['vram_peak_mb'] for r in runs):.3f}",
                "thinker_ms": f"{runs[-1]['stage_ms'].get('thinker', 0.0):.3f}",
                "talker_ms": f"{runs[-1]['stage_ms'].get('talker', 0.0):.3f}",
                "code2wav_ms": f"{runs[-1]['stage_ms'].get('code2wav', 0.0):.3f}",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    print()
    print(
        f"{'batch':>5} {'wall ms':>9} {'per req ms':>11} {'req/s':>7} {'same':>6} {'vram MiB':>9}"
    )
    for row in rows:
        print(
            f"{row['batch_size']:>5} {float(row['wall_ms']):>9.0f} "
            f"{float(row['per_request_ms']):>11.0f} {float(row['req_per_s']):>7.2f} "
            f"{row['same_audio']:>6} {float(row['vram_mb']):>9.0f}"
        )
    print()
    print("same = requests whose audio matched their own single-request run")
    return 0


def cmd_time(args: argparse.Namespace) -> int:
    prompts = _resolve_prompts(args.prompts)

    if args.pipeline == "full":
        import torch

        from nanovllm_omni import Omni
        from nanovllm_omni.bench.runner import run_n_full

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = _kwargs(args)
        kwargs.pop("open_thinking", None)  # Omni.generate has no open-thinking switch
        omni = Omni(
            model=args.model,
            device=device,
            dtype="float16" if device.startswith("cuda") else "float32",
            trust_remote_code=True,
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
    """Run CUDA-Graph cells in isolated processes."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_cells = (
        ("none", False),
        ("thinker", True),
    )
    failed: list[str] = []
    for graph, thinker_graph in graph_cells:
        out = out_dir / f"graph-{graph}.csv"
        command = [
            sys.executable,
            "-m",
            "nanovllm_omni.bench",
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
        ]
        if args.device:
            command.extend(("--device", args.device))
        if args.mimi:
            command.extend(("--mimi", args.mimi))
        if args.prompts:
            command.extend(("--prompts", args.prompts))
        print(f"running graph={graph}: {' '.join(command)}")
        if subprocess.run(command).returncode:
            failed.append(f"graph={graph}")
    if failed:
        print(f"failed cells: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Length matrix: prompts x {--lengths} budget sweep (defect B verification).

    Each cell runs ``run_n(bundle, prompt, n=args.runs, warmup=args.warmup,
    max_tokens=L)`` and reports the
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
        "nanovllm_omni.bench",
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
    p = argparse.ArgumentParser(prog="nanovllm_omni.bench")
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
        help="Thinker CUDA Graph on/off. Applies to --pipeline full only; "
        "the thinker-bundle path has no graph to toggle.",
    )
    common.add_argument(
        "--thinker-max-capture-bs",
        type=int,
        default=None,
        help="How many decode batch sizes get a CUDA graph (default: 1, the "
        "stage default). Larger values graph the batch buckets too, at the "
        "cost of the bs=1 graph no longer matching eager byte-for-byte.",
    )
    common.add_argument(
        "--pipeline",
        choices=("thinker", "full"),
        default="thinker",
        help="measurement target: thinker single-stage decode (default, the "
        "historical bench) or the full three-stage E2E through Omni.generate.",
    )

    p_batch = sub.add_parser(
        "batch",
        parents=[common],
        help="Throughput and output identity as a function of batch size",
    )
    p_batch.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes to measure (default: 1,2,4,8)",
    )
    p_batch.add_argument("--runs", type=int, default=3)
    p_batch.add_argument("--warmup", type=int, default=1)
    p_batch.add_argument(
        "--out",
        default="docs/perf/batch.csv",
        help="CSV output path (default: docs/perf/batch.csv)",
    )
    p_batch.set_defaults(func=cmd_batch)

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
