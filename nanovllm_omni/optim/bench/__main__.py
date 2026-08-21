"""CLI for the TK-011 Session-1 benchmark harness.

Subcommands:

* ``time``         -- run N iterations; write CSV + print markdown table.
* ``trace-torch``  -- run once under torch.profiler; export Chrome trace.
* ``trace-nsys``   -- re-invoke ``_nsys-inner`` under ``nsys profile``.
* ``_nsys-inner``  -- private inner command used by ``trace-nsys``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .prompts import BENCH_PROMPTS, BenchPrompt
from .report import markdown_table, write_csv
from .runner import RunResult, run_n


def _load_bundle(args: argparse.Namespace):
    from nanovllm_omni.models.minimind_omni import create_bundle

    kwargs: dict[str, str] = {}
    if args.mimi:
        kwargs["mimi_model_id"] = args.mimi
    return create_bundle(model_id=args.model, device=args.device, **kwargs)


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
    print(markdown_table(results))
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

    p_time = sub.add_parser("time", parents=[common], help="Time N runs and write CSV")
    p_time.add_argument("--runs", type=int, default=5)
    p_time.add_argument("--warmup", type=int, default=1)
    p_time.add_argument(
        "--out",
        default="docs/perf/session-1.csv",
        help="CSV output path (default: docs/perf/session-1.csv)",
    )
    p_time.set_defaults(func=cmd_time)

    p_torch = sub.add_parser(
        "trace-torch", parents=[common], help="Capture torch.profiler Chrome trace"
    )
    p_torch.add_argument("--out", required=True, help="Chrome trace JSON path")
    p_torch.set_defaults(func=cmd_trace_torch)

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
