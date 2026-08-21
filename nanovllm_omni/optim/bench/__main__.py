"""CLI for the TK-011 Session-1 benchmark harness.

Subcommands:

* ``time``       -- run N cold iterations over the prompt set; write CSV.
* ``trace-torch`` -- run once under :class:`torch.profiler`; dump to a dir.
* ``trace-nsys`` -- re-invoke ``time --runs=1`` under ``nsys profile``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .prompts import BENCH_PROMPTS
from .report import write_csv
from .runner import run_n


def _load_bundle(args: argparse.Namespace):
    from nanovllm_omni.models.minimind_omni import create_bundle

    kwargs: dict[str, str] = {}
    if args.mimi:
        kwargs["mimi_model_id"] = args.mimi
    return create_bundle(model_id=args.model, device=args.device, **kwargs)


def _resolve_prompts(args: argparse.Namespace) -> tuple[str, ...]:
    if args.prompts is None or args.prompts == "all":
        return BENCH_PROMPTS
    # --prompts accepts either "0,3,5" or a single int like "2".
    parts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    indices = [int(p) for p in parts]
    return tuple(BENCH_PROMPTS[i] for i in indices)


def cmd_time(args: argparse.Namespace) -> int:
    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args)
    results = run_n(
        bundle,
        prompts,
        n=args.runs,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    rows = [r.as_csv_row() for r in results]
    out = write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def cmd_trace_torch(args: argparse.Namespace) -> int:
    from .profile import torch_profiler

    bundle = _load_bundle(args)
    prompts = _resolve_prompts(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch_profiler(out_dir):
        results = run_n(
            bundle,
            prompts,
            n=1,
            warmup=0,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    print(f"torch trace saved under {out_dir}; {len(results)} result(s)")
    return 0


def cmd_trace_nsys(args: argparse.Namespace) -> int:
    from .profile import run_nsys

    target = [
        sys.executable,
        "-m",
        "nanovllm_omni.optim.bench",
        "time",
        "--runs",
        "1",
        "--warmup",
        str(args.warmup),
        "--model",
        args.model,
        "--out",
        args.csv_out,
    ]
    if args.device:
        target.extend(["--device", args.device])
    if args.mimi:
        target.extend(["--mimi", args.mimi])
    if args.prompts:
        target.extend(["--prompts", args.prompts])
    if args.max_tokens is not None:
        target.extend(["--max-tokens", str(args.max_tokens)])
    rc = run_nsys(args.out, target_args=target)
    return rc


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
        help='Comma-separated prompt indices ("0,3,5"); default = all six.',
    )
    common.add_argument("--max-tokens", type=int, default=16)
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--top-p", type=float, default=0.9)

    p_time = sub.add_parser("time", parents=[common], help="Time N runs and write CSV")
    p_time.add_argument("--runs", type=int, default=5)
    p_time.add_argument("--warmup", type=int, default=1)
    p_time.add_argument(
        "--out",
        default="docs/perf/session-1.csv",
        help="CSV output path (default: docs/perf/session-1.csv)",
    )
    p_time.set_defaults(func=cmd_time)

    p_torch = sub.add_parser("trace-torch", parents=[common], help="Capture torch.profiler trace")
    p_torch.add_argument("--out", required=True, help="Directory for tensorboard trace_handler")
    p_torch.set_defaults(func=cmd_trace_torch)

    p_nsys = sub.add_parser("trace-nsys", parents=[common], help="Capture nsys profile")
    p_nsys.add_argument("--out", required=True, help="nsys output prefix (e.g. /tmp/s1)")
    p_nsys.add_argument("--warmup", type=int, default=1)
    p_nsys.add_argument(
        "--csv-out",
        default="/tmp/bench-nsys.csv",
        help="CSV output of the child time run",
    )
    p_nsys.set_defaults(func=cmd_trace_nsys)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
