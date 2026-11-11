#!/usr/bin/env python3
"""Bench: the fork Scheduler on the MLX backend (Mac track).

Drives ``ThinkerDecodeSession`` — the same entry point the thinker stage
uses — so the numbers include what production pays.

Four shapes:

- ``single``  one request.
- ``ragged``  four requests of different prompt lengths, one batch.
- ``uneven``  four requests where one finishes long before the others.
  The finished one leaves the batch, so its cache row is recycled and the
  remaining rows keep decoding.
- ``join``    one request runs for a few steps, then a second arrives.
  It is prefilled on the next step and ``BatchKVCache.extend``-ed onto the
  running cache; the step after that decodes both. The table reports the
  new request's latency from arrival, next to its standalone wall — the
  number the snapshot batch could not produce because it only picked up
  new arrivals after the batch drained.

The numbers are MLX-track only. ADR 0014 forbids subtracting them from
CUDA numbers: the two tracks live on different machines.

Usage (from the repo root):
    PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
        uv run --no-project --with mlx --with mlx-lm --python 3.12 \
        python tools/bench_mlx_scheduler.py --runs 5 --warmup 2
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import time
from typing import Any

from nanovllm_omni.bench.prompts import BENCH_PROMPTS

DEFAULT_MODEL = os.path.expanduser("~/models/minimind-3o/mlx-thinker")

# The fork's SamplingParams forbids temperature=0; the CUDA thinker clamps
# greedy to 1e-5 for the same reason.
_GREEDY = 1e-5


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _stats(values: list[float]) -> str:
    return f"p50={statistics.median(values):.2f} min={min(values):.2f} max={max(values):.2f}"


def _prompt_text(prompt_id: str) -> str:
    for prompt in BENCH_PROMPTS:
        if prompt.id == prompt_id:
            return prompt.text
    raise SystemExit(f"unknown prompt id {prompt_id!r}")


def _build_runner(model_dir: str, dtype: str, max_num_seqs: int, num_blocks: int) -> Any:
    import mlx.core as mx
    from mlx_lm.utils import load as mlx_load

    from nanovllm_omni.backend.mlx_backend import MlxConfig, MlxStageRunner

    model, tokenizer = mlx_load(model_dir)
    model.set_dtype(getattr(mx, dtype))
    mx.eval(model.parameters())

    config = MlxConfig(
        model_path=model_dir,
        dtype=dtype,
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=num_blocks,
    )
    runner = MlxStageRunner.__new__(MlxStageRunner)
    runner._mx = mx
    runner.config = config
    runner._model = model
    runner.tokenizer = tokenizer
    runner._cache = None
    runner._rng = None
    runner._left_padding = [0]
    runner._bridge = None
    runner._bridge_layer = 0
    config.eos = min(int(token) for token in tokenizer.eos_token_ids)
    return runner


def _tokenize(runner: Any, prompt_id: str, limit: int = 48) -> list[int]:
    text = _prompt_text(prompt_id)
    return list(runner.tokenizer.encode(text, add_special_tokens=False))[:limit]


def _think(ids: list[int], max_tokens: int) -> Any:
    from nanovllm_omni.models.minimind_omni.stage_runner import ThinkerDecode

    return ThinkerDecode(token_ids=ids, max_tokens=max_tokens, temperature=_GREEDY)


def _bench_single(runner: Any, args: argparse.Namespace) -> dict[str, Any]:
    from nanovllm_omni.models.minimind_omni.stage_runner import decode_minimind_mlx

    ids = _tokenize(runner, args.prompt_id)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        decode_minimind_mlx(runner, [_think(ids, args.max_tokens)])
        times.append((time.perf_counter() - start) * 1000)
    return {
        "shape": "single",
        "rows": "1",
        "prompt_len": str(len(ids)),
        "wall_ms": _stats(times),
    }


def _bench_ragged(runner: Any, args: argparse.Namespace) -> dict[str, Any]:
    from nanovllm_omni.models.minimind_omni.stage_runner import decode_minimind_mlx

    ids = [_tokenize(runner, one) for one in args.ragged_prompts.split(",")]
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        decode_minimind_mlx(runner, [_think(one, args.max_tokens) for one in ids])
        times.append((time.perf_counter() - start) * 1000)
    return {
        "shape": "ragged",
        "rows": str(len(ids)),
        "prompt_len": "+".join(str(len(one)) for one in ids),
        "wall_ms": _stats(times),
    }


def _bench_uneven(runner: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Four requests where one finishes long before the others.

    The finished request leaves the batch, so its cache row is recycled
    and the remaining rows keep decoding. This is the shape that exercises
    the row-recycling path.
    """
    from nanovllm_omni.models.minimind_omni.stage_runner import decode_minimind_mlx

    ids = [_tokenize(runner, one) for one in args.ragged_prompts.split(",")]
    # The first request stops early; the rest run to the common length.
    limits = [args.early_stop] + [args.max_tokens] * (len(ids) - 1)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        decode_minimind_mlx(
            runner, [_think(one, limit) for one, limit in zip(ids, limits, strict=True)]
        )
        times.append((time.perf_counter() - start) * 1000)
    return {
        "shape": "uneven",
        "rows": f"{len(ids)} -> {len(ids) - 1}",
        "prompt_len": "+".join(str(len(one)) for one in ids),
        "wall_ms": _stats(times),
    }


def _bench_join(runner: Any, args: argparse.Namespace) -> dict[str, Any]:
    """A newcomer arrives after the first request has started decoding.

    The old snapshot batch could not pick it up until the running batch
    drained, so its baseline is "decode alone afterwards". The join path
    reports its latency from arrival instead.
    """
    from nanovllm_omni.models.minimind_omni.stage_runner import (
        ThinkerDecodeSession,
        decode_minimind_mlx,
    )

    ids_a = _tokenize(runner, args.prompt_id)
    ids_b = _tokenize(runner, args.join_prompt_id)

    def run_once() -> tuple[float, float]:
        """One join scenario: (newcomer latency, whole wall), in ms."""
        start = time.perf_counter()
        session = ThinkerDecodeSession(runner, backend="mlx")
        session.add(_think(ids_a, args.max_tokens), "a")
        for _ in range(args.join_after):
            session.step()
        arrived = time.perf_counter()
        session.add(_think(ids_b, args.max_tokens), "b")
        b_done = None
        while session.has_work():
            for request_id, _, _ in session.step():
                if request_id == "b":
                    b_done = time.perf_counter()
        return (b_done - arrived) * 1000, (time.perf_counter() - start) * 1000

    def run_alone() -> float:
        start = time.perf_counter()
        decode_minimind_mlx(runner, [_think(ids_b, args.max_tokens)])
        return (time.perf_counter() - start) * 1000

    run_once()
    run_alone()

    times = []
    b_after_join = []
    for _ in range(args.runs):
        newcomer_ms, wall_ms = run_once()
        b_after_join.append(newcomer_ms)
        times.append(wall_ms)

    b_alone = [run_alone() for _ in range(args.runs)]

    return {
        "shape": "join",
        "rows": "1 -> 2",
        "prompt_len": f"{len(ids_a)}+{len(ids_b)}",
        "wall_ms": _stats(times),
        "b_after_join_ms": _stats(b_after_join),
        "b_alone_ms": _stats(b_alone),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-id", default="medium_01")
    parser.add_argument("--join-prompt-id", default="short_01")
    parser.add_argument(
        "--early-stop",
        type=int,
        default=4,
        help="how many tokens the first request in the uneven shape generates",
    )
    parser.add_argument(
        "--ragged-prompts",
        default="short_01,medium_01,short_02,medium_02",
        help="comma-separated bench prompt ids; runs one ragged batch",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--join-after",
        type=int,
        default=3,
        help="decode steps the first request runs before the join scenario adds the second",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    args = parser.parse_args()

    import mlx.core as mx

    load_start = time.perf_counter()
    runner = _build_runner(args.model, args.dtype, args.max_num_seqs, args.num_kvcache_blocks)
    load_ms = (time.perf_counter() - load_start) * 1000

    from nanovllm_omni.models.minimind_omni.stage_runner import decode_minimind_mlx

    for _ in range(args.warmup):
        decode_minimind_mlx(runner, [_think(_tokenize(runner, args.prompt_id), args.max_tokens)])

    print(f"machine      {platform.platform()}")
    print(f"mlx          {mx.__version__}")
    print(f"commit       {_git_commit()}")
    print(f"model load   {load_ms:.1f} ms")
    print(f"runs/warmup  {args.runs}/{args.warmup}")
    print()

    header = f"{'shape':<8} {'rows':<8} {'prompt len':<14} {'steps':<6} wall"
    print(header)
    for bench in (_bench_single, _bench_ragged, _bench_uneven, _bench_join):
        result = bench(runner, args)
        steps = result.get("steps", "-")
        print(
            f"{result['shape']:<8} {result['rows']:<8} {result['prompt_len']:<14} "
            f"{steps:<6} {result['wall_ms']}"
        )
        if "b_after_join_ms" in result:
            print(
                f"{'':<8} {'':<8} B after join {result['b_after_join_ms']}; "
                f"B alone {result['b_alone_ms']}"
            )

    print()
    print(f"peak memory   {mx.get_peak_memory() / 2**20:.1f} MiB")
    print(f"active memory {mx.get_active_memory() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
