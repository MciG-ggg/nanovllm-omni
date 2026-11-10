#!/usr/bin/env python3
"""Bench: MiniMind-O thinker on the MLX backend (Mac track).

Uses the CUDA bench prompt set and the CUDA thinker's sampling recipe, but the
numbers are MLX-track only: ADR 0014 forbids subtracting them from CUDA numbers.

``--batch-prompts`` runs a static dense batch (left-padded to the longest
prompt); without it the bench runs one sequence.

Usage (from the repo root; the project venv does not ship mlx):
    PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
        uv run --no-project --with mlx --with mlx-lm --python 3.12 \
        python tools/bench_mlx_thinker.py --runs 5 --warmup 2
    PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
        uv run --no-project --with mlx --with mlx-lm --python 3.12 \
        python tools/bench_mlx_thinker.py --runs 5 --warmup 2 \
        --batch-prompts short_01,medium_01,short_02,medium_02
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import statistics
import subprocess
import time

from nanovllm_omni.bench.prompts import BENCH_PROMPTS

DEFAULT_MODEL = os.path.expanduser("~/models/minimind-3o/mlx-thinker")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-id", default="medium_01")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--batch-prompts",
        default=None,
        help="comma-separated bench prompt ids; runs one static dense batch",
    )
    args = parser.parse_args()

    import mlx.core as mx

    from nanovllm_omni.backend.mlx_backend import (
        MlxBackend,
        generate_mlx,
        generate_mlx_batch,
    )
    from nanovllm_omni.config.params import SamplingParams

    backend = MlxBackend(dtype=args.dtype)
    config = backend.make_config(os.path.expanduser(args.model))
    started = time.perf_counter()
    runner = backend.make_stage_runner(object, config)
    load_s = time.perf_counter() - started

    tokenizer = runner.tokenizer

    def _prompt_ids(prompt) -> list[int]:
        messages = [{"role": "user", "content": prompt.text}]
        if prompt.system:
            messages.insert(0, {"role": "system", "content": prompt.system})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, open_thinking=False
        )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        extra={"repetition_penalty": args.repetition_penalty},
    )

    if args.batch_prompts:
        prompts = [
            next(p for p in BENCH_PROMPTS if p.id == prompt_id.strip())
            for prompt_id in args.batch_prompts.split(",")
        ]
        token_ids_list = [_prompt_ids(prompt) for prompt in prompts]
        samplings = [sampling] * len(prompts)
        eos_ids = [tokenizer.eos_token_id] * len(prompts)
        mx.reset_peak_memory()
        for _ in range(args.warmup):
            generate_mlx_batch(runner, token_ids_list, samplings, eos_ids=eos_ids)
        results = [
            generate_mlx_batch(runner, token_ids_list, samplings, eos_ids=eos_ids)
            for _ in range(args.runs)
        ]
        last = results[-1]
        ttfts = [result.ttft_s * 1000 for result in results]
        step_ms = [result.decode_s * 1000 / max(result.steps - 1, 1) for result in results]
        prompt_desc = (
            f"batch {len(prompts)} prompts {[len(ids) for ids in token_ids_list]} tokens, "
            f"padded to {max(len(ids) for ids in token_ids_list)}"
        )
        token_desc = f"tokens per sequence: {[len(tokens) for tokens in last.token_ids]}"
        text_desc = None
    else:
        prompt = next(p for p in BENCH_PROMPTS if p.id == args.prompt_id)
        token_ids = _prompt_ids(prompt)
        mx.reset_peak_memory()
        for _ in range(args.warmup):
            generate_mlx(runner, token_ids, sampling, eos_id=tokenizer.eos_token_id)
        results = [
            generate_mlx(runner, token_ids, sampling, eos_id=tokenizer.eos_token_id)
            for _ in range(args.runs)
        ]
        last = results[-1]
        ttfts = [result.ttft_s * 1000 for result in results]
        step_ms = [result.decode_s * 1000 / max(len(result.token_ids) - 1, 1) for result in results]
        prompt_desc = f"prompt {prompt.id} ({len(token_ids)} tokens), open_thinking=False"
        token_desc = f"tokens ({len(last.token_ids)}): {last.token_ids}"
        text_desc = tokenizer.decode(last.token_ids, skip_special_tokens=True)

    info = mx.device_info()

    print(f"git: {_git_commit()}")
    print(f"host: {platform.platform()} | {info['device_name']}")
    print(f"mlx {importlib.metadata.version('mlx')} mlx-lm {importlib.metadata.version('mlx-lm')}")
    print(f"model: {os.path.expanduser(args.model)} | dtype {args.dtype}")
    print(prompt_desc)
    print(
        f"sampling: temperature={args.temperature} top_p={args.top_p} "
        f"repetition_penalty={args.repetition_penalty} "
        f"max_tokens={args.max_tokens} seed={args.seed}"
    )
    print(f"warmup={args.warmup} runs={args.runs}")
    print(f"load: {load_s * 1000:.1f} ms")
    print(f"TTFT ms: {_stats(ttfts)}")
    print(f"decode ms/step: {_stats(step_ms)}")
    print(f"peak memory during runs: {mx.get_peak_memory() / 2**20:.1f} MiB")
    print(f"active memory after runs: {mx.get_active_memory() / 2**20:.1f} MiB")
    print(token_desc)
    if text_desc is not None:
        print(f"text: {text_desc!r}")


if __name__ == "__main__":
    main()
