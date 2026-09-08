#!/usr/bin/env python3
"""Cold/hot CUDA Graph benchmark for nanovllm-omni thinker decode.

Separates measurement into two phases:
  - **cold path**: from a clean decoder (no graphs) through warmup, capture,
    instantiate, and the first request's full decode.
  - **hot path**: subsequent requests that reuse the established graphs
    (only prefill + replay, no capture overhead).

Also tests **interleaved shapes**: alternating prompt lengths and
``max_tokens`` budgets that trigger recapture, to measure the amortisation
breakeven and recapture cost.

Usage (RTX 3050 on WSL)::

    cd ~/nanovllm-omni
    source ~/venvs/nanovllm-omni/bin/activate
    HF_HUB_OFFLINE=1 python tools/bench_cold_hot_cuda_graph.py \\
        --model /home/mcig/minimind-3o \\
        --mimi  /home/mcig/mimi \\
        --max-tokens 16 \\
        --hot-runs 20 \\
        --seed 42

Without CUDA the script prints a CPU-only contract check and the WSL
command above, then exits 0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ColdMetrics:
    """Timings for the first-request path (ms)."""

    prefill_ms: float = 0.0
    warmup_ms: float = 0.0  # eager warmup forwards inside _capture
    capture_ms: float = 0.0  # graph capture + instantiate
    first_decode_ms: float = 0.0  # first generate_tokens after capture
    total_cold_ms: float = 0.0  # wall clock from clean decoder to first result
    num_graphs: int = 0
    num_warmup_forwards: int = 0


@dataclass
class HotMetrics:
    """Timings for repeat-request path (ms)."""

    latencies_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    num_tokens_per_request: list[int] = field(default_factory=list)

    def compute(self) -> None:
        if not self.latencies_ms:
            return
        s = sorted(self.latencies_ms)
        n = len(s)
        self.p50_ms = s[n // 2]
        self.p95_ms = s[int(n * 0.95)]
        self.mean_ms = statistics.mean(s)


@dataclass
class InterleavedMetrics:
    """Metrics for shape-varying requests."""

    recapture_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    shapes: list[str] = field(default_factory=list)  # "prompt_len:max_tokens"
    p50_ms: float = 0.0
    p95_ms: float = 0.0

    def compute(self) -> None:
        if not self.latencies_ms:
            return
        s = sorted(self.latencies_ms)
        n = len(s)
        self.p50_ms = s[n // 2]
        self.p95_ms = s[int(n * 0.95)]


@dataclass
class BenchResult:
    cold: ColdMetrics = field(default_factory=ColdMetrics)
    hot: HotMetrics = field(default_factory=HotMetrics)
    interleaved: InterleavedMetrics = field(default_factory=InterleavedMetrics)
    eager_baseline_ms: float = 0.0
    vram_peak_mb: float = 0.0
    amortisation_requests: int = 0  # requests to break even on capture cost


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _vram_mb() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def _run_eager(model: object, input_ids: object, max_new: int, seed: int) -> tuple[float, int]:
    """Run eager stream_generate and return (wall_ms, num_frames)."""
    import torch

    from nanovllm_omni.models.minimind_omni.generation import stream_generate

    eos_id = getattr(model, "eos_token_id_2", 2)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    frames: list[list[int]] = []
    for _text, audio in stream_generate(
        model,
        input_ids,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_id,
        open_thinking=False,
    ):
        if audio is not None:
            frames.append(audio)
    torch.cuda.synchronize()
    return _ms_since(t0), len(frames)


def _run_graphed(decoder: object, input_ids: object, seed: int) -> tuple[float, int]:
    """Run graphed generate_tokens and return (wall_ms, num_tokens)."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = decoder.generate_tokens(input_ids, seed=seed, return_audio=True)
    torch.cuda.synchronize()
    ms = _ms_since(t0)
    text_codes = tokens[0] if isinstance(tokens, tuple) else tokens
    return ms, len(text_codes)


def bench_cold(decoder: object, input_ids: object, seed: int) -> ColdMetrics:
    """Measure the cold path: prefill + warmup + capture + first decode."""
    import torch

    # Force a clean state: drop any existing graphs
    decoder.steps = []
    decoder._captured = False
    decoder._captured_len = -1
    decoder._captured_n_steps = -1
    decoder._zero_kv_contents()

    metrics = ColdMetrics()
    num_decode_graphs = max(decoder.n_steps - 1, 0)
    metrics.num_graphs = num_decode_graphs
    # Each graph gets 1 warmup forward + 1 capture forward
    metrics.num_warmup_forwards = num_decode_graphs

    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    # The cold path is a single generate_tokens call that internally:
    # 1. _prefill (eager)
    # 2. _capture (warmup + capture n_steps-1 graphs)
    # 3. replay loop
    torch.manual_seed(seed)
    decoder.generate_tokens(input_ids, seed=seed, return_audio=True)
    torch.cuda.synchronize()

    metrics.total_cold_ms = _ms_since(t_total_start)

    # We can't separate prefill/warmup/capture/decode without instrumenting
    # the decoder, so total_cold_ms is the authoritative cold number.
    # The components are estimated from the structure:
    # - num_warmup_forwards warmups + num_graphs captures + 1 prefill + num_decode replays
    return metrics


def bench_hot(decoder: object, input_ids: object, seed: int, num_runs: int) -> HotMetrics:
    """Measure repeat requests on established graphs."""
    import torch

    metrics = HotMetrics()
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.manual_seed(seed + i)
        tokens = decoder.generate_tokens(input_ids, seed=seed + i, return_audio=True)
        torch.cuda.synchronize()
        ms = _ms_since(t0)
        text_codes = tokens[0] if isinstance(tokens, tuple) else tokens
        metrics.latencies_ms.append(ms)
        metrics.num_tokens_per_request.append(len(text_codes))
    metrics.compute()
    return metrics


def bench_interleaved(
    model: object,
    prompts: list[object],
    max_tokens_list: list[int],
    seed: int,
    num_runs: int,
) -> InterleavedMetrics:
    """Alternate between different prompt lengths and max_tokens budgets."""
    import torch

    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph

    metrics = InterleavedMetrics()
    prev_n_steps = -1
    prev_plen = -1

    for i in range(num_runs):
        idx = i % len(prompts)
        input_ids = prompts[idx]
        max_tok = max_tokens_list[idx % len(max_tokens_list)]
        plen = input_ids.shape[1]

        decoder = enable_cuda_graph(model, n_steps=max_tok, max_len=plen + max_tok + 4)
        assert decoder is not None

        if plen != prev_plen or max_tok != prev_n_steps:
            metrics.recapture_count += 1 if i > 0 else 0
        prev_plen = plen
        prev_n_steps = max_tok

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.manual_seed(seed + i)
        decoder.generate_tokens(input_ids, seed=seed + i, return_audio=True)
        torch.cuda.synchronize()
        ms = _ms_since(t0)

        metrics.latencies_ms.append(ms)
        metrics.shapes.append(f"{plen}:{max_tok}")

    metrics.compute()
    return metrics


def cpu_contract_check() -> None:
    """Run without CUDA: verify capture count and replay count semantics."""
    # Verify n_steps - 1 capture semantics via source inspection
    import inspect

    from nanovllm_omni.engine import cuda_graph as cg

    src = inspect.getsource(cg.CudaGraphDecoder._capture)
    assert (
        "num_decode_graphs = max(self.n_steps - 1, 0)" in src
    ), "Capture loop should use n_steps - 1"

    src_gen = inspect.getsource(cg.CudaGraphDecoder.generate_tokens)
    assert "for k in range(self.n_steps - 1):" in src_gen, "Replay loop should iterate n_steps - 1"

    # Verify recapture triggers
    import types

    class StubAttn:
        _kv_pos = 0
        _nanovllm_kv_buffer = True

    class StubModel:
        config = types.SimpleNamespace(audio_pad_token=2051)

        def modules(self):
            yield self
            yield StubAttn()

    dec = object.__new__(cg.CudaGraphDecoder)
    dec.model = StubModel()
    dec.n_steps = 16
    dec._captured = True
    dec._captured_len = 10
    dec._captured_n_steps = 16
    dec._prefill_len = 10

    assert not dec._needs_recapture(), "same shape should reuse"
    dec._prefill_len = 20
    assert dec._needs_recapture(), "different prompt len should recapture"
    dec._prefill_len = 10
    dec.n_steps = 8
    assert dec._needs_recapture(), "different budget should recapture"

    print("[CPU] Contract checks passed:")
    print("  - Capture count: n_steps - 1 graphs (not n_steps)")
    print("  - Replay count:  n_steps - 1 iterations")
    print("  - Recapture triggers: prompt length change, budget change")
    print("  - Position-dependent: each graph bakes KV offset (by design)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold/hot CUDA Graph benchmark")
    parser.add_argument("--model", default="/home/mcig/minimind-3o")
    parser.add_argument("--mimi", default="/home/mcig/mimi")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--hot-runs", type=int, default=20)
    parser.add_argument("--interleaved-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        cpu_contract_check()
        print("\nNo CUDA. Run on RTX 3050 (WSL):")
        print("  cd ~/nanovllm-omni && source ~/venvs/nanovllm-omni/bin/activate")
        print(
            f"  HF_HUB_OFFLINE=1 python tools/bench_cold_hot_cuda_graph.py "
            f"--model {args.model} --mimi {args.mimi} "
            f"--max-tokens {args.max_tokens} --hot-runs {args.hot_runs}"
        )
        return 0

    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    print(f"Loading model from {args.model} ...", flush=True)
    bundle = load_minimind_omni_bundle(model_id=args.model, mimi_model_id=args.mimi)
    model = bundle.model
    tokenizer = bundle.tokenizer

    prompts_text = [
        "Hello, how are you?",
        "Tell me about the weather today.",
        "What is the meaning of life? Please explain in detail.",
    ]
    prompts = [tokenizer(p, return_tensors="pt").input_ids.cuda() for p in prompts_text]
    prompt_lens = [p.shape[1] for p in prompts]
    print(f"Prompt lengths: {prompt_lens}", flush=True)

    primary_ids = prompts[0]
    primary_plen = prompt_lens[0]

    # --- Eager baseline ---
    print("\n=== Eager baseline ===", flush=True)
    eager_times = []
    for i in range(3):
        ms, nf = _run_eager(model, primary_ids, args.max_tokens, args.seed + i)
        eager_times.append(ms)
        print(f"  eager run {i}: {ms:.2f} ms, {nf} frames", flush=True)
    eager_median = sorted(eager_times)[1]

    # --- Cold path ---
    print("\n=== Cold path (first request, includes capture) ===", flush=True)
    decoder = enable_cuda_graph(
        model, n_steps=args.max_tokens, max_len=primary_plen + args.max_tokens + 4
    )
    assert decoder is not None
    cold = bench_cold(decoder, primary_ids, args.seed)
    print(f"  total cold: {cold.total_cold_ms:.2f} ms", flush=True)
    print(f"  graphs captured: {cold.num_graphs} (n_steps-1 = {args.max_tokens - 1})", flush=True)
    print(f"  warmup forwards: {cold.num_warmup_forwards}", flush=True)

    # --- Hot path ---
    print(f"\n=== Hot path ({args.hot_runs} requests, graphs reused) ===", flush=True)
    hot = bench_hot(decoder, primary_ids, args.seed, args.hot_runs)
    print(f"  p50: {hot.p50_ms:.2f} ms", flush=True)
    print(f"  p95: {hot.p95_ms:.2f} ms", flush=True)
    print(f"  mean: {hot.mean_ms:.2f} ms", flush=True)
    print(f"  tokens/request: {hot.num_tokens_per_request[:5]}...", flush=True)

    # --- Amortisation ---
    capture_overhead = cold.total_cold_ms - hot.p50_ms  # approximate
    if hot.p50_ms < eager_median and capture_overhead > 0:
        saving_per_req = eager_median - hot.p50_ms
        amort = int(capture_overhead / saving_per_req) + 1
    else:
        amort = -1  # graph not faster or no overhead
    print("\n=== Amortisation ===", flush=True)
    print(f"  eager median: {eager_median:.2f} ms", flush=True)
    print(f"  graph hot p50: {hot.p50_ms:.2f} ms", flush=True)
    print(f"  capture overhead (cold - hot_p50): {capture_overhead:.2f} ms", flush=True)
    if amort > 0:
        print(f"  breakeven after: {amort} requests", flush=True)
    else:
        print("  breakeven: N/A (graph not faster than eager)", flush=True)

    # --- Interleaved shapes ---
    print(
        f"\n=== Interleaved shapes ({args.interleaved_runs} requests) ===",
        flush=True,
    )
    max_tokens_variants = [args.max_tokens, args.max_tokens + 8, args.max_tokens]
    interleaved = bench_interleaved(
        model, prompts, max_tokens_variants, args.seed, args.interleaved_runs
    )
    print(f"  recaptures: {interleaved.recapture_count}", flush=True)
    print(f"  p50: {interleaved.p50_ms:.2f} ms", flush=True)
    print(f"  p95: {interleaved.p95_ms:.2f} ms", flush=True)
    print(f"  shapes: {interleaved.shapes[:10]}...", flush=True)

    torch.cuda.reset_peak_memory_stats()
    _, _ = _run_graphed(decoder, primary_ids, args.seed)
    vram = _vram_mb()

    result = BenchResult(
        cold=cold,
        hot=hot,
        interleaved=interleaved,
        eager_baseline_ms=eager_median,
        vram_peak_mb=vram,
        amortisation_requests=amort,
    )

    print("\n=== Summary ===", flush=True)
    print(f"  VRAM peak: {vram:.1f} MB", flush=True)
    print(
        f"  Eager: {eager_median:.2f} ms | "
        f"Graph cold: {cold.total_cold_ms:.2f} ms | "
        f"Graph hot p50: {hot.p50_ms:.2f} ms",
        flush=True,
    )
    if amort > 0:
        print(f"  Capture cost amortised after {amort} requests", flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
        print(f"  Written to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
