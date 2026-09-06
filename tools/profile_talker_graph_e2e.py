#!/usr/bin/env python3
"""E2E test of talker CUDA Graph on real weights (RTX 3050).

Drives Omni.generate end-to-end with use_talker_cuda_graph on, verifies
the WAV is valid, and reports wall time + per-stage breakdown. Compares
against use_talker_cuda_graph: false (eager talker) to quantify the saving.

Usage:
    HF_HUB_OFFLINE=1 python tools/profile_talker_graph_e2e.py \
        --model ~/minimind-3o --mimi ~/mimi --runs 3
"""

from __future__ import annotations

import argparse
import io
import statistics
import time
import wave
from pathlib import Path

import torch


def _assert_wav(payload: bytes) -> None:
    assert payload.startswith(b"RIFF"), f"not a WAV: {payload[:8]!r}"
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24_000


def _time_generate(omni, prompt: str, runs: int) -> tuple[float, bytes]:
    """(median_ms, last_payload_bytes)."""
    times: list[float] = []
    last = b""
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = omni.generate([prompt], None)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)
        last = outs[0].multimodal_output["audio"].wav_bytes()
    return statistics.median(times), last


def _write_deploy(use_talker_graph: bool, path: Path) -> None:
    # Use realistic MiniMind-O defaults so stage 1 dominates.
    path.write_text(
        f"max_batch: 1\n"
        f"use_cuda_graph: false\n"
        f"use_talker_cuda_graph: {str(use_talker_graph).lower()}\n"
        f"post_eos_padding_count: 128\n"
        f"internal_stop_token_id: 17\n"
        f"talker_max_steps_after_last_thinker_token: 192\n"
        f"stages:\n"
        f"  - name: thinker\n"
        f"    max_num_batched_tokens: 512\n"
        f"    default_sampling_params: {{temperature: 0.7, max_tokens: 512}}\n"
        f"  - name: talker\n"
        f"    default_sampling_params: {{temperature: 0.2, watchdog_limit: 192}}\n"
        f"  - name: code2wav\n"
        f"    enforce_eager: true\n"
        f"    default_sampling_params: {{}}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/mcig/minimind-3o")
    parser.add_argument("--mimi", default="/home/mcig/mimi")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2

    from nanovllm_omni import Omni
    from nanovllm_omni.models.minimind_omni import create_bundle

    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    print(f"model={args.model}, mimi={args.mimi}")
    print()

    # Load real bundle (real weights)
    bundle = create_bundle(model_id=args.model, mimi_model_id=args.mimi)

    # --- Eager (use_talker_cuda_graph: false) ---
    deploy_eager = Path("/tmp/minimind_eager.yaml")
    _write_deploy(False, deploy_eager)
    omni_eager = Omni(
        args.model,
        device="cuda",
        dtype="float16",
        deploy_config_path=str(deploy_eager),
        extra={"bundle": bundle},
    )
    eager_med, eager_wav = _time_generate(
        omni_eager,
        "Hello, how are you?",
        args.runs,
    )
    _assert_wav(eager_wav)
    print("--- EAGER talker (use_talker_cuda_graph: false) ---")
    print(f"  median wall (over {args.runs}): {eager_med:.1f} ms")
    print(f"  WAV bytes: {len(eager_wav)}")
    print()

    # --- Graph (use_talker_cuda_graph: true) ---
    deploy_graph = Path("/tmp/minimind_graph.yaml")
    _write_deploy(True, deploy_graph)
    omni_graph = Omni(
        args.model,
        device="cuda",
        dtype="float16",
        deploy_config_path=str(deploy_graph),
        extra={"bundle": bundle},
    )
    graph_med, graph_wav = _time_generate(
        omni_graph,
        "Hello, how are you?",
        args.runs,
    )
    _assert_wav(graph_wav)
    print("--- GRAPHED talker (use_talker_cuda_graph: true) ---")
    print(f"  median wall (over {args.runs}): {graph_med:.1f} ms")
    print(f"  WAV bytes: {len(graph_wav)}")
    print()

    # --- Comparison ---
    saved = eager_med - graph_med
    pct = 100 * saved / max(eager_med, 0.001)
    print("--- Comparison ---")
    print(f"  Eager:    {eager_med:.1f} ms")
    print(f"  Graphed:  {graph_med:.1f} ms")
    print(f"  Saved:    {saved:.1f} ms ({pct:+.1f}%)")
    if saved > 0:
        print(f"  Graph is {eager_med / max(graph_med, 0.001):.2f}x faster")
    else:
        print(
            "  (Graph was not faster on this config; "
            "possibly due to capture overhead on small max-tokens)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
