# EngineCore Phase 0 baseline (WSL RTX 3050)

Captured 2026-09-08 on `ssh mcigs-wsl` (`DESKTOP-Q29JAU0`).

- Host: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Torch: 2.14.0+cu130
- WSL repo: `/home/mcig/nanovllm-omni` @ `174d620` (`fix: zero KV buffer on every prefill entry`)
- This is **pre-EngineCore** code. Local macOS still has uncommitted Phase 1–3 work; it was **not** pushed, so this baseline is the old paged path.
- Command:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python tools/bench_paged_single_graph.py \
  --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
  --max-new-tokens 16 --repeats 20 \
  --out docs/perf/aligned/enginecore-baseline
```

## Numbers (ms, 20 repeats, 3 prompts)

| cell | graphs | p50 | p95 | vram_mb | frames |
|---|---:|---:|---:|---:|---:|
| eager | 0 | 1064.1 | 1208.5 | 527 | 8 |
| perpos | 15 | 267.6 | 317.3 | 1075 | 16 |
| **paged** | **1** | **183.9** | **199.4** | **1094** | 16 |

Paged recapture-on-n_steps-change: `false`. Speedup vs eager: paged 5.79×, perpos 3.98×.

Phase 4 gates vs this paged row: p50 ≤ 187.6 ms (+2%), p95 ≤ 209.4 ms (+5%), VRAM ≤ 1126 MiB (+32).

Eager `frames=8` vs graph `frames=16` is a real baseline quirk, not a copy error.

## Full MiniMind-O example

`examples/offline_inference/minimind_o/end2end.py --model /home/mcig/minimind-3o --mimi /home/mcig/mimi` **OOM** on 4GB when thinker+talker CUDA Graphs stay on (~2.56 GiB weights + ~1009 MiB graph pools). See `e2e.log`. The Phase 0 paged command is thinker-only and completed.
