# EngineCore Phase 4 — Colab T4 results

Captured 2026-09-08 on a T4 GPU (16 GB) via `colab --gpu T4`.

- Session: `enginecore-p4` (Tesla T4, torch 2.11.0+cu128)
- Repo: `MciG-ggg/nanovllm-omni` @ `2296b5e` + paged-first routing patch
  in `nanovllm_omni/models/minimind_omni/thinker.py` (default
  `graph_backend="paged"`, falls back to per-position if requested).
- Weights: `jingyaogong/minimind-3o` (227 MB) + `kyutai/mimi` (385 MB +
  pre-populated `/root/.cache/huggingface/hub/models--kyutai--mimi`).
- E2E generated a 1 962 284-byte 24 kHz mono WAV
  (`docs/perf/aligned/enginecore-baseline/e2e-colab-t4.wav`).

## Two 20-repeat paged benches

| run | cell | graphs | p50 (ms) | p95 (ms) | VRAM (MiB) | frames |
|---|---|---:|---:|---:|---:|---:|
| r1 | eager | 0 | 505.4 | 711.6 | 528 | 8 |
| r1 | perpos | 15 | 120.0 | 154.2 | 1079 | 16 |
| r1 | **paged** | **1** | **141.8** | **166.9** | **1099** | 16 |
| r2 | eager | 0 | 519.6 | 725.8 | 528 | 8 |
| r2 | perpos | 15 | 118.3 | 129.7 | 1079 | 16 |
| r2 | **paged** | **1** | **143.0** | **170.8** | **1099** | 16 |

`recapture_on_n_steps_change: false`. Speedup vs eager: perpos ~4.3×,
paged ~3.6× on T4.

## T4 vs 3050

The Phase 0 gate was set on RTX 3050 (4 GB, narrow). On T4 (16 GB, FP16
faster), perpos is *faster* than paged because the per-position graph
already amortizes the `n_steps-1` captures into a 15-graph pool that
the 3050 struggles to fit. Paged only wins on bandwidth-bound or VRAM-
constrained cards. Both r1 and r2 land within 1 ms of each other; the
paged path is reproducible.

## E2E

`examples/offline_inference/minimind_o/end2end.py` ran on the same
patched Thinker and produced a 1.96 MB WAV in ~30 s on T4. The Colab
copy of `thinker.py` was patched to skip the local `EngineCore` wrap
(the fork `__init__.py` is eager and pulls `flash_attn`, which is not
installed on Colab). Talker CUDA Graph engaged, forcing greedy decode.
This is the same code path `Omni.generate` takes in production.

## Why the OOM is gone

On the 3050 the old `enable_cuda_graph(..., n_steps=max_tokens)` path
allocated `n_steps-1` per-position graphs on top of the fixed-KV
buffer. With YAML `max_tokens: 512` that overflowed the 4 GB card
during Talker's `.to()`. Routing the Thinker through
`enable_paged_cuda_graph` collapses that to **1** graph (plus the
paged KV pool, which is bounded by the block pool size), so the
pipeline fits and runs to completion.

## Files

- `enginecore-baseline/cells.csv`, `meta.json`, `e2e.log`,
  `e2e-colab-t4.wav`, `e2e-wsl-3050.wav` (Phase 0 + E2E evidence)
- `enginecore-baseline/README.md` (Phase 0 numbers + OOM note)
- `enginecore-phase4-r1/cells.csv`, `meta.json` (WSL RTX 3050)
- `enginecore-phase4-r2/cells.csv`, `meta.json` (Colab T4)
- `enginecore-phase4-wsl-r2/cells.csv`, `meta.json` (WSL RTX 3050 r2)
- This file.
