# flash-attn on RTX 3050 4GB: empirical verdict (Sep 2026)

## Question

Does installing real `flash-attn` (the Astral prebuilt wheel against
torch 2.12 + cu130 + cp312) measurably beat torch-native SDPA on the
paged single-graph decode loop? Earlier speculation said "yes, 2-3×";
this run answers it with numbers.

## Setup

- Hardware: NVIDIA GeForce RTX 3050 4GB Laptop GPU, capability (8, 6)
- torch: **2.12.0+cu130** (cp312 wheel from PyPI's cu130 index)
- flash-attn: `2.8.3.post1+cu.13.0.torch.2.12-cp312-cp312-manylinux_2_24_x86_64`
  prebuilt wheel from `https://wheels.astral.sh/simple/cu130/flash-attn/`
  -- **zero compilation**, both packages installed in ~30 seconds
- block_size=256 (forced by `flash_attn_varlen_func`'s block-size
  divisibility check; matches the fork's `Sequence.block_size` default)
- Same model (MiniMind-O), same prompts, same seed, same 10 reps as
  every other bench in this directory. CSV: `torch212-fa-v1/cells.csv`
  vs `torch212-native-v1/cells.csv`.

The bench script's `--no-flash` flag sets an env var
(`NANOVLLM_DISABLE_FLASH=1`) that forces the adapter to take the
torch-native SDPA branch even when flash-attn is importable. This is
the only way to compare **kernel paths on the same torch version**;
comparing torch 2.12 against torch 2.14 confounds the variable.

## Result

| cell | torch-native | flash-attn | delta |
|---|---|---|---|
| eager | 812.3 ms | 795.8 ms | -2.0% (noise) |
| perpos | 183.4 ms | 189.0 ms | +3.0% (noise) |
| **paged** | **199.1 ms** | **202.7 ms** | **+1.8% (noise)** |

frame counts identical (16/16 for both graph cells, 8 for eager --
the eager-vs-graph stopping difference is unchanged by SDPA backend).

**flash-attn does not win on this workload.** All deltas are within
+/- 3%, well inside bench noise.

## Why

The paged decode window is small (`prompt_len + max_new_tokens ≈ 36`,
rounded up to one 256-token block). At `seq_k=256` with batch=1 and
12 heads × 64 dim, the attention math is GEMM-bound and tiny. Under
that shape, cuDNN's `scaled_dot_product_attention` **already dispatches
to flash-attn's kernels internally** -- the Astral wheel and cuDNN's
path do the same arithmetic through different ABIs.

flash-attn's wins show up at large `seq_k` (2k+) where its
work-partitioning beats cuDNN's heuristic. Our window is 12× smaller
than that, so the cuDNN heuristic is already optimal for the shape
we have.

## What this changes in the project

- The torch-native paged SDPA path we already shipped **stays**.
- `block_size` defaults bumped from 16 to 256 because flash-attn's
  varlen kernel asserts divisibility; matching the fork's default is
  free.
- `enable_paged_kv_cache`'s `flash_attn_available()` routing still
  works -- if anyone runs this on a workload where the window grows
  past 1k tokens, flash-attn will kick in and the bench will probably
  tell a different story.
- Resume / blog wording about "torch-native flash-attn fallback" was
  speculative; the measured numbers say the fallback never matters.

## What this did NOT change

- 6.36× thinker decode / 3.61× full E2E speedups (from the per-position
  graph + paged single-graph work) are independent of the SDPA
  backend. Those come from graph replay, not kernel choice.
- The `paged-v2/cells.csv` numbers (torch 2.14 + native, 233.7 ms hot
  p50) are from a previous torch version; the torch 2.12 + native
  numbers here (199.1 ms) show torch 2.12 itself is ~15% faster than
  torch 2.14 on this workload, but that is a torch-version delta, not
  a kernel-path delta.
