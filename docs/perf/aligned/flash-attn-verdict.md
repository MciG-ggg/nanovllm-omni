# flash-attn on RTX 3050 4GB: empirical verdict (Sep 2026)

## Question

Does installing real `flash-attn` (the Astral prebuilt wheel against
torch 2.12 + cu130 + cp312) measurably beat torch-native SDPA on the
paged single-graph decode loop? Earlier speculation said "yes, 2-3×";
this run answers it with numbers at three regimes.

## Setup

- Hardware: NVIDIA GeForce RTX 3050 4GB Laptop GPU, capability (8, 6)
- torch: **2.12.0+cu130** (cp312 wheel from PyPI's cu130 index)
- flash-attn: `2.8.3.post1+cu.13.0.torch.2.12-cp312-cp312-manylinux_2_24_x86_64`
  prebuilt wheel from `https://wheels.astral.sh/simple/cu130/flash-attn/`
  -- **zero compilation**, both packages installed in ~30 seconds
- block_size=256 (forced by `flash_attn_varlen_func`'s block-size
  divisibility check; matches the fork's `Sequence.block_size` default)
- Same model (MiniMind-O), 3 prompts × N reps, seed 42, every bench in
  this directory. CSV names: `torch212-{fa,native}-n{16,128,512}/`.
- `--no-flash` sets an env var (`NANOVLLM_DISABLE_FLASH=1`) that
  forces the adapter to take the torch-native SDPA branch even when
  flash-attn is importable. This is the only way to compare kernel
  paths on the same torch version; comparing torch 2.12 against
  torch 2.14 confounds the variable.

## Result matrix (paged single-graph cell only)

Same torch (2.12), same model, same prompts -- kernel backend differs.

| n_steps | window (blocks × 256) | max seq_k | native p50 | flash p50 | delta | flash wins? |
|---|---|---|---|---|---|---|
| 16 | 1 block | ~36 | 199.1 ms | 202.7 ms | +1.8% | no (noise) |
| 128 | 1 block | ~148 | 1220.1 ms | 1202.9 ms | -1.4% | no (noise) |
| 512 | 3 blocks | ~520 | 5738.1 ms | 5757.9 ms | +0.3% | no (noise) |

Per-step latency at n_steps=512: native 11.21 ms/step vs flash 11.25
ms/step. Perpos (fixed-KV, 127 capture cells) was also tried at
n_steps=128: native 9.5 ms/step vs flash 9.4 ms/step -- both within
noise. Frame counts identical across all (16/16 at 16 steps, 128/128
at 128 steps, 512/512 at 512 steps).

**flash-attn does not win on this workload at any tested regime.**
All deltas are within +/- 2%, well inside bench noise.

## Why

The paged decode window stays at one 256-token block for
prompt_len + n_steps up to ~250 tokens. cuDNN's
`scaled_dot_product_attention` **already dispatches to flash-attn's
kernels internally** at this shape -- the Astral wheel and cuDNN's
path do the same arithmetic through different ABIs.

flash-attn's wins show up at large `seq_k` (2k+) where its
work-partitioning beats cuDNN's heuristic. Even n_steps=512 with a
3-block window (~520 tokens) is 4x smaller than the regime where
flash-attn-vs-naive-SDPA diverges by a measurable amount.

This is also why the per-step cost at n_steps=512 (~11.2 ms) is not
~32x the per-step cost at n_steps=16 (~9.5 ms): the decode loop is
GEMM-bound at this model size, not attention-bound. The bottleneck is
the FFN/MLP and the q/k/v projections -- attention is a small slice.

## Cave- was

- **No batched serving tested.** The model + bench harness is B=1.
  flash-attn's paged kernel could win more cleanly at B>=2 where
  the cu_seqlens / block_tables path goes through a different
  heuristic. But that would require building a B>2 harness on top of
  the existing `PagedKVCache` (`max_batch_size` is already plumbed
  through), and no production path currently exercises it.
- **Window bounded by prompt + n_steps.** A request with
  `prompt_len=2048` would push the window to 9 blocks (~2.3k tokens)
  even at n_steps=16. The bench uses prompts of length 19--23; that
  is the realistic regime for this model, not a worst case.
- **No RTX 30xx / 40xx high-end data.** The bench hardware is the laptop
  RTX 3050 4GB. flash-attn's work-partitioning is most visible on
  A100 / H100 with large seq_k, neither of which is here.

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
