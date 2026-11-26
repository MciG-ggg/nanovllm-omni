# Apples-to-apples bench — measured on RTX 3050 4GB Laptop

> **Hardware**: NVIDIA GeForce RTX 3050 4GB Laptop GPU, 53°C / 1725 MHz at start
> **Software**: torch 2.14.0+cu130, commit `3efc9a0`
> **Protocol**: `docs/perf/bench-protocol-aligned.md`
> **Date**: 2026-09-07
> **Runs**: 6 prompts × 20 runs × max_tokens=16, seed=42, temperature=0.7, top_p=0.9

## 4-cell matrix

| Cell | fusion | thinker CUDA Graph | talker CUDA Graph | median range | median-of-medians | vs A |
|---|---|---|---|---:|---:|---:|
| **A** baseline | off (`--enforce-eager`) | off | on (deploy YAML default) | 941–1240 ms | 984.6 ms | 1.00× |
| **B** fusion-only | on | off | on (deploy YAML default) | 804–946 ms | 872.9 ms | **1.13×** (−11.3%) |
| **C** fusion + thinker graph | on | on (`--use-thinker-cuda-graph`) | on | **CRASHED** | — | — |
| **D** full stack | on | on | on | **NOT RUN** (same crash as C) | — | — |

## Per-prompt A vs B

| prompt | A median (ms) | B median (ms) | delta | speedup |
|---|---:|---:|---:|---:|
| medium_01 | 1240.4 | 946.2 | −294.2 | 1.31× |
| medium_02 | 1014.2 | 930.9 | −83.3 | 1.09× |
| short_01 | 945.9 | 805.9 | −140.0 | 1.17× |
| short_02 | 941.0 | 803.7 | −137.3 | 1.17× |
| short_03 | 1151.8 | 826.8 | −324.9 | 1.39× |
| system_01 | 954.9 | 919.0 | −35.9 | 1.04× |
| **median-of-medians** | **984.6** | **872.9** | **−111.7** | **1.13×** |

Raw CSVs: `docs/perf/aligned/A-baseline.csv`, `docs/perf/aligned/B-fusion-only.csv`.

## What this means

**Fusion-only E2E improvement is 1.13× (−11.3%)**, not the 7.51× / ~20× numbers from the thinker-decode primitive.

The 7.51× (320 → 42 ms) and ~20× (846 → 42 ms) numbers are **thinker-decode primitive only** — they exclude talker MTP, Mimi decode, tokenize, and WAV encode. Those stages together dominate the remaining wall-clock after fusion is applied. Profile-detail earlier showed generate is ~95% of E2E, but generate includes talker MTP which is **not** fused (talker uses a different class name than `Attention` / `FeedForward`).

Cell A vs Cell B is the **honest fusion-only E2E number**: same script, same hardware, same torch, same prompt set, same seed. The only difference is `--enforce-eager`.

## Cell C crash (known gap)

`--use-thinker-cuda-graph --pipeline full` crashes with:

```
aten/src/ATen/native/cuda/TensorCompare.cu:121: Assertion
`probability tensor contains either inf, nan or element < 0` failed.
```

This is the known gap documented in `bench-protocol-aligned.md` §4:

1. `Omni.generate` does not receive `--use-thinker-cuda-graph` (bench `_kwargs` pops it)
2. CUDA Graph leftover state from previous runs can poison subsequent runs
3. Talker CUDA Graph (deploy YAML default `true`) forces greedy decode, which interacts badly with the leftover state

**Cell D was not run** — it would hit the same crash.

## Comparison to previous numbers

| Previous number | What it actually was | This measurement |
|---|---|---|
| ~928 ms E2E "baseline" | project-statement target, not measured | **A: 941–1240 ms** (real baseline) |
| 846.38 ms | thinker-decode primitive, torch 2.13 | not re-measured (different scope) |
| 320 ms | thinker-decode primitive post-fusion | not re-measured (different scope) |
| ~42 ms / 7.51× | thinker-decode primitive + CUDA Graph | **C crashed** (Omni.generate doesn't thread the flag) |
| 628–1038 ms E2E | fusion-only, torch 2.14, commit `8f6d522` | **B: 804–946 ms** (same scope, different commit, GPU warmer after Cell A) |

The 628–1038 ms number is **not reproduced** under this protocol (B is 804–946). Possible causes: different commit (`8f6d522` vs `3efc9a0`), GPU thermal (Cell B ran after Cell A warmed the GPU), different sampling params. This is why the protocol exists — previous numbers were not apples-to-apples.

## Resume-ready numbers (after this measurement)

**Use these, not the previous ones:**

> Full three-stage E2E (thinker → talker → MTP → Mimi → WAV) on RTX 3050 4GB: unfused baseline 941–1240 ms (median-of-medians 985 ms) → fused 804–946 ms (median-of-medians 873 ms), **1.13× / −11.3%**. Thinker-decode primitive (not user-facing) separately measured at 846 → 42 ms (~20×) with CUDA Graph, but that number excludes talker MTP / Mimi / tokenize which dominate remaining wall-clock.

**Do not write:**
- "7.51× E2E" — that's thinker-decode primitive only
- "320 → 42 ms" as the headline — same reason
- "628–1038 ms" as current state — not reproduced under this protocol
- Any talker-only speedup — not measured (Cell C/D crashed)

## 8-cell Graph diagnostic (not resume-eligible)

A later isolated-process sweep was run on the same RTX 3050 / torch 2.14.0+cu130
protocol: fusion on/off x `{none, thinker, talker, both}`, 6 prompts x 20 runs,
`max_tokens=16`, seed 42, temperature 0.7, top-p 0.9. Raw CSVs are in
`docs/perf/aligned/graph-sweep-v3/`.

| fusion | Graphs | median-of-prompt-medians |
|---|---|---:|
| off | none | 873.0 ms |
| off | thinker | 540.0 ms |
| off | talker | 871.6 ms |
| off | both | 514.0 ms |
| on | none | 885.8 ms |
| on | thinker | 535.4 ms |
| on | talker | 858.2 ms |
| on | both | 511.7 ms |

**Do not compare the thinker/both rows against the eager rows.** Eager cells emitted
9/10/15 Mimi frames according to prompt; thinker-graph cells emitted 15 frames for
every sample. Matching temperature/top-p and resetting per-request EOS state did
not restore the deploy `post_eos_padding_count` / internal-stop bridge behavior.
Talker graph also forces greedy decode. These are diagnostic timings only.

The shipped follow-up (`364ca1a`) restores the full pipeline's eager thinker bridge
path and makes `bench time --pipeline full --use-thinker-cuda-graph` fail loudly
until a graph decoder implements equivalent post-EOS bridge-state capture. The
valid resume number therefore remains the fusion-only 1.13x result above.
