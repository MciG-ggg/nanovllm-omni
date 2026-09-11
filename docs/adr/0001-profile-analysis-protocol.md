# ADR 0001: Profile-analysis protocol for sd-turbo / smolvlm / smolvla

- Status: Accepted
- Date: 2026-09-XX

We extend the minimind-omni profile methodology (Kineto + nsys on WSL RTX 3050, ncu SOL on Colab T4, output in `docs/perf/<model>-<date>.{detail.md,trace.json.gz,md}`) to the three new model families, with one structural change: **Phase 1 of each model is a baseline-only profile, not a pre-defined cell matrix**, because the minimind "36.6% CUDA-API time in `cudaLaunchKernel`" finding (launch-bound) does not transfer to sd-turbo / smolvlm / smolvla, and we have no profile data to assume otherwise for any of them.

## Protocol (4 phases per model)

1. **Phase 1 -- Baseline profile** (eager mode, no CUDA Graph, no fusion patches). Output: per-stage wall + top-kernel list (Kineto) + ncu SOL (T4).
2. **Phase 2 -- Bottleneck identification + cell design** (human review of Phase 1; cells targeting the bottleneck type).
3. **Phase 3 -- Wall-clock parity** vs `hf.generate` / `diffusers` reference. Always runs after Phase 2.
4. **Phase 4 -- Numerical parity** (token / pixel / action diff vs reference). On demand -- only when Phase 3 shows suspicious deviation.

Phases 2-4 may be deferred or skipped if Phase 1 shows no actionable bottleneck.

## Per-model input-set sizing

| Model     | Phase 1 size | Input count | Runs per input | Total measurements |
|-----------|--------------|-------------|----------------|--------------------|
| sd-turbo  | tiny         | 1           | 5              | 5                  |
| smolvlm   | standard     | 3           | 10             | 30                 |
| smolvla   | thorough     | 6           | 20             | 120                |

Justification: `sd-turbo` is 1-step diffusion -- wall variance is dominated by UNet + VAE-decode micro-cost, not by input distribution, so 5 runs of one prompt are enough. `smolvlm` is AR text gen, so prompt-length distribution affects prefill vs decode split, and 3 x 10 gives a usable scatter. `smolvla` is 2-stage (AR + flow), and the per-flow-step distribution is the bottleneck-sensitivity question, so we need 6 x 20 for statistical confidence on the per-step breakdown.

## Per-model NVTX markers

Each model gets its own NVTX range names (not a shared schema):

| Model     | NVTX ranges                                          |
|-----------|------------------------------------------------------|
| sd-turbo  | `:tokenize` / `:unet` / `:vae-decode`                |
| smolvlm   | `:tokenize` / `:vlm-prefill` / `:vlm-decode`         |
| smolvla   | `:tokenize` / `:vlm` / `:flow-step` / `:action-decode` |

Forcing a uniform schema would lose per-flow-step visibility in `smolvla`.

## Considered alternatives

- **Pre-defined cell matrix upfront** (the minimind 4-cell protocol). Rejected because it bakes in an optimization direction before any profile exists for the new models.
- **Single measurement per model** (no cells at all). Rejected because some bottlenecks need an on/off comparison to be observable (e.g. CUDA Graph's launch-overhead reduction).
- **Numerical parity always runs**. Rejected because the project's test suite already covers correctness; doubling it in profile work is wasted effort when wall-clock parity looks clean.

## Consequences

- Each model runs Phase 1 first and only -- no optimization work (CUDA Graph capture, fusion patches, etc.) for any of the three models until the profile identifies a bottleneck. Honors the project's "do not write unmeasured optimizations" rule.
- Three models can run Phase 1 in parallel (no cross-model data dependency); Phase 2 happens per-model after Phase 1.
- Lock-in cost is real: any change to the protocol (cell matrix timing, merging phases, dropping the per-model NVTX schema) requires a new ADR.

## Open

- LIBERO task + demo episode for `nanovllm_omni/bench/smolvla_prompts.py` (currently empty tuple with TODO).
- First execution order: sd-turbo (simplest profile) -> smolvlm -> smolvla (most complex).