# Phase 2 — SmolVLM CUDA Graph Parity (RESOLVED)

**Status: eager baseline ran clean. CUDA graph capture now works after fixing two upstream bugs in `third_party/nano-vllm`. Parity OK — eager and CG produce identical decoded text on the short prompt.**

Date: 2026-09-18
WSL repo: `/mnt/d/edge-inference/nanovllm-omni/nanovllm-omni` @ `a6333740666c9201f0fa45521d4aff6f05a7614b` (`feat(bench): --use-cuda-graph flips smolvlm decode eager (default unchanged)`)
GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU, driver 591.86

## Submodule fixes (third_party/nano-vllm)

Two bugs blocked CUDA graph capture on the SmolVLM decode path. Both were inside the `third_party/nano-vllm` submodule (fork upstream of `nanovllm` that this repo depends on); neither was in our code.

### Bug 1 — host sync inside graph capture (`attention.py:74`)

`_gather_paged_kv_decode` called `int(context_lens.max().item())` to size the gather buffer. `.item()` forces a CUDA→host sync, which is illegal inside `torch.cuda.graph()` capture (`cudaErrorStreamCaptureUnsupported`); the failure then poisons `capture_end` itself with `cudaErrorStreamCaptureInvalidated` during cleanup.

**Fix**: precompute `S_max` on the host side, store on the `Context` dataclass as a Python int, read from `get_context().s_max` inside the gather.

- `third_party/nano-vllm/nanovllm/utils/context.py` — added `s_max: int = 0` to `Context`; `set_context` accepts `s_max=...`.
- `third_party/nano-vllm/nanovllm/engine/model_runner.py` — `prepare_decode` computes `s_max = max(context_lens, default=0)` from the host-side Python list before moving tensors to GPU; `capture_cudagraph` sets `s_max = max_num_blocks * self.block_size` so the captured graph uses the static capacity bound.
- `third_party/nano-vllm/nanovllm/layers/attention.py` — `_gather_paged_kv_decode` reads `get_context().s_max`; if `s_max == 0` (legacy callers), falls back to the old `.item()` path so unmigrated code keeps working.

### Bug 2 — graph capture OOM on 4GB GPU (`model_runner.py:329`)

After Bug 1 was fixed, capture ran but crashed at `v_cts.repeat_interleave(...)` with `cudaErrorMemoryAllocation`. Root cause: `capture_cudagraph` enumerated batch sizes `[1, 2, 4, 8] + list(range(16, max_bs + 1, 16))` with `max_bs = min(max_num_seqs, 512) = 512`. Each captured graph holds a static-sized k/v gather of shape `[bs, H_kv, S_max, D]` across all 24 layers. With `S_max = 4096`, `H_q = 15`, `H_kv = 5`, `D = 64`, `bs = 512` would have needed ~25 GB per graph — way over the 4 GB GPU.

**Fix**: cap graph capture at bs=8 by default; expose `Config.max_capture_bs` so multi-request workloads can raise it on bigger hardware.

- `third_party/nano-vllm/nanovllm/engine/model_runner.py` — `self.graph_bs` now derived from `getattr(config, "max_capture_bs", 8)`; the up-to-512 list only materialises if that knob is raised.

## Eager vs CUDA-graph wall (10 runs × 3 inputs, commit `a633374`)

| input | max_new_tokens | eager p50 (ms) | eager p99 (ms) | CG p50 (ms) | CG p99 (ms) | speedup p50 | CG peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| smolvlm_short | 64 | 219.251 | 247.143 | 129.491 | 135.552 | **1.7×** | 2190.274 |
| smolvlm_medium | 64 | 5180.159 | 5522.421 | 675.501 | 1730.953 | **7.7×** | 2190.677 |
| smolvlm_long | 64 | 5170.739 | 5670.904 | 1541.556 | 1589.966 | **3.4×** | 2191.799 |

- Eager CSV: `docs/perf/bench_smolvlm_a633374_eager.csv`
- CG CSV:    `docs/perf/bench_smolvlm_a633374_cg.csv`
- Eager trace: `docs/perf/smolvlm-2026-09-18.trace.json.gz` (in-repo path; the bench script's date-only naming means a same-day CG run would overwrite, so the CG run was redirected to `/tmp/bench_smolvlm_cg.log` only — only the eager trace needed for downstream kernel analysis).
- Eager env: `docs/perf/smolvlm-2026-09-18.env.txt`

CG speedup is largest on `medium` (7.7×) where the eager Python-level per-token loop dominates; smallest on `short` (1.7×) where warmup overhead is a larger share of total time. The pattern (per-token launch overhead being the eager bottleneck) matches the Kineto top-kernels from Phase 1: aten::linear, aten::matmul, gemv2T, aten::clone — all called once per token per layer in eager mode.

CG p99 on `medium` (1730 ms) is higher than p50 (676 ms); suspect CUDA graph pool fragmentation on first replay after warmup. Within Phase 2 follow-up scope if it shows up in user-facing inference.

## Parity check (Task 1c)

`/tmp/parity_smolvlm.py` on WSL loads the eager pipeline and the CG pipeline in the same process, runs both on `smolvlm_short` with `max_new_tokens=64`, and asserts decoded text matches.

```
eager(7): ' Paris.'
cg   (7): ' Paris.'
parity: OK
```

Identical decoded string. The static `s_max = max_num_blocks * block_size` upper bound with the `valid` mask in `_gather_paged_kv_decode` does not perturb numerical outputs for the captured workload.

## Task 2 — sd-turbo / smolvla weights

**weights missing, deferred.** `/mnt/d/models/` on WSL still contains only `SmolVLM-500M-Instruct`. User confirmed they'll rsync the other two caches from a local Mac download when ready; pending.

## Files on WSL

- `/tmp/bench_smolvlm_eager.csv` — same as `docs/perf/bench_smolvlm_a633374_eager.csv`
- `/tmp/bench_smolvlm_cg.csv` — same as `docs/perf/bench_smolvlm_a633374_cg.csv`
- `/tmp/parity_smolvlm.py` — parity script
- `/tmp/phase2-done.txt` — one-line result: `eager=219.251 cg=129.491 parity=OK`

## Open follow-up

1. **Submodule rebase**: both submodule edits are local changes; rebasing on upstream will require re-applying. Either: (a) commit them on a fork branch and point the submodule there, or (b) patch in `nanovllm_omni/engine/stage_runner.py` instead so the submodule stays pristine. Decision needed before pushing.
2. **`max_capture_bs` knob**: currently in code with default 8. Should become a `Config` dataclass field once the fork rebase story is settled.
3. **CG p99 on `medium`**: jump from 676 → 1730 ms; investigate first-replay-vs-rest variance.
4. **sd-turbo VAE decomposition** and **smolvla flow-step sweep** are still gated on weights arriving in `/mnt/d/models/`.

Done-file: `eager=219.251 cg=129.491 parity=OK`.
