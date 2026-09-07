# Profiler originals (2026-09-07)

Raw traces from the WSL 3050 + Colab T4 rerun. No post-processing.

## torch.profiler / Kineto (WSL RTX 3050)

| file | what |
|---|---|
| `profile-detail.trace.json.gz` | Chrome/Kineto trace, `short_03`, warmup=1, runs=1, max-tokens=16 |
| `profile-detail.detail.md` | `parse_kineto_trace` output: generate wall 1.29s, 9689 `cudaLaunchKernel` |
| `minimind-2026-08-27.trace.json.gz` | leftover 6-prompt × 32-step run |
| `minimind-2026-08-27.detail.md` | parsed leftover |

## nsys (WSL RTX 3050)

| file | what |
|---|---|
| `trace-nsys.nsys-rep` | before `stage()` — NVTX only CUB internals |
| `trace-nsys.stats.txt` | CUDA API sum: `cudaLaunchKernel` 10118 / 36.6% |
| `trace-nsys.nvtx.txt` | CUB-only NVTX |
| `trace-nsys-v3.nsys-rep` | after `stage()` — NVTX shows `:tokenize` / `:generate` / `:decode` / `:wav` / `:generate.step` |
| `trace-nsys-v3.nvtx.csv` | `nvtx_sum` CSV of v3 |
| `trace-nsys-v3.csv` | bench CSV of the inner run |

WSL CUPTI still skips GPU kernel timing (`cuda_gpu_kern_sum SKIPPED`).

## ncu

WSL 3050 cannot collect counters (`ERR_NVGPUCTRPERM` in `ncu-gemv.log`).
Colab T4 originals live in `ncu-colab/`:

| file | kernel regex | first-instance SOL |
|---|---|---|
| `ncu-colab/k1-fmha.ncu-rep` | `.*fmha_cutlass.*` | Compute 3.19% / Memory 2.99% / grid 8 |
| `ncu-colab/k2-gemv.ncu-rep` | `.*gemv2T.*` | Compute 22.13% (later instance 46.7% / DRAM 52.6%) |
| `ncu-colab/k3-direct-copy.ncu-rep` | `.*unrolled_elementwise_kernel.*` | Compute 0.09% / grid 1 |
| `ncu-colab/k4-cat.ncu-rep` | `.*CatArrayBatched.*` | Compute 0.07% / grid 2 |
| `ncu-colab/summary.csv` | first-instance metrics extracted via `ncu --import` | |

See `ncu-generate-kernels-2026-09-01.md` §36 for the write-up.
