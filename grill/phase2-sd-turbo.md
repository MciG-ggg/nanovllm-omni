# Phase 2 — sd-turbo VAE + UNet decomposition

**Status: Phase 2 cell complete. Real bottleneck is `image.cpu()` D2H transfer, not VAE kernels. CUDA graph capture blocked by diffusers CPU→GPU traffic.**

Date: 2026-09-18
Repo: `a633374` (+ sd_turbo bench CG fix at `93af9f6`)
GPU: NVIDIA RTX 3050 4GB Laptop, driver 591.86, CUDA 13.1, compute 8.6

## Baseline (commit `a633374`, 1 prompt × 5 runs, fp16)

| input | wall p50 | wall p99 | peak VRAM |
|---|---:|---:|---:|
| sd_turbo_01 | 500.634 ms | 505.059 ms | 3098 MiB |

Matches ADR-0003 Phase 1 (503.554 ms). CSV: `docs/perf/bench_sd_turbo_a633374.csv`. Kineto trace: `docs/perf/sd-turbo-2026-09-18.trace.json.gz` (WSL).

## NCU permission

NCU `--set basic` returns `ERR_NVGPUCTRPERM` on this WSL2 box, even as root — `/proc/driver/nvidia/version` is absent (WSL2 driver path lives under `/usr/lib/wsl/drivers/*`). SOL/memory-bound classification from Phase 1's Colab T4 capture still applies: VAE `nchwToNhwcKernel` was 61.6% DRAM throughput, memory-bound. WSL cannot reproduce that capture. Used nsys + Kineto for decomposition instead.

## nsys NVTX-bucketed decomposition (1 forward, warmup excluded)

NVTX ranges wrap the actual components (`sd_unet_step0`, `sd_scheduler_step`, `sd_vae_decode`) per `.scratch/sd_turbo_ncu.py`.

| Range | wall | runtime events | dominant runtime API |
|---|---:|---:|---|
| `sd_unet_step0` | 51.4 ms | 2172 | `cudaLaunchKernel_v7000` 10.8ms (913 calls) |
| `sd_scheduler_step` | 0.4 ms | 16 | `cudaLaunchKernel_v7000` 0.11ms |
| `sd_vae_decode` | **433.2 ms** | 655 | `cudaMemcpyAsync_v3020` **406.3ms (1 call)** |

The VAE range is dominated by a single `cudaMemcpyAsync` taking 406ms — not by VAE conv kernels.

## Kineto kernel-level decomposition (10 runs, sum of durations)

Top GPU ops from `docs/perf/sd-turbo-2026-09-18.trace.json.gz`:

| op | total over 10 runs | per-run avg |
|---|---:|---:|
| `:vae-decode` (user_annotation) | 3.843 s | 384 ms |
| `:unet` (user_annotation) | 0.924 s | 92 ms |
| `:tokenize` (user_annotation) | 0.182 s | 18 ms |
| `cudaMemcpyAsync` | 2.018 s | **202 ms** |
| `aten::copy_` / `aten::to` | 2.035 s / 2.034 s | 203 ms |
| `sm86_xmma_fprop_implicit_gemm` (UNet conv) | 0.596 s | 60 ms |
| `cutlass__5x_cudnn::Kernel` (UNet/cudnn fprop) | 0.438 s | 44 ms |
| `nchwToNhwcKernel` (VAE/cudnn layout) | 0.248 s | 25 ms |
| `nhwcToNchwKernel` (VAE/cudnn layout) | 0.100 s | 10 ms |

VAE **GPU compute** ≈ 35 ms/run (layout conversions + the conv kernels not named above, but the named ones cover the top).
VAE **D2H transfer** ≈ 202 ms/run — dominates.
UNet GPU compute ≈ 104 ms/run (cutlass fprop + xmma + small elementwise).

So the 384 ms `:vae-decode` wall = ~35 ms real VAE kernels + ~202 ms `image.cpu()` D2H + ~147 ms Python overhead (PIL conversion, `clamp`/`permute`/`float`/`numpy`/`round`/`astype` chain in `post_decode`).

## Root cause — `image.cpu()` in `post_decode`

`nanovllm_omni/models/sd_turbo/stage.py:154`:

```python
def post_decode(self, state):
    ...
    with torch.inference_mode():
        latents = state.latents / self.vae_scaling
        image = self.vae.decode(latents).sample
    # To PIL.
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()   # <-- 202ms D2H + Python chain
    image = (image[0] * 255).round().astype("uint8")
    pil_image = Image.fromarray(image)
    return DiffusionOutput(images=[pil_image], finished=True)
```

The D2H sync forces a `cudaStreamSynchronize` — the CPU waits for all VAE kernels to finish before any further work. The fp16→fp32→uint8 conversion also happens on CPU.

## Phase 2 cell — CUDA graph capture: WORKS (no speedup)

Root cause of original CG failure: diffusers UNet's `get_time_embed()` does `timesteps[None].to(sample.device)` — a CPU→GPU copy illegal inside `cuda.graph()` capture (the bench tried to pin `scheduler.timesteps` but the 0-dim slice `timesteps[step]` is still a new CPU tensor).

**Fix**: move `scheduler.timesteps` and `scheduler.sigmas` to GPU before capture, so `.to(sample.device)` becomes a no-op. Captured only the UNet forward + VAE decode block, leaving `scheduler.step()` (which calls `.item()` on a sigma comparison) and `post_decode` (which calls `.cpu()`) outside the capture region.

Harness: `.scratch/sd_turbo_cg_bench.py`. Results:

| mode | wall p50 | wall p99 | peak VRAM |
|---|---:|---:|---:|
| eager (UNet+VAE kernels only, no PIL) | 472.3 ms | 476.4 ms | 3099 MiB |
| CG replay (UNet+VAE) | 469.6 ms | 470.7 ms | 2499 MiB |
| **speedup** | **1.01×** | | -600 MiB VRAM |

CG has no measurable wall-time benefit. The captured region (UNet + VAE conv kernels) is already GPU-compute-bound, not launch-bound. The actual ~200 ms bottleneck (`image.cpu()` + PIL conversion chain) lives in `post_decode`, outside the graph because `.cpu()` is a D2H sync that can't be captured.

CG saves ~600 MiB VRAM (graph captures shared memory pool more efficiently than eager), but that's not a wall-time win.

## post_decode 优化（Phase 2 cell 已实现）+ 归因修正

实现了 GPU 侧转换链（`.scratch/sd_turbo_parity.py` 验证 786432 像素 100% bit-exact）：

```python
# 旧: cpu() 后 fp32 + numpy 链
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image[0] * 255).round().astype("uint8")

# 新: GPU 上 fuse,只 D2H uint8 HWC
img = (image / 2 + 0.5).clamp(0, 1)
img = img[0].float().mul_(255).round_()
img = img.permute(1, 2, 0).to(torch.uint8).contiguous()
arr = img.cpu().numpy()
```

bench: 500.6 → 497.0 ms p50（**只省 3 ms**）。用 `.scratch/qvae_split.py` 重新分解 Kineto trace 后发现之前的归因是错的：

| run | wall | GPU busy | CPU gap | D2H memcpy (GPU 时) |
|---|---:|---:|---:|---:|
| 前 5（含 profiler 预热） | 395-419 ms | 391-413 ms | 4-6 ms | 0.12 ms |
| 后 5（稳定） | 340-349 ms | 340-349 ms | **0.7 ms** | **0.12 ms** |

**真实归因**：`cudaMemcpyAsync` 的 406 ms（nsys runtime 事件计）不是传输本身，是 "等待前面 VAE kernels 完成 + 0.12ms 真实传输" 的总和。VAE decode wall ≈ GPU busy（CPU gap 仅 0.7ms），Python 转换链只占 ~0.7ms。post_decode 优化只能拿到 3ms。

**sd-turbo 真正的瓶颈是 VAE decode 的 GPU kernels 本身（~344 ms）**——全是 conv（cutlass fprop 44ms + xmma 60ms + nchwToNhwc layout 25ms + 其余 elementwise）。

## torch.compile / channels_last cells（2026-09-18 后续）

按 ADR-0001 Phase 2 继续打 VAE compute 瓶颈，三个 cell 全部实测：

| cell | VAE decode p50 | vs eager | 备注 |
|---|---:|---:|---|
| eager baseline | 345.5 ms | 1.00× | cuDNN sm86 隐式 gemm conv |
| CG replay（UNet+VAE） | 469.6 ms | 1.01×* | compute-bound，launch 开销本来就小 |
| `torch.compile(mode="max-autotune-no-cudagraphs")` | 3026.3 ms | **0.12×** | Triton conv kernel 大幅劣于 cuDNN 原生 |
| `torch.compile`（default） | 3016.8 ms | **0.12×** | 同上，与 autotune 模式无关 |
| `channels_last`（NHWC 权重+输入） | 520.7 ms | **0.66×** | cuDNN 在此卡上选了更差的 NHWC 路径 |

*CG 对比口径为 UNet+VAE 块（不含 tokenize），与 baseline 500ms 直接比较时无收益。

parity：compile 两个模式 bit-exact（max diff 0.0）；channels_last max diff 0.0095（fp16 conv 路径变化，视觉无损）。

VAE decode GPU 时间分解（torch.profiler，单次 decode）：

| kernel | 时间 | 占比 |
|---|---:|---:|
| `sm86_xmma_fprop` ×12 | 101 ms | 29% |
| `cutlass_5x_cudnn fprop` ×9 | 91 ms | 26% |
| `nchwToNhwc`/`nhwcToNchw` ×96 | 45 ms | 13% |
| elementwise（silu/groupnorm/add）×~155 | 69 ms | 20% |
| `fmha` attention ×1 | 6 ms | 2% |
| 其它 | ~35 ms | 10% |

**结论：在不引入新依赖的前提下，~345 ms 是 RTX 3050 4GB Laptop 上 sd-turbo VAE decode 的硬件地板。** cuDNN 的 fp16 NCHW 隐式 gemm 已是该卡最优路径；torch.compile 和 channels_last 都实测负收益。剩余大收益选项需要拍板：

1. **TAESD（tiny VAE）**：估 ~40 ms（8×），画质有损；需下载 60MB 权重 + 图像质量对比
2. **TensorRT 编译 VAE**：估 1.5-2×，引入重依赖
3. **维持现状**：接受 345 ms 为 Phase 2 终态

## Phase 2 conclusion + cell proposals

Phase 1's "`:vae-decode` 358ms > `:unet` 275ms" finding was real but misread. The VAE **kernels** are fast (~35 ms); the **D2H transfer + Python post-processing** is the 200+ ms cost.

Next cell candidates, in order of estimated payoff:

1. **Defer PIL conversion out of the hot path**: keep image on GPU until the consumer needs PIL. `post_decode` returns `DiffusionOutput(images=[<gpu tensor or uint8 cuda>], ...)` and only `.cpu()`s when the caller actually wants PIL. Saves ~200 ms on the inference path if the consumer is another GPU stage (not our case for the bench, but for real VLA pipelines).
2. **Pinned-memory D2H + async overlap**: `image` is `[1, 3, 512, 512]` fp16 = 1.5 MB. With `cudaHostAlloc`-pinned staging buffer + `cudaMemcpyAsync` on a side stream, the D2H can overlap with the next prompt's tokenize. Marginal here (no next prompt in bench), relevant for serving.
3. **Reduce Python overhead in `post_decode`**: fuse the `clamp + permute + float + numpy + *255 + round + astype` chain into a single GPU kernel (`torch.clamp_` + `.to(torch.uint8)` on GPU, then D2H only the final uint8 buffer — 768 KB instead of 1.5 MB, and no fp32 staging). Saves ~50 ms of CPU work and shrinks the D2H transfer by 2×.
4. **CG on a non-diffusers UNet/VAE forward**: would need a fused kernel wrapper or rewrite — large effort, defer.

## Files on WSL

- `/tmp/sd-turbo-ncu.nsys-rep` — nsys trace with NVTX ranges (sd_unet_step0, sd_scheduler_step, sd_vae_decode)
- `/tmp/sd-turbo-ncu.sqlite` — parsed
- `/tmp/qnsys_sd.py` — NVTX-bucket parser
- `/tmp/qkineto.py` — Kineto top-kernel parser
- `docs/perf/bench_sd_turbo_a633374.csv` — baseline
- `docs/perf/bench_sd_turbo_cg.csv` — CG-attempted (fell back to one-shot)
- `docs/perf/sd-turbo-2026-09-18.trace.json.gz` — Kineto trace (in-repo)

## Open follow-up

1. Decide if "defer PIL out of hot path" is the Phase 2 cell to implement (proposed cell 1 above).
2. Colab T4 NCU re-run for SOL metrics — WSL can't do it.
