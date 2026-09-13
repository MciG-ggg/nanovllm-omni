# 三模型 Phase 1 Profile 结果（2026-09-13）

## 环境与口径

- 机器：WSL，NVIDIA GeForce RTX 3050 4GB Laptop GPU
- GPU 显存：4096 MiB
- PyTorch：2.11.0+cu130
- 代码：WSL 仓库 HEAD 为 `ce0f111`；源码由本地工作树同步，bench CSV 的 commit 字段仍为 `unknown`
- 模式：CUDA，warmup 在计时外
- profile：`torch.profiler` / Kineto；阶段区间使用 `nanovllm_omni.utils.profiling.profile_range`
- 原始 trace：保存在 WSL `/home/mcig/docs/perf/`，大文件不进 git

阶段 profile 会增加 profiler 和 `record_function` 开销。因此：

- 绝对 wall time 以同一代码路径、未加阶段 annotation 的 baseline 为准。
- 阶段区间用于判断热点和决定 Phase 2 优化方向。
- 不把不同日期、不同 checkout 或不同 instrumentation 下的 wall time 直接当作回归结论。

## 总览

| 模型 | baseline wall p50 | baseline wall p99 | 峰值显存 | 当前热点 |
|---|---:|---:|---:|---|
| sd-turbo | 503.554 ms | 517.670 ms | 3098 MiB | VAE decode |
| smolvlm short | 291.197 ms | 313.179 ms | 2181 MiB | AR decode |
| smolvlm medium | 5075.190 ms | 5306.517 ms | 2182 MiB | AR decode |
| smolvlm long | 4449.954 ms | 4524.247 ms | 2183 MiB | AR decode |
| smolvla | 1385.219 ms | 2291.388 ms | 2021 MiB | flow matching loop |

## sd-turbo

配置：1 个 prompt、5 次计时、1 次 warmup、1-step、fp16、seed=42。

阶段 annotation 的 5 次均值：

| 阶段 | 平均耗时 |
|---|---:|
| `:tokenize` | 84.1 ms |
| `:unet` | 275.1 ms |
| `:vae-decode` | 358.1 ms |

结论：VAE decode 约占阶段总和的一半，UNet 约占 38%。当前没有证据支持先做 UNet fusion；应先评估 VAE decode 的实现和 kernel 组成。峰值显存为 3.1 GiB，占 4 GiB 的约 76%，显存压力是第二约束。

Phase 2：先做 diffusers parity 和 VAE/UNet 的 kernel 分解，再决定 CUDA Graph。此前 CUDA Graph 因 CPU-to-GPU copy 未 pin 而 capture 失败，暂不把它当成已完成优化。

CSV：`docs/perf/bench_sd_turbo_wsl_stages.csv`。

## smolvlm

配置：short / medium / long 三种 text-only prompt，每种 10 次、1 次 warmup，`max_new_tokens=64`，bf16。

未加阶段 annotation 的 baseline：

| 输入 | p50 | p99 | 峰值显存 |
|---|---:|---:|---:|
| short | 291.197 ms | 313.179 ms | 2181 MiB |
| medium | 5075.190 ms | 5306.517 ms | 2182 MiB |
| long | 4449.954 ms | 4524.247 ms | 2183 MiB |

带 annotation 的 trace 捕获到：

- 外层 `:vlm-decode`：已捕获事件均值约 5043 ms；它包住一次完整 VLM 调用。
- 内部 `:tokenize`：均值约 2.7 ms。
- 内部 `:vlm-prefill`：均值约 69.9 ms。

这次 Kineto 提示会在 cycle 结束时清理 events，三个输入的内部 annotation 没有完整保留，所以这里不把事件数量不足的结果当成严格的每输入分解。现有证据仍足够说明：AR decode 远大于 tokenize 和 prefill，是第一优化目标；GPU 利用率低和大量小 kernel launch 与该判断一致。

Phase 2：只针对 decode 做 CUDA Graph / static KV cache cell；每个 cell 重新测 wall time，并做 greedy token parity。prefill graph 不应默认加入，除非单独测量证明它有收益。

CSV：`docs/perf/bench_smolvlm_wsl_stages.csv`。

## smolvla

配置：1 个 synthetic observation（256x256 RGB + 8-dim zero state），20 次、1 次 warmup，10 个 flow-matching steps。

未加阶段 annotation 的 baseline：p50 1385.219 ms，p99 2291.388 ms，峰值显存 2021 MiB。

阶段 annotation 均值：

| 阶段 | 平均耗时 |
|---|---:|
| VLM 调用与当前预处理区间（代码名 `:tokenize`） | 170.1 ms |
| 当前 `vlm2action` 转换区间（代码名 `:vlm-prefill`） | 0.06 ms |
| `:flow-step` | 1126.3 ms |
| `:action-decode` | 0.36 ms |

这里现有 bench 的 marker 命名与实际边界并不完全一致：`vlm` 调用被包在 `:tokenize`，`vlm2action` 转换被包在 `:vlm-prefill`。因此报告使用实际执行含义解释数据，不把 marker 名字当作事实来源。

结论：flow matching 约占主要执行时间，显存 2.0 GiB（约 49%）不是当前第一瓶颈。Phase 2 应先比较 flow step 数量和单步耗时；batch 扩展只有在真实 LIBERO 输入和控制周期需求成立后再测。

CSV：`docs/perf/bench_smolvla_wsl_stages.csv`。

## Colab T4 NCU 补充（2026-09-13）

环境：Colab Tesla T4（compute capability 7.5），NCU 2025.1.1；每项只 capture 少量 matching launch，启用 `SpeedOfLight`、`Occupancy` 和 `MemoryWorkloadAnalysis`。T4 与 WSL RTX 3050 的架构不同，以下数值只用于判断瓶颈类型，不能替代 3050 上的 wall-time 结论。

| 模型 / 实际 kernel | compute-memory throughput | DRAM throughput | active warps | 结论 |
|---|---:|---:|---:|---|
| sd-turbo VAE `nchwToNhwcKernel` | 61.6% | 61.6% | 100% | 高 DRAM 吞吐的 layout conversion；应先研究 VAE tensor layout，不能把它当作 compute fusion 候选。 |
| smolvlm decode `gemv2T_kernel` | 93.7% | 93.7% | 100% | 明确 memory-bound；单 kernel 优化空间很小，支持优先测试 decode-only CUDA Graph 来减少大量 sequential launch。 |

另有两份 capture 不能用于热点决策：

- sd-turbo 的宽泛 CUTLASS regex 实际抓到 `fmha_cutlassF_f16...` attention kernel，不是已由阶段 profile 确认的 VAE decode 热点。
- smolvla 的未过滤 GEMM capture 实际抓到 VLM 的 `magma_sgemmEx_kernel`；按 `:flow-step` 做 NCU NVTX filter 没有命中。现有 NVTX range 是 push/pop range，而本机 NCU include 配置只匹配 start/end range，因此还没有有效的 flow-step kernel SOL 数据。

原始报告为本地产物：`docs/perf/ncu-2026-09-13-{sd-vae,sd-unet,smolvlm-gemv,smolvla-flow}.ncu-rep`。下一次 smolvla NCU 应先改为可被 NCU range filter 识别的 marker，或用独立 flow-step harness，再测 flow kernel；不要用当前 VLM GEMM 代替它。

## 已知限制

1. `bench/env.py` 的仓库根目录推断仍导致 CSV 的 commit 为 `unknown`；这不影响本次计时，但削弱了结果溯源。
2. smolvlm 的 `torch.profiler` cycle 清理导致内部 annotation 不完整；需要完整分阶段数据时，应调整 profiler schedule 或单独按输入运行。
3. smolvla 使用 synthetic observation，不能替代真实 LIBERO episode。
4. 本轮只完成 Phase 1 profile，没有 CUDA Graph、fusion、static KV cache 或 batch 优化结果。

## Phase 状态

- Phase 1 baseline：三模型已完成。
- Phase 2 bottleneck/cell：未完成；smolvlm decode CG 是首个候选。
- Phase 3 reference parity：未完成。
- Phase 4 numerical parity：按 ADR 0001 的条件触发。
