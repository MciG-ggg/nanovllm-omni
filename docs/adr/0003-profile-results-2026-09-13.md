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

该 flow-step harness 已加入仓库：`nanovllm_omni/bench/bench_smolvla_flow.py`。它只建一次 prefix，然后重复跑 flow loop，NCU 的 launch-count 窗口因此落在 flow kernel 上。命令：

```
python -m nanovllm_omni.bench.bench_smolvla_flow --repeats 20 --steps 10
```

## Colab T4 优化结果（2026-10-20）

环境：Colab Tesla T4（16 GB），torch 2.9.0+cu128；代码版本为 `perf/colab-2x` 分支，主仓 `e86a153`，fork submodule `7cd0d4a`。口径：1 次 warmup + 5 次计时，wall 取中位；峰值显存取 `torch.cuda.max_memory_allocated`。

| 模型 | baseline wall p50 | 优化后 wall p50 | 加速 | 峰值显存 | 依据 |
|---|---:|---:|---:|---:|---|
| smolvlm | 586.0 ms | 185.7 ms | 3.16x | 3951.7 → 3959.9 MB | `docs/perf/smolvlm-2026-10-20.md` |
| sd-turbo | 377.9 ms | 298.4 ms | 1.27x | 3101.0 → 3109.1 MB | `docs/perf/sd-turbo-2026-10-20.md` |
| minimind-o | 未测 | 未测 | — | — | `docs/perf/minimind-omni-2026-10-20.md` |
| smolvla | 未测 | 未测 | — | — | 见下 |

smolvlm 的加速来自两项改动，两项都有 nsys profile 支撑：

1. decode 模型按请求 dtype 构建。之前 decode 跑 fp32，峰值显存 9976 MB → 7163 MB，wall 不变。
2. decode 循环走 CUDA graph。eager 684.0 ms → graph 177.3 ms。profile 显示 GPU 只有 24% 时间在忙，瓶颈是 kernel launch。fork 现在给捕获的 KV 窗口封顶（`Config.graph_max_seq_len`），上下文超窗或捕获失败都回落到 eager。

sd-turbo 的 1.27x 来自响应路径：PNG 用最低压缩级别写出（146.3 ms → 34.8 ms），像素不变。2x 对这个模型在 T4 上不可达：一次 512x512 请求的 GPU 成本是 VAE decode 174.3 ms + UNet 72.9 ms = 247 ms，而 2x 需要的预算是 149 ms，VAE decode 一项就超了。

两个模型未测，原因不同：

- minimind-o：hub 的 `trust_remote_code` 建模文件让本会话的沙箱拒绝了运行。项目侧的解法已经落地：仓库注册自己的 `minimind-o` config class（`nanovllm_omni/models/minimind_omni/hf_config.py`），`AutoConfig` 因此不再加载 hub 的 `auto_map` 文件，thinker、talker、tokenizer 都不再执行远端代码。权重的两段切分也已按真实 checkpoint 核对（2026-10-21）：195 个 tensor，thinker 拿 91 个、talker 拿 92 个，两个集合不重叠，被丢掉的 12 个是本项目用自己的 bridge 取代的 `audio_proj.*` / `vision_proj.*`。剩下的缺口是一块 GPU。
- smolvla：`lerobot` 0.6.1（`transformers` 5.5.4，`torch` 2.11.0+cpu）已在 Colab 的 CPU session 上装好，`HuggingFaceVLA/smolvla_libero` 的权重也用仓库自己的调用加载成功（489 个 weight tensor，`chunk_size` 50、`max_action_dim` 32、`num_steps` 10、action 7 维）。安装过程暴露一个项目侧缺口：`lerobot.policies` 的导入链会走到 `lerobot.datasets`，它需要 PyAV，因此 `pyproject.toml` 的 `smolvla` extra 现在显式声明 `av`。剩下的缺口是一块 GPU。

原始 nsys / ncu 报告是本地产物，本节只保留结论。

## 已知限制

1. `bench/env.py` 的仓库根目录推断仍导致 CSV 的 commit 为 `unknown`；这不影响本次计时，但削弱了结果溯源。
2. smolvlm 的 `torch.profiler` cycle 清理导致内部 annotation 不完整；需要完整分阶段数据时，应调整 profiler schedule 或单独按输入运行。
3. smolvla 使用 synthetic observation，不能替代真实 LIBERO episode。
4. 本轮只完成 Phase 1 profile，没有 CUDA Graph、fusion、static KV cache 或 batch 优化结果。
5. 2026-10-20 的 T4 数字和本文的 3050 数字来自不同硬件，不能直接相减当作回归结论；两组数字只在各自硬件内比较。

## Phase 状态

- Phase 1 baseline：三模型已完成。
- Phase 2 bottleneck/cell：smolvlm 的 decode CUDA graph cell 已完成（3.16x，2026-10-20），sd-turbo 的 cell 已完成（1.27x，2026-10-20）；minimind-o 与 smolvla 未完成。
- Phase 3 reference parity：未完成。
- Phase 4 numerical parity：按 ADR 0001 的条件触发。
