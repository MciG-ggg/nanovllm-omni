# ADR 0009: SmolVLA 吞吐——基线、CUDA Graph 阻塞与 torch.compile（2026-10-30）

- Status: Accepted
- Date: 2026-10-30
- Supersedes: 本文 2026-10-30 初版中"本仓库这一层无可改之处"的结论（错误，见下）

## 问题

`smolvla` 单请求约 1.4 s。本文回答：时间花在哪、哪些能在本仓库改、改了值不值。

## 环境

- WSL2 RTX 3050 4GB（4096 MiB），torch 2.11.0+cu130，lerobot 0.6.1。
- 权重 `~/models/smolvla_libero`（1218 MB；policy 为 bf16：600.9M bf16 + 4.0M fp32）。
- 代码 `perf/colab-2x`，基线取 `b9c20d7`，后续改动见"改动"一节。
- 输入：合成观测 256×256 RGB + 8 维零 state，指令 `pick up the red mug`，
  `num_inference_steps=10`。
- 口径：`python -m nanovllm_omni.bench.bench_smolvla --runs 5 --warmup 1 --no-trace`，
  warmup 在计时外，取 p50。

## 基线（改动前，10 次稳态）

| 阶段 | p50 | 占比 |
|---|---:|---:|
| `vlm`（AR + vlm2action） | 162.8 ms | 11% |
| `prepare`（采样初始噪声） | 0.4 ms | 0.03% |
| **`flow`（10 个 flow step）** | **1255.6 ms** | **87%** |
| `decode` | 3.9 ms | 0.3% |
| **total** | **1437.6 ms** | |

峰值显存 2150.9 MiB。同日复跑 total 1503.1 / flow 1317.3 ms，即 run-to-run
漂移约 5%。

## 单个 flow step 的构成（`torch.profiler`，CPU+CUDA）

| 指标 | 值 |
|---|---:|
| CUDA kernel 种类 / 调用次数 | 66 / **2383** |
| GPU 实际忙时间 | **8.7 ms** |
| `cudaLaunchKernel`（CPU） | 39.33 ms × 2121 |
| `cudaMalloc`（CPU） | 25.02 ms × 35 |
| `aten::mm` / `aten::mul` / `aten::copy_` | 11.78 / 10.36 / 9.78 ms |
| `aten::pow` | 5.78 ms × 163（`apply_rope` 每次重建 cos/sin） |

一步 125 ms 里 GPU 只忙约 9 ms，其余是主机侧派发。瓶颈类型与 smolvlm / minimind
相同，只是发生位置在 lerobot 的 `denoise_step` 内部。

## CUDA Graph：不可行（依赖挡住）

把固定 shape 的 `policy.model.denoise_step` 放进 `torch.cuda.graph`：

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

lerobot 的 `denoise_step` 内部有一次未 pin 的 CPU→CUDA 拷贝，capture 直接拒绝。
本仓库这一层不拥有那次拷贝，要修必须改依赖的调用路径。

## torch.compile：可行，3.07×，但不是 bit-exact

`SmolVLAActionPipeline` 现在是每步调用的唯一选择点，因此 `torch.compile` 可以
挂在本仓库这一层，不需要动依赖。

`bench_smolvla --compile-flow` 的实测（同机、同 seed、`--runs 5`）：

| cell | p50 | 峰值显存 |
|---|---:|---:|
| eager（默认） | 1341.9 ms | 2021 MiB |
| `--compile-flow` | **436.6 ms** | 1919 MiB |

**3.07×**，显存略降。

代价是输出不再是逐位一致的参考结果：`tests/test_smolvla_compile_parity.py`
（`@pytest.mark.smoke`，每 cell 一个子进程、两侧都显式播种）测得
`max|compiled − eager| = 3.6e-3`（float32 动作块，量级 ~1）。

这正是本阶段的契约问题：`models/smolvla/action_stage.py` 的注释承诺
"bit-exact alignment with `policy.predict_action_chunk`"。kernel fusion 让
reduction 顺序变化，逐位一致与 3× 提速在这个实现里不可兼得。

## 决定

1. **默认保持 eager**，`bit-exact` 契约不变。
2. `compile_flow` 保留为显式 opt-in（stage 级读取 `args.extra["compile_flow"]`，
   bench 用 `--compile-flow` 打开），因为这只是一个对比 cell，不是运行时旋钮——
   与 ADR-0005 一致：决定留在 stage，`DeployConfig` 不带这个字段。
3. 测试钉的是**漂移边界**（`DRIFT_BOUND = 1e-2`）而不是相等：它防的是"compile
   路径行为变了"，不假装编译后仍是参考实现。
4. 若将来要把 compile 转为默认，必须先回答"动作漂移 3.6e-3 是否可接受"这个问题，
   并且答案要写进本文，而不是靠改默认值悄悄生效。

## 更正

本文初版写过"本仓库这一层无可改之处，因此不提交性能代码"。那是错的：
初版只验证了 CUDA Graph 一条路，就把"一条路走不通"当成"没有路"。
`torch.compile` 在同一层给出了 3.07×，且不需要改依赖。

## 限制

1. 单请求串行入口，吞吐即 1 / wall；并发已被 ADR-0010 否掉（GIL）。
2. 合成观测，不代表真实 LIBERO 分布；动作正确性未与 lerobot 参考实现逐位对比
   （compile 路径按上文边界比较，eager 路径是参考本身）。
3. 一次 profiler 只抓一步，`cudaMalloc` 的 25 ms 可能含首次增长段的成分。
4. `torch.compile` 的首次调用含编译时间，未计入上表；bench 的 warmup 会吸收它，
   但真实服务的第一请求会变慢。
5. 原始 CSV 是本地产物（WSL `/tmp`），不进 git；复现命令见文首。
