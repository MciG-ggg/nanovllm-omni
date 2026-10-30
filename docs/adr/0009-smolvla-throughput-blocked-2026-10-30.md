# ADR 0009: SmolVLA 吞吐基线与被阻塞的优化（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

## 问题

`smolvla` 单请求约 1.4 s。本文回答：时间花在哪、哪些能在本仓库改、哪些不能。
结论先行：**瓶颈在 lerobot 的 flow step 内部，本仓库这一层没有可改的杠杆，
结构性修法（CUDA Graph）被依赖里的一次未 pin 拷贝挡住。** 因此本次没有代码改动。

## 环境

- WSL2 RTX 3050 4GB（4096 MiB），torch 2.11.0+cu130，lerobot 0.6.1。
- 权重 `~/models/smolvla_libero`（1218 MB，policy 为 bf16：600.9M bf16 + 4.0M fp32）。
- 代码 `perf/colab-2x` @ `b9c20d7`。
- 输入：合成观测 256×256 RGB + 8 维零 state，指令 `pick up the red mug`，
  `num_inference_steps=10`，`seed=0`。
- 口径：同进程 warmup 一次后连续 10 次，取中位；`perf_counter` 手工包住各阶段。

## 基线（10 次，稳态）

| 阶段 | p50 | 占比 |
|---|---:|---:|
| `vlm`（AR + vlm2action） | 162.8 ms | 11% |
| `prepare`（采样初始噪声） | 0.4 ms | 0.03% |
| **`flow`（10 个 flow step）** | **1255.6 ms** | **87%** |
| `decode`（截断 + 转 numpy） | 3.9 ms | 0.3% |
| **total** | **1437.6 ms** | |

峰值显存 2150.9 MiB。同日复跑一次得到 total 1503.1 / flow 1317.3 ms，
即 run-to-run 漂移约 5%，小于任何候选改动的预期收益量级。

## 单个 flow step 的构成（`torch.profiler`，CPU+CUDA）

| 指标 | 值 |
|---|---:|
| CUDA kernel 种类 / 调用次数 | 66 / **2383** |
| GPU 实际忙时间 | **8.7 ms** |
| `cudaLaunchKernel`（CPU） | 39.33 ms × 2121 |
| `cudaMalloc`（CPU） | **25.02 ms × 35** |
| `aten::mm` / `aten::mul` / `aten::copy_` | 11.78 / 10.36 / 9.78 ms |
| `aten::pow` | 5.78 ms × 163（`apply_rope` 每次重建 cos/sin） |
| `aten::empty_strided` | 5.47 ms × 471 |

即：一步 125 ms 里 GPU 只忙约 9 ms，其余是主机侧派发与分配。这与 ADR-0003
记录的"`:flow-step` 是热点"一致，但把原因定位到了**每步 2383 次 kernel 调用**，
而不是算力。

## 尝试过但没有落地的修法

### 1. CUDA Graph 包住 flow step（被依赖挡住）

手工实验：把 `policy.model.denoise_step` 的固定 shape 调用放进
`torch.cuda.graph`，先侧流 warmup 再 capture。结果：

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

lerobot 的 `denoise_step` 内部存在一次未 pin 的 CPU→CUDA 拷贝，capture 直接拒绝
（与 ADR-0001 记录的 sd-turbo CG 失败同一类问题）。这在我们这一层无法修：
本仓库的 `SmolVLAActionPipeline.denoise_step` 只做 `timestep` 构造与转发，
不拥有那次拷贝。要落地必须改依赖的调用路径（例如让它接受外部准备好的
device 张量），或在依赖里 pin 掉那张表。

### 2. 分配器配置（无效）

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：

| cell | flow p50 | total p50 |
|---|---:|---:|
| 默认 | 1317.3 ms | 1503.1 ms |
| expandable_segments | 1378.6 ms | 1544.3 ms |

略差，且 4 GB 卡上 2.1 GB 已用，碎片不是主因。**不作为建议**。

### 3. 本仓库这一层（无可改之处）

`models/smolvla/action_stage.py` 每步只做：构造 `timestep` 张量、转发
`policy.model.denoise_step`、`latents += v * dt`。三项合计 <1 ms/step，
删掉也不改变量级。**没有实测收益就没有改动**，本次不提交性能代码。

## 下一个可行方向（需要设计，不是小改）

flow loop 有约 90% 的时间 GPU 空闲，所以**并发**是比"抠单步开销"更现实的杠杆：
一个请求的 flow loop 与另一个请求的 `vlm` 阶段可以重叠。前提是先把三处进程级
全局状态处理掉（fork 的 `Context`、`runner.sampler` 的就地替换、全局 RNG）——
它们让 VLM 阶段目前不可并发。这属于 ADR-0008 结尾那条"把一个决定收进一处"的
同一类工作：先建容器，再谈并发。

## 限制

1. 单请求串行入口，吞吐即 1 / wall。
2. 合成观测，不代表真实 LIBERO 分布；动作正确性未与 lerobot 参考实现对比。
3. 一次 profiler 只抓一步，`cudaMalloc` 的 25 ms 可能包含首次增长段的成分；
   它与 `cudaLaunchKernel` 的相对占比在两次运行间稳定。
4. 原始脚本 / 日志是本地产物（`.scratch/`、WSL `/tmp`），不进 git。
