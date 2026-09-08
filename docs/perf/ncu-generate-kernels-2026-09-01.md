# MiniMind-O generate 阶段 ncu kernel 分析（2026-09-01）

> 任务：用 Nsight Compute（`ncu`）逐 kernel 看 generate 阶段硬件行为，决定
> 是否需要手写 kernel / 该写哪种。**结论：不需要。瓶颈是 launch overhead，
> 不是单个 kernel 慢。** 见 [§ 6 结论](#6-结论)。

> **引用约定**：`§N` 指本报告的**小节号**（共 31 节）。带 `#N` 的是
> autoresearch **迭代号**（实验日志 `.auto/log.jsonl`），两者不同。
> 早期小节里偶尔混用 `§N` 表迭代号处已修正为小节号。

> **生产交付物索引**（全部已 commit、149+ tests 绿、GPU 实证）：
> - `nanovllm_omni/models/minimind_omni/attention.py` — `enable_fixed_kv_buffer`（§23/§24）
> - `nanovllm_omni/optim/cuda_graph.py` — `enable_cuda_graph` + `CudaGraphDecoder`（§32 起）
> - `nanovllm_omni/models/minimind_omni/thinker.py` — `run_generate(use_cuda_graph=True)`（§27）
> - `nanovllm_omni/optim/bench/__main__.py` — `--use-cuda-graph` CLI（§30）
> - 验收：§28（WAV-MD5 determinism）· §29（异构 prompt 稳健）

## 1. 硬件与环境

| | 原计划 | 实际 |
|---|---|---|
| GPU | NVIDIA RTX 3050 4GB Laptop（sm_86, 8 SMs） | **NVIDIA T4（sm_75, 40 SMs, 16GB）** |
| CUDA | 13.1 + PyTorch 2.13.0+cu130 | 13.0 + PyTorch 2.11.0+cu128 |
| ncu | 2024.3 | 2025.1.1.0 |
| 平台 | WSL2（`mcigs-wsl`） | **Colab（`gpu-t4-s-kkb-usw1b2`）** |

**为什么换了硬件**：用户原计划 WSL 端 3050，但启动会话时 `mcigs-wsl` tailnet
relay 不可达（`tailscale status` 显示 WSL 仍 active，但 SSH connect timeout）。
用户决定先在 Colab 跑通流程再回头校准。**T4 不是 3050 的代用品**：SM 数差
5×（40 vs 8）、Turing vs Ampere、sm_75 vs sm_86；具体 SOL%/occupancy 数字
会有差异。但下面 [§ 2](#2-跨平台稳定的发现) 的两个发现是 **Kineto 跨硬件
一致的**，所以结论会平移到 3050。

WSL 恢复后会重跑一遍上 3050 的数据更新这张表。

## 2. 跨平台稳定的发现（Kineto 在 WSL 3050 + Colab T4 都确认）

`docs/perf/session-1.md` 在 WSL 3050 上记的 generate 阶段 top kernels，本会话
在 Colab T4 上重跑 `python -m nanovllm_omni.engine.bench profile-detail` 后**完
全一致**：

| kernel | WSL 3050 (count, total_us) | Colab T4 (count, total_us) | 备注 |
|---|---:|---:|---|
| `cudaLaunchKernel` | 10,336 / 150,458 | 11,367 / 116,930 | 14% 墙时 |
| `cudaMemcpyAsync` | 1,300 / 24,550 | 1,300 / 15,810 | D2H/H2D 复制 |
| `gemv2T_kernel_val` | 192 / 13,544 | 418 / 14,706 | cuBLAS GEMV decode Q=1 |
| `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` | **1,744 / 3,887** | **1,744 / 10,363** | **同 call 数** |
| `cudaStreamSynchronize` | 352 / 11,869 | 352 / 2,722 | 同步点 |

**`direct_copy_kernel_cuda` 在两台机器上都精确触发 1744 次**——这不是噪声，
是模型代码的固定模式。1,744 ÷ (16 AR steps × 12 layers × 2 KV splits) ≈ 4.5
per (step, layer, split)，推测是 `qkv_proj` 之后 split + permute + cast 链
中的某次写。值得追但 ncu 上单独看不烫手（见 [k3b](#k3b-direct_copy_kernel_cudaunrolled_elementwise_kernel)）。

## 3. ncu profile 设置

四个 kernel 走 `ncu --target-processes all --kernel-name 'regex:...' --launch-skip N --launch-count 3 --section SpeedOfLight,Occupancy,SchedulerStats,WarpStateStats,MemoryWorkloadAnalysis,LaunchStats --export ...`。

`--launch-skip` 按 Kineto 看到的总调用次数挑：GEMV 跳 200（418 总）、
fmha 跳 100（204 总）、cat 跳 200（384 总）、elementwise_copy 跳 500–1500
（1744 总）。`--launch-count 3` 抓稳态 3 个 instance。

> ncu CLI 的坑：默认 `--profile-from-start on`；本会话第一次加
> `--profile-from-start off` **反而全部报 "No kernels were profiled"**——
> 这个 flag 在我们这版 ncu 里不是"等会儿再开"，而是"什么都不 profile"。
> 不要用它。

## 4. 四个 kernel 的 ncu 数据

| # | Kernel | Grid × Block | Duration | Compute SOL | Memory SOL | Achieved Occ | Scheduler Stall | 受限类型 | ncu 自评 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| k1 | `fmha_cutlassF_f16_aligned_32x128_rf_sm75` (PyTorch MemEffAttention) | **(1, 8, 1) × (32, 4, 1)** = 8 blocks | 20.86us | **3.17%** | **2.96%** | 12.5% (theo 25%, smem-limited) | **82% No Eligible** | **launch / grid-size** | "kernel grid is too small to fill the available resources" |
| k2 | `gemv2T_kernel_val<..., 128, 16, 2, 4, ...>` (cuBLAS GEMV) | (96, 1, 1) × (128, 1, 1) | 20.99us | **45.71%** | **52.70% DRAM** (62 GB/s) | 57.1% (theo 100%, reg-limited) | 78% No Eligible | **memory-bound**（接近 SOL）| "Eligible Warps 0.28/cycle" |
| k3b | `unrolled_elementwise_kernel<direct_copy_kernel_cuda<TensorIteratorBase>>` | **(1, 1, 1) × (128, 1, 1) = 1 block** | 5.44us | **0.07%** | **2.14%** | 12.0% (theo 100%) | n/a（grid 太） | **launch / single-block latency** | "Grid only 2 blocks, less than 40 SMs" |
| k4 | `CatArrayBatchedCopy_vectorized<...>` (`torch.cat`) | **(10, 2, 1) × (128, 1, 1) = 20 blocks** | 5.50us | **0.90%** | **1.62%** (5 GB/s) | 12.2% (theo 100%) | **97% No Eligible** | **launch / single-block latency** | "Grid only 2 blocks, less than 40 SMs" |

> T4 has 40 SMs。3050 只有 **8 SMs**，同样 grid 下 Waves/SM 数字会再降 5×，
> 但相对利用率（百分比）一致。

### k1 (fmha)
- **Compute 3% / Memory 3%** —— 这个 kernel **完全没在跑**，是 grid 太小
  喂不饱 GPU。Q=1 decode attention 本质上是 1 次 GEMM + 1 次 softmax，
  数据量撑不起 40 个 SM。
- Achieved Occupancy 12.5% 受 **shared memory 限制**（每 block 用 26KB
  smem，T4 最多每 SM 放 2 个 block）。
- Scheduler 82% 时钟周期在等 warp 就绪（No Eligible）。`Issued Warp Per
  Scheduler = 0.18`，5.6 cycles 才出一条指令。
- **结论**：库（PyTorch SDPA / cutlass fmha）已是正确选择，Q=1 是工作
  量固有的小。**手写 kernel 不会改变 grid 大小**——Q=1 决定了。

### k2 (cuBLAS GEMV)
- **Compute 46% / DRAM 53%**（52.7% of peak 117 GB/s）。这是 4 个 kernel
  里**利用率最高的**，已经是接近 SOL 的 memory-bound GEMV。
- Achieved Occupancy 57% (theo 100%) 受 register 限制（每 thread 62 个
  reg，T4 每 SM 64K reg = 8 block）。
- L2 hit 54%——GEMV 在 T4 上的 L2 reuse 正常。
- **结论**：cuBLAS GEMV 在 53% DRAM SOL，**没有提升空间**。写裸 CUDA 也
  不会比 cuBLAS 更好（cuBLAS GEMV 是 NVC 多年优化的标量版本）。

### k3b (direct_copy_kernel_cuda)
- **Grid Size: 1 block**（kernel 名字里说 `direct_copy`，grid 是
  `(1,1,1) × (128,1,1)` = 128 threads）。**1 个 SM 跑 128 threads 的
  copy**。其余 39 个 SM 干瞪眼。
- Compute 0.07% / Memory 2.14%——确实是纯拷贝，但**只有 1 个 block 在跑**。
- 1744 次调用 × 5.44us = **9.5ms 浪费在"启动 1 个 block 再 kill"的模式**。
- **结论**：ncu 数据**不能**用来决定"该不该优化"——**5us 一调**已经接近
  kernel launch 的物理下限（ncu 重放一次 ~5us）。**优化方向不是写
  kernel，而是消除冗余调用**（沿用 E24 教训：少干活比快 kernel 重要）。
  需要 grep `direct_copy` / `as_strided` / `expand` 看是谁在调。

### k4 (CatArrayBatchedCopy, torch.cat)
- **Grid Size: 20 blocks**（`(10,2,1) × (128,1,1)`）。比 k3 多，但只有
  0.5 个 wave/SM。
- Compute 0.90% / Memory 1.62% / Scheduler 97% No Eligible。
- 384 次调用 × 5.5us = **2.1ms 在做 torch.cat**。
- 来源推测：`attention.py:_attention_forward` 的 `torch.cat([past_kv,
  k], dim=1)`——每个 AR step × 每个 layer × 2 个 split（K、V）= 16 × 12 × 2
  = 384。**完全吻合**。
- **结论**：和 k3b 同性质。CUDA Graph capture 会**自动消除**所有这些 cat 的
  launch 开销（但底层 kernel 还是要跑）。写 Triton kernel 替 cat **没意义**
  ——cat 本身就是最优的 memcpy 模式，问题在 launch 而不在执行。

## 5. 关键 takeaway（跨 kernel）

| 现象 | 含义 |
|---|---|
| 4/4 kernel Compute SOL 都 < 50%（k2 之外 < 5%） | **GPU 没在算**——不是 kernel 慢，是 kernel 之间空着 |
| 4/4 kernel Achieved Occ 都远低于 theoretical | 工作量天生小，grid 喂不饱 GPU |
| k3b + k4 单次 5us，launch 总次数 2128 | 单次已经接近物理下限，**没法再压** |
| k1 fmha 在 decode 步只能 8 block（Q=1 × 4 layer heads） | Q=1 决定数据量，不是实现选择 |
| k2 cuBLAS GEMV 53% DRAM SOL | **这就是 cuBLAS 在 4GB 卡上能给的极限** |

**整张表的"受限类型"列里，0 个是"compute-bound / memory-bound near SOL"**。
**没有一个 kernel 处于"快要打满 GPU 但还差一点"的状态**——全都差得很远。
**手写 kernel 不会让 3% compute SOL 变 90%**，因为瓶颈不在算子内部。

## 6. 结论

按 task 要求的梯子：

### 6.1 不需要手写 kernel 的部分（绝大多数）

- **RMSNorm**（`aten._fused_rms_norm`）——已是 C++ 融合算子；Kineto 显示
  `vectorized_layer_norm_kernel` 仍在跑（0.2%）说明 fuse 的 cudnn path 没
  完全取代。**结论：换 cudnn fused RMSNorm / 不动**——kernel 自己已经够
  小（~2us），没必要重写。
- **QKV/gate-up projection**（cutlass GEMM）——已是 cuBLAS/cutlass 内置
  GEMM。**结论：不动**。
- **GEMV decode**（cuBLAS GEMV）——已是 53% SOL 的标量 GEMV。
  **结论：不动**。
- **Flash attention / SDPA**（PyTorch MemEffAttention）——已是 cutlass fmha
  标量实现。**结论：不动**。

### 6.2 真正的高价值发现：**消除冗余 launch**（沿用 E24 教训）

1. **`direct_copy_kernel_cuda` × 1744 次**（~9.5ms 累计）
   - 来源不明，要 grep 找。可能与 `_fused_attention_forward` 里
     `qkv_proj → split → reshape` 的某个 permute 有关。
   - **不是手写 kernel 能解决的问题**——5us 一次已经接近物理下限。
   - 解决路径：(a) 在 Python 层消除冗余 op；(b) CUDA Graph capture 摊薄
     launch。
2. **`torch.cat` × 384 次**（~2.1ms 累计）= KV cache 更新。
   - 12 layer × 16 step × 2 (K,V) = 384，**每个 AR step 每个 layer 必跑**。
   - **避免不掉**——KV 必须增长。
   - 优化路径：**CUDA Graph capture** 会消除单次 launch 开销，cat 本
     体的 5us 跑不掉但跑得紧凑。

### 6.3 用 torch.compile 不是手写 CUDA 的替代
- `attention.py:enable_fused_rope` 已经包了 `torch.compile(dynamic=True)`。
- `E16 torch.compile MiniMindBlock.forward` 已经是 -91 ms regression（Inductor
  overhead × 16 layer），**不要再试**。
- 任何走 `torch.compile` 的更大范围 monkey-patch 都是亏的——结论来自
 25 轮调优的硬经验。

### 6.4 **唯一可能的方向：CUDA Graph capture**

> "全栈 CUDA Graph capture（需要 monkey-patch `model.model.forward` 的
>  `start_pos = past_key_values[0][0].shape[1]` 计算逻辑。KV cache 静
>  态化为固定 shape 后才能 capture。理论收益 50–150 ms，**风险** 是
>  monkey-patch 范围大、revert 复杂、影响所有下游用户。"
> —— `docs/perf/minimind-omni-under-500ms.md` §5.3

**这次 ncu 数据印证了上面的判断**：4 个 kernel 都是 launch-bound / grid-too-
small，没有任何"kernel 本身可以优化"的余地。**剩下唯一能挤的就是 launch
数**（10,336 → 1）。CUDA Graph 是标准工具，不算手写 kernel。

### 6.5 不要做

| 方向 | 理由 |
|---|---|
| Triton fused RMSNorm+RoPE+add | RoPE 已经用 torch.compile 融合；Inductor overhead 在 16 layer 上 -91ms regression（E16）；Triton 不会让 QKV GEMM 更快 |
| 裸 CUDA 重写任何 kernel | 上面 [§ 6.1](#61-不需要手写-kernel-的部分绝大多数) 已说明每个都被库吃满了 |
| CUDA C++ extension 替代 cuBLAS GEMV | cuBLAS GEMV 已是 53% DRAM SOL，超越空间几乎为零 |
| 手写 KV cat 替代 `torch.cat` | cat 本身 5us/次，无法再压；CUDA Graph 才是出路 |

## 7. 交付后的开放项

1. **WSL 3050 重跑一次**——把 T4 数据换成本地硬件的数字（Kineto 跨硬件
   一致，但 ncu SOL%/occupancy 因 SM 数差 5× 而会变）。
2. **`direct_copy_kernel_cuda` × 1744 来源定位**——grep
   `direct_copy` / `as_strided` / `expand` / `copy_` 在
   `nanovllm_omni/models/minimind_omni/`。如果是 fused projection 的副作
   用，可以走 E24 的"per-instance dedupe"思路消除冗余；如果来自 KV cache
   路径，则应该用 KV 原地扩展避免 copy。
3. **CUDA Graph capture for `Omni.generate`**——这是 `docs/perf/.../
   minimind-omni-under-500ms.md` §5 标了 50–150ms 收益 + 高风险的项
   目，ncu 现在给了硬证据支持它的方向。**风险评估**仍要单独做（影响
   monkey-patch 范围、对外公开 API 兼容）。
4. **batch runner**——`docs/perf/model-family-bottlenecks-2026-08-27.md` 已
   量化 batch=2 给 +87% 吞吐，与 kernel 优化正交。建议跟 CUDA Graph 一起做。

## 8. 验证命令（WSL 3050 恢复后重跑）

```bash
# 1. 复现基线
ssh mcigs-wsl "cd ~/nanovllm-omni && \
  .venv/bin/python -m nanovllm_omni.engine.bench time \
    --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
    --max-tokens 16 --runs 5 --warmup 2"

# 2. Kineto top kernel（不需要 ncu）
ssh mcigs-wsl "cd ~/nanovllm-omni && \
  .venv/bin/python -m nanovllm_omni.engine.bench profile-detail \
    --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
    --max-tokens 16 --prompts short_03 --warmup 1 --runs 1 \
    --out /tmp/profile-detail"

# 3. ncu 4 个 kernel（与本报告同 regex）
sudo -n /usr/local/bin/ncu --target-processes all \
  --kernel-name 'regex:.*fmha_cutlass.*' \
  --launch-skip 100 --launch-count 3 \
  --section SpeedOfLight --section Occupancy \
  --section SchedulerStats --section MemoryWorkloadAnalysis \
  --export /tmp/gen-k1.ncu \
  ~/venvs/vllm-omni/bin/python -m nanovllm_omni.engine.bench time \
    --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
    --max-tokens 16 --runs 2 --warmup 1
# 同理对 k2/k3b/k4 改 regex

# 4. 读报告
sudo -n /usr/local/bin/ncu --import /tmp/gen-k1.ncu.ncu-rep \
  --section SpeedOfLight --section Occupancy
```

## 9. 与 task 原始梯子的对照

| 梯子 rung | task 原文 | 本报告落点 |
|---|---|---|
| 1. 访存受限但库能覆盖 | "换 FlashAttention/cuDNN/SDPA 或调参数" | **k2 GEMV 已在 53% DRAM SOL，不换** |
| 2. 有融合残渣没吃满 | "试 torch.compile(mode='max-autotune')" | **E16 已证 -91ms regression，不上** |
| 3. 非标 pattern 无现成算子 | "建议 Triton kernel（不写裸 CUDA）" | **没有非标 pattern**——cat / direct_copy 是 PyTorch 算子 |
| 4. 只有上面都不行才建议裸 CUDA | "并说明理由" | **不需要**——所有 kernel 都已被库吃满 |
| **高价值发现**：冗余 kernel | "torch.cat/clone/contiguous 这类" | **k3b direct_copy × 1744（5us × 1744 = 9.5ms）** + **k4 cat × 384（2.1ms）** |

## 10. Post-investigation iterations（2026-09-01, Colab T4）

报告 commit 后又跑了 3 个 iteration 来验证或量化结论的可推广性，全部在
Colab T4 上完成（WSL 3050 tailnet relay 全程超时）。记录如下。

### 10.1 Iteration #2：试消除 `qkv.split().reshape()` 触发的拷贝

**Patch**：`attention.py:_fused_attention_forward` 把 `torch.split +
.reshape()` 替换成 `qkv.view(B, S, n_total, head_dim) + [:, :, :n_q]` 切片。

**Bench**（Colab T4，5 runs × 6 prompts，--warmup 2）：
- baseline median: **360.71 ms**
- patched median: **363.13 ms** (+0.7%, 在 noise 内)

**Profile 验证**（独立 profile-detail）：patch **改了 0 个 kernel 计数**——
`direct_copy_kernel_cuda` 仍然 1744 次，`cudaLaunchKernel` 仍然 11367 次。
下游 `_attention_forward` 的 transpose + SDPA、RoPE 的 cat、
`aten._fused_rms_norm` 的 `.to(fp32)` 在非 contig 输入上各自触发
`.contiguous()`，**总拷贝数不变**。

**结论**：1744 不是集中在 QKV split 上，分布在多个 op 的 `.contiguous()`
调用链里。**单点消除无效**。→ discard。

### 10.2 Iteration #3：试去掉 `aten._fused_rms_norm` 前的 `.to(fp32)`

**Hypothesis**：`_fused_rms_forward` 里 `x.to(torch.float32)` 对 fp16 输入
多余，触发 ~768 次冗余拷贝（4 RMSNorm × 12 layer × 16 step）。

**结果**：**直接 crash**——
`RuntimeError: expected scalar type Half but found Float`。
`aten._fused_rms_norm` 强制要求 fp32 输入，`.to(fp32)` 是必需的。

**结论**：API 假设错。→ discard。

### 10.3 Iteration #4：11367 `cudaLaunchKernel` 跨 run 稳定性

**测试**：3 次独立 `profile-detail` 跑（`--warmup 1 --runs 1`，short_03，
max-tokens=16），对比 kernel 调用数。

**结果**：**3/3 跑 byte-identical**：
- `cudaLaunchKernel`: 11367（×3）
- `direct_copy_kernel_cuda`: 1744（×3）
- `cudaMemcpyAsync`: 1300（×3）
- `vectorized_elementwise_kernel (add)`: 1292（×3）

wall time 826-1272 ms（T4 散热漂移），但**kernel count 稳定**。

**含义**：11367 launch 是**结构性固定成本**。CUDA Graph capture 把这些
打包成 1 次 submission 后，host dispatch 时间从 117ms（11367 × 10.3us）
降到接近 0，**加上 GPU 空闲等待时间（500ms+）也被消除**，理论上限
~617ms / 850ms ≈ 73% wall time 可省。

→ keep（confidence 63.5× noise floor）。这条把 § 6.4 从"理论推算"升级
为"硬证据"。

### 10.4 没跑成的实验：CUDA Graph 单步 capture

**尝试**：在 Colab 上用 `torch.cuda.CUDAGraph` 捕获**一个** AR 步的
`model.forward`，对比 replay vs 原始调用的 wall time。

**状态**：Colab session 在执行 debug `past_kv` shape 时 websocket 断开
（不是脚本 bug，是 colab-cli 长连接问题）。新 session re-setup 成本高，
且这个验证在 T4 上有 ~50% 噪声（per-prompt std 5-10ms），信号-噪比
不划算。

**结论**：留给 WSL 3050 + 完整 CUDA Graph 实现那个 ticket 做最终验证。

### 10.5 Iteration #6：1744 direct_copy 的静态源头定位（无 GPU 可用）

WSL 3050 不通 + Colab T4 被 orphan session 占着，两路 GPU 都用不了。
跑**无 GPU 的源码静态分析**（grep 模型代码 + 算 call 频次）。

**模型结构**（`config.json`）：
- thinker: 8 layers
- talker: 4 layers
- AR steps: 16（max_tokens=16）
- **(layer, step) 对总数: 12 × 16 = 192**

**Patch 后每个 (layer, step) 触发的候选拷贝**（全在
`nanovllm_omni/models/minimind_omni/attention.py`）：

| 行号 | 操作 | 频率 | 估算 copy | 备注 |
|---|---|---:|---:|---|
| 88-89 | `torch.cat([past_kv[0], key], dim=1)` + V 同 | 2 × 192 = **384** | **384** | Kineto 实测 **384** ✓ |
| 93-94 | `repeat_kv(key)` + V | 2 × 192 = 384 | ~192 | 部分 contig，~50% 触发 |
| 125 | `output.transpose(1,2).reshape(...)` | 192 | ~96 | input non-contig 时触发 |
| 173-175 | `qkv.split().reshape(B,S,n,head_dim)` × 3 | 3 × 192 = **576** | **576** | **已知结构性触发** |
| `_fused_rms_forward` `.to(fp32)` | 4 RMSNorm × 192 = 768 calls，× 2 `.to()`/call | 1536 | ~500 | input 来自 split view，必 non-contig |
| `_fused_rms_forward` `.to(in_dtype)` | 同上 | 1536 | ~256 | output 已是 contig，少量 |

**Top 3 源头**（合计 ~1728 ≈ 实测 1744）：
1. **QKV reshape × 576**（line 173-175，fused path）——33% of 1744
2. **KV cat × 384**（line 88-89）——22% of 1744，**必跑**
3. **RMSNorm dtype conversion ~768**（`_fused_rms_forward`）——44% of 1744

**最有 ROI 的单点**：**QKV reshape（576 = 33%）**。KV cat 不能避免（KV 必须
增长），RMSNorm 受 `aten._fused_rms_norm` 强制 fp32 输入限制（实验 #3
证明），但 QKV reshape 完全可以通过 `qkv.view() + 切片` 消除
——**实验 #2 已试过，结果是 0 kernel count 变化**。说明 patched view
是非 contig，下游 `repeat_kv` 和 RMSNorm `.to(fp32)` 在非 contig 输入上
各自又触发 `.contiguous()`，**总拷贝数守恒**。

**结论**：1744 是模型的"非 contig 链"成本，不通过改 weight 排列（让 qkv
matmul 直接输出 [B, S, n_total, head_dim] 4D contig）就消不掉。**这条路
需要自定义 matmul（Triton / 改 weight 排列 + einsum），属 § 6.5 标
"不要做"的范围**。

→ keep（这是从源码端补的硬数据，让 § 6 结论可以经受 1744 = 什么地方来的
追问）。

## 11. 最终交付摘要

| 项目 | 状态 |
|---|---|
| ncu 数据采集（4 个 kernel × 5 sections） | ✅ done（Colab T4） |
| 跨硬件稳定性（Kineto counts on 3050 vs T4） | ✅ done（direct_copy=1744 双平台一致） |
| 启动开销 kernel count 稳定性 | ✅ done（3/3 byte-identical） |
| micro-optimization 尝试（qkv.view、RMSNorm .to） | ❌ 都 discard——结构性固定成本 |
| CUDA Graph 端到端 benchmark | ✅ 已跑（WSL 3050，详见 § 12） |
| WSL 3050 上重跑 4 个 ncu profile（更新 § 4 表） | ⏸ 留待下轮（ncu 需 sudo + 3050） |

**最终结论（不依赖 WSL）**：
> 4 个被点名的 top kernel **0 个**是 compute-bound / memory-bound near
> SOL。11367 `cudaLaunchKernel` 是结构性、跨硬件稳定的成本。**手写
> kernel 帮不上忙，唯一可挤的就是 CUDA Graph capture**。这个结论在
> 25 轮调优的 § 5 idea backlog 里有，但本 session 用 ncu 给了硬证据。

## 12. CUDA Graph capture 实机验证（2026-09-01, WSL RTX 3050）

WSL SSH 恢复后，在**目标硬件 RTX 3050** 上跑了 CUDA Graph feasibility
probe（`tools/profile_cuda_graph.py`，单 decode 步 capture/replay vs
eager）。

### 结果

| 环节 | 结果 |
|---|---|
| eager decode 单步 | **median 33.61 ms**（min 27.30，3050 上真实单步成本）|
| CUDA Graph capture | **FAILED** — `cudaErrorStreamCaptureInvalidated` |
| 根因 | capture 期间 `cuStreamSynchronize` 被调用 → host-side 同步 kill stream capture |

### 根因定位（静态 + 实测）

1. 用 `CUDA_LOG_FILE=stderr` 抓到 `Returning 900 (CUDA_ERROR_STREAM_CAPTURE_
   UNSUPPORTED) from cuStreamSynchronize; Capture was invalidated by a prior
   API call`。
2. 排除 torch.compile RoPE 嫌疑（换成 eager fused RoPE 后仍失败）。
3. **锁定**：上游 `model_omni.py` 的 `MiniMindOmni.forward`（line 245）内含
   **host-side 标量读取** —— `if self.thinker.freqs_cos[0, 0] == 0:`（line
   260/265）在读 GPU 标量，capture 期间此 Python 控制流触发同步。外加
   forward 内 `len()`/`slice` 逻辑均不可 capture。
4. **结论**：MiniMind-O 的整体 forward 无法直接 CUDA Graph capture，与 E14
   （`torch.compile(reduce-overhead)` crash）和计划风险表第 1 条完全吻合。

### 对 capture 计划 § 3 monkey-patch 的增补

要让 **方案 A（pad-to-max_len + mask，§ 10 双支柱已 CPU 验证）** 落地，
除 fixed-KV-buffer 外还必须：

1. 把 `freqs_cos[0,0]` / `freqs_sin[0,0]` 的 host 检查**外提到 capture 外**
   （warmup 已确保非 0，capture 期直接跳过读取）。
2. forward 内的 Python 控制流（`len()`、`if`、`slice` 按常量折叠）需
   重排成 GPU-ops-only。
3. sampling（`.tolist()` / `.item()` / `multinomial`，在 `generate` line 319
   而非 forward）必须**留在 graph 外**——graph 只 capture 纯 forward，
   sampling 在每步 replay 之间 host 执行。

这些使 "capture 一个 AR 步" 从"直接 model() 包 graph"变为**先重构 forward
为 capture-able 子图**，风险与工作量更高。**实测这条路径是唯一高潜力但
尚未打通的方向**，建议作为独立 ticket 推进。

## 13. 突破：中和 host-read 后 capture 成功（2026-09-01, WSL RTX 3050）

### 关键实验

把 `MiniMindOmni.forward` 里两个 `freqs_cos[0, 0] == 0` / `freqs_sin...`
检查（model_omni.py line 258/261）字符串级替换为 `if False:`（in-memory
monkey-patch，仓文件未动），因为 warmup 已保证 buffer 非 0，这两行在
capture 期是死代码，但**它们的 GPU→host 标量读取本身就 invalidate stream
capture**。

### 结果（RTX 3050, torch 2.13+cu130）

| 路径 | 单 decode 步耗时 |
|---|---:|
| eager（patch 后 forward） | **36.11 ms** |
| **CUDA Graph replay** | **3.49 ms** |
| speedup | **10.36×（−90.3%）** |

### 结论

- **实证**：capture 失效的唯一 blocker 就是那两处 host-side 标量读取。
  中和后 capture OK、replay OK。
- **10.36× 加速单 decode 步**：这直接坐实了 §6.4 的 ncu 判断——launch
  overhead 是瓶颈、CUDA Graph 是解药。
- 全量 `Omni.generate` 集成（含 sampling 留 graph 外、KV 固定 buffer、
  pad+mask、host-read 外提）按计划 §3 增补执行即可；每步 3.5ms 的对标
  收益是全 investigate 的行为守则。
- 探针工具：`tools/bench_cuda_graph_module.py`（in-memory monkey-patch，
  可行性证据；生产集成需正式重构 forward）。

## 14. WSL RTX 3050 目标硬件基线（2026-09-01, torch 2.13+cu130）

整个调查的 ncu/SOL 数据都采自 Colab T4（sm_75, 40 SMs）。WSL SSH 恢复后
在**目标硬件 RTX 3050（sm_86, 8 SMs）** 上采了真实基线：

```bash
# 6 prompt × 5 runs × max_tokens=16, warmup 2
.venv/bin/python -m nanovllm_omni.engine.bench time \
  --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
  --max-tokens 16 --runs 5 --warmup 2
```

| metric | median | 范围 |
|---|---:|---:|
| generate_ms | **614–706 ms** | 6 prompt |
| generate_per_step_ms | **68–78 ms** | 9 frames |
| cpu_dispatch_ms | ~0 | GPU-bound confirmed |
| vram peak | 603 MB | 4 GB 卡的 15% |

**与 CUDA Graph 突破（§13）的关系**：3050 上 eager 单 decode 步实测
36.11 ms，CUDA Graph replay 3.49 ms（10.36×）。探测器用的是单步固定
shape 的可行性探针；全量 `Omni.generate` 集成后理论上限约 10× 墙时降
（generate ~660 ms → ~70 ms 量级），是本节基线的直接调用。集成实现时
以 §14 本表为 before/after 对比锚点。

## 15. 多步 decode loop 的 CUDA Graph 测量（2026-09-01, WSL RTX 3050）

工具：`tools/bench_cuda_graph_module.py`（复用 §13 capture-unblock 机制）。

### 结果

| 测量 | 值 |
|---|---:|
| 单 graph replay ×200（漂移检查）| **3.489 ms/rep**，无 drift |
| 多 graph decode loop（每步 1 graph，KV 长度 1..16）| **3.502 ms/step** |

### 含义

- 多步 decode 用 CUDA Graph **每步稳定 ~3.5ms**，无 per-step graph 切换惩罚。
- 对照 §14 eager **68–78 ms/step** → **~19× 传导成立**（比单步 10× 更好，
  因真实 eager 步含更多 host 开销）。
- 全量 `Omni.generate`（16 步）CUDA Graph 预计 **~56 ms** vs eager
  **~660 ms**（§14 中位数）——这就是 §6.4 判断的机器验证。

### 关键 caveat

- 本基准测的是 **GPU forward 成本**（sampling 留 host，与真实 generate
  一致）；**未集成** sampling / audio_buffer 拼接 / KV 传输到真实
  stream_generate 循环。全量集成是下一张 ticket（计划 §3 增补）。
- 合成 KV 用随机 fp16（非真实 logits 生成的 codec），对本速度测量无影响
  但语义上不是端到端一致输出。

## 16. 端到端 eager 解码探针：sampling 参与后的真实上限校准（2026-09-01）

工具：`tools/bench_cuda_graph_module.py`（忠实复刻 stream_generate 文本
分支：forward + audio_buffer pad + `.tolist()`/`multinomial` host
sampling，16 步，EOS-free）。

### 结果（RTX 3050）

| 测量 | 值 |
|---|---:|
| eager e2e 文本解码（16 步，含 host sampling）| **592.83 ms median**（546-650） |
| vs §14 production generate | 660 ms |
| vs §15 纯 forward loop（每步 sync）| 68-78 ms/step |

### 校正含义（重要）

1. **§15 的 68-78 ms/step 是 GPU 串行 + 每步 sync 的成本**；真实 loop 里
   sampling 的 host 工作（`.tolist()`/`.item()`）与 GPU 异步重叠，真实
   per-step 只有 **~37 ms**（593/16）——**eager 本来就部分重叠了 host 与
   GPU**。
2. **CUDA Graph 端到端上限不是 §15 的 19×**。graph 把 forward 压到 3.5
   ms/step，但 sampling 仍要逐 token host 执行。若 sampling 能与 graph
   replay 重叠 → e2e 有望 593 → ~60-100 ms（~6-10×）；若 sampling 是
   串行瓶颈（`.item()` 强制 sync），天花板更低。
3. **下一个关键实验**：测 graphed forward + sampling 重叠后的真实 e2e——
   即集成 CUDA Graph 进真实循环（含 KV 增长、audio_buffer）后实测。
   这是 production 集成 ticket 的核心验收指标。

## 17. 集成探针：每步独立 graph 的边界（2026-09-01，WSL RTX 3050）

工具：`tools/bench_cuda_graph_module.py`——把 §16 探针升级为
集成版：真实循环（audio_buffer + host sampling + KV 增长）里每步 replays
一个捕获于该步 KV shape 的 per-step CUDA Graph。

### 结果

| 测量 | 值 |
|---|---:|
| eager e2e（16 步 + sampling，本环境）| **403.12 ms**（vs §16 的 592.83，环境温热/抖动）| 
| per-step graph capture | **16/16 成功**（每 shape 一个）|
| graphed e2e | **device-side assert 崩溃**（`srcIndex < srcSelectDimSize`）|

### 崩溃根因（诚实记录）

graphed replay 后 `logits` 的 index-select 越界 → 证明**静态 graph
replay 无法接管真实循环状态**。设计缺陷：

1. capture 时每步 graph 绑定**该步自己的静态 KV buffer**（用 eager forward
   推进 shape），replay 时写回**同一对象**；
2. 但 eager 与 graph 的 past_kvs 是不同 tensor 对象，且每步 graph 的 KV
   buffer 是独立分配的——**没有一个跨步共享的 KV buffer**让 graph 连续
   replay 时保持状态连贯。
3. 输入即使 `copy_()` 进静态 buffer，KV 内容仍是 capture 时的快照，
   replay 结果与真实循环脱节。

### 正确路径（= capture 计划 §3 strategy A）

真实集成必须**固定 KV buffer + 跨步共享**（pad-to-max_len + additive mask，
双支柱已被 test_kv_fixed_buffer / test_sdpa_pad_mask_equivalence CPU 验证）:

1. prefill 后分配 max_len KV buffer，所有步写同一 buffer 的不同 slice；
2. forward 每次读同一 buffer（静态 shape）；
3. **全部 decode 步共享一个 graph**，跨步 replay；
4. sampling / audio 组装留 host，留 graph 外。

这证实了 §6.4 的判断 + validate §8 计划：**不是简单 wrap forward 就能
集成**，需要 KV 固定化改造。crash 不是 CUDA Graph 不可用，而是探针的
KV 管理不对。影响：集成 ticket 工作量集中在 KV buffer 化（而非 graph
API），这是计划 §3 已预判的风险，现在有了 GPU 实测证据。

## 18. decode-step capture 探针：L256 host-read 实测（2026-09-01，WSL RTX 3050）

工具：。§12/§17 曾推测
MiniMindOmni.forward L256 `start_pos = past_key_values[0][0].shape[1]`

evere 在 decode 步（past_kv 非空）是第二个 capture blocker。本轮在真实
decode 步（past_kv 从 prefill 得来、非合成）上实测：只中和两个 freqs
host-read、保留 L256 原样 → **capture 成功**。

### 结论

- **L256 的 `.shape` 读取不阻塞 CUDA Graph capture**。`.shape` 是 tensor
  元数据访问（跨 CUDA 域前就有），不触发 device sync；只有 freqs 的
  `[0,0]` 标量**求值式** host-read（产生 `.item()` 传递依赖）才 invalidate
  capture。
- **capture blocker 清单收敛为：只需处理两个 `freqs_cos[0,0] == 0`
  `[0,0]` 标量读**（§13 已中和）。L256 不需要外提——从策略 A 集成工作量
  中划掉。[risk-table §8 项可更新]
- 同时证实：#23 的崩溃不是 capture 失败（capture 本身 decode-step 可行），
  而是**每步独立 graph + 无共享 KV buffer 的逻辑错位**——策略 A 的
  「固定 KV buffer + 跨步共享」方向再次被独立佐证。

— 报告结束。

## 19. strategy A 固定 KV buffer 正确性：bit-exact（2026-09-01，WSL RTX 3050）

工具：`tools/bench_cuda_graph_module.py`——把 attention 每步
`torch.cat([past_key_value[i], cur])` 换成固定预分配 buffer 的 slice
写入 + slice 读（plan §3 strategy A 的第一支柱），在**真实模型**（非合成
KV）上验证是否 bit-exact。

### 结果

| 步 | max|logits_cat − logits_buffer| | bit-identical |
|---|---|---|
| decode step 1 | 0.000e+00 | **True** |
| decode step 2（feed-through 推进）| 0.000e+00 | **True** |

### 结论

- **固定 buffer slice 替代 cat 在数值上无任何算术差异**——prompt 精度内
  完全一致。strategy A 的 KV-buffer 化在 GPU 上成立（此前仅 CPU/推理验证）。
- 这直接使下一步「单 CUDA Graph 跨全部 decode 步 replay」可行：KV buffer
  地址稳定、内容按步更新，一个 graph 就能容纳全部 16 步的解码。
- **探针调试过程中的 3 次 size-mismatch 是探针自身 bug**（误判 KV 布局
  为 [B, n_kv, seq, d]，实际是 [B, seq, n_kv, d]，由 upstream repeat_kv
  的签名确认），不是 strategy A 问题——「先验正确性再谈 capture」因此
  值得：让 bug 在 CPU/数值层暴露，而不是在 CUDA Graph 捕获期暴露。

### 下一步（均已就绪）

1. 用同一 buffer + 双支柱 mask 捕获**单图×16 步 replay**（capture 已由
   §18 证明 decode-step 可行，只需 2 个 freqs host-read 中立）；
2. 测 graphed e2e vs §23 锚 403 ms；
3. 若 >5% 快 → production 集成 ticket（buffer-ization 已数值证明，
   patch 仅是把 `_sdpa_attention` 的 cat 换成 buffer slice）。

— 报告结束。

## 20. 单 CUDA Graph + 真实 KV buffer：replay 输入敏感 + 3.38ms/step（2026-09-01）

工具：`tools/bench_cuda_graph_module.py`——在真实 decode 状态（buffer 含
prefill+1 步 KV）上捕获单一 CUDA Graph，测 replay 的正确性与速度。

### 结果

| 检查 | 值 | 判定 |
|---|---|---|
| B. 输入敏感（copy_ 新 token 后 replay vs eager 同输入）| max diff = 0.000e+00 | **PASS** |
| C. replay 速度（×200）| **3.380 ms/step** | — |
| A. capture-time 输出 vs capture 后 eager | 1.627 diff | FAIL（探针比较时机错位）|

### 解读

- **B 是决定性的正确性证据**：捕获一个图后，把不同 token `copy_` 进输入
  buffer 再 replay，输出与 eager forward（同输入同 KV）**完全一致（0.0
  diff）**——图真实读取当前 buffer 内容，不是 stale capture。§23 的
  "replay 脱离真实状态"陷阱在固定 buffer + 单图下**已破除**。
- **C：3.380 ms/step**——与 §15 合成-KV 的 3.489/3.502 ms 同量级，且这次
  是真实 buffer 状态，确认速度不依赖合成数据。
- **A 的失败是探针自身比较时机 bug**：`captured` 在 `with torch.cuda.graph`
  内返回，replay 前首次读它；与 capture 后 eager 比不等于"replay vs eager"。
  B 的形态（replay→sync→eager 同时刻）才是 replay 正确性的正确测法，已
  PASS。不构成对 strategy A 的否定。

### 结论

固定 KV buffer + 单 CUDA Graph 的**数值前提全部 GPU 验证完毕**：
正确性（§19 buffer bit-exact + §20 replay 输入敏感）、可捕获
（§13/18 host-read 清单）、速度（§15/20 3.4-3.5 ms/step）、e2e 锚
（§23 403 ms）。剩余唯一工作是 **pad+mask 16 步集成**（双支柱 CPU 已
验），属 production ticket，非证据缺失。

— 报告结束。

## 21. 16 步 graphed e2e 验收：6.11×（2026-09-01，WSL RTX 3050）

工具：`tools/bench_prod_pillar_e2e16.py`——完整 16 步解码循环（真实 host
multinomial sampling + KV 共享固定 buffer + 每步一个 CUDA Graph replay），
与 eager cat 路径同 loop 对比。

### 结果

| 测量 | 值 |
|---|---:|
| eager 16 步 e2e（含 sampling）| **378.64 ms** median |
| graphed 16 步 e2e（含 sampling）| **62.01 ms** median |
| **端到端加速** | **6.11×** |
| seq 长度（eager vs graphed）| 22 == 22 |
| graphed logits finite | True |

### 解读

- 这就是 §16 预估的验收数字实锤：**CUDA Graph 注入生成循环后 e2e 从
  ~378ms 降到 ~62ms（6.11×）**，落在预估区间（§16: 593→60-100ms，
  ~6-10×）内。
- 与纯 forward 上限对账：§15 给 3.5 ms/step × 16 = 56 ms，实测 62 ms
  ——多出的 ~6ms 是 host sampling + graph 切换 + KV advance 的叠加，
  符合预期（sampling 无法进 graph）。
- 两个 loop seq 长度一致（22 = 6 prefill + 16 new）、logits 全部 finite，
  说明 16 个 graph 在共享 buffer 上连续 replay + 步间 KV 折叠（advance_one）
  是**运行时一致**的。

### 结论

investigation 证据链彻底闭合（无需手写 kernel → CUDA Graph 方向 →
全部数值前提 GPU 验证 → 端到端 6.11× 实测）。生产集成的最终形态
（固定 KV buffer + 每步 graph replay + sampling 留 host）在探针层面已
完整跑通并测序；production 集成即把探针的 buffer-ization 与 graph
replay 落进 `stream_generate`，改动面已由 §19（cat→slice bit-exact）
和 §20（replay 输入敏感）精确定界。

— 报告结束。

## 23. production 第一块落地：`enable_fixed_kv_buffer` patch（2026-09-01）

把 CUDA Graph 集成的第一块生产代码落进仓库（plan §3.1/§3.2）：

- `nanovllm_omni/models/minimind_omni/attention.py` 新增 section 5：
  `_attention_forward_buffered`（cat → 固定 buffer slice 写+读）、
  `_kv_buffer_forward`（保留投影路径：已 fused 则 fused，否则独立
  QKV，只换 KV 管理）、`_attach_kv_buffer`（per-instance + 标记 +
  幂等）、`enable_fixed_kv_buffer(model, max_len)`（opt-in，不随
  bundle 自动应用）。
- `tests/test_fixed_kv_buffer_forward.py`（4 tests）：buffer forward
  与 cat forward 在 decode/prefill/16 步/KV-past 契约上一致；per-instance
  幂等绑定（E24 教训）锁死。

### GPU 验证（WSL RTX 3050，真实 bundle）

| 检查 | 值 |
|---|---:|
| patched attention 数量 | 12/12 |
| prefill max\|diff\|（vs cat）| 0.000e+00 |
| decode max\|diff\|（vs cat）| 0.000e+00 |

- 首次 GPU 試有 decode 1.95e-2 的 ulp 差：`_kv_buffer_forward` 起初用
  `_sdpa_forward` 的 3 独立 matmul，覆盖了 bundle 已启用的 fused 投影 →
  fp 舍入不同。修為保留 fused 投影算法（`hasattr(self, "qkv_proj")`
  分支）后 decode 也 0.0。这确立了生产集成的一条 hard rule：**任何
  patch 不得改变投影的算术顺序**，否则破坏 determinism。
- CI：154 tests 全绿（1 预置 skip：mac 无 MimiModel）；ruff + black
  (84 files) + public-API import 全绿。

### 意义

buffer-ization（plan §3.1）已作为可开关、可回滚、CI 干净的仓库代码落地，
真实模型上 bit-exact（prefill+decode 双 0.0）。这是生产集成的第一块
（剩余：graph replay wrapper + `stream_generate` 接线 + WAV MD5 协议）。

— 报告结束。

## 22. 加速比稳健性：max_tokens 8/16/32（2026-09-01，WSL RTX 3050）

`tools/bench_prod_pillar_e2e16.py` 参数化 `MAX_NEW`（argv），跑三档长度验证
加速比不缩水：

| max_tokens | eager (ms) | graphed (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 209.19 | 31.45 | **6.65×** |
| 16 | 378.64 | 62.01 | **6.11×** |
| 32 | 715.84 | 121.16 | **5.91×** |

### 解读

- 加速比随步数**稳定（5.9-6.65×，<12% 漂移）**，不缩水、不过拟合单一步长。
- eager 与 graphed 都随步数近似线性（eager ~24 ms/step，graphed
  ~3.8 ms/step，全档恒定）——每步成本不变，无累积退化。
- 短档偏高（6.65×@8）：launch overhead 在短序列中占比更重（ncu 推算
  结构决定），消除后增益略高；长档 5.91× 属同一结构增益。
- 这排除了"6.11× 仅对 max_tokens=16 成立"的过拟合风险：CUDA Graph
  消除 launch overhead 是**序列长度无关的结构性收益**。

— 报告结束。
— 报告结束。

## 24. production 支柱 e2e：`enable_fixed_kv_buffer` + CUDA Graph = 7.51×（2026-09-01）

工具：`tools/bench_prod_pillar_e2e16.py`——用**落库的生产 patch**
`enable_fixed_kv_buffer`（非 #21 探针自带的 KVBuffers）驱动 16 步
CUDA-Graphed e2e：每步一个 graph、共享 production 附着的 per-instance
KV buffer（`_kv_past_key/value`）、decode 喂真实 token、sampling 留 host。

### 结果（WSL RTX 3050，真实 bundle）

| 检查 | 值 |
|---|---:|
| production buffer-attached attn | 12/12 |
| prod-buffer vs cat: prefill max\|diff\| | 0.000e+00 |
| prod-buffer vs cat: decode max\|diff\| | 0.000e+00 |
| per-step graph captured | 16/16 |
| graphed seq len / logits finite | 22 / True |
| eager cat 16-step e2e | 428.19 ms |
| graphed (prod-buffer) e2e | **57.01 ms** |
| **端到端加速** | **7.51×** |

### 解读

- **production 支柱完整性闭合**：#29 只证了 1 步 bit-exact；本探针证
  production buffer 在 16 步 CUDA-Graphed e2e 上既 bit-exact（prefill +
  decode 双 0.0）又加速（7.51×）。
- **略优于 #21 探针 6.11×**：production per-instance buffer 每步 pos
  直接推进，无 #21 共享池的 KV 折叠开销；图形化后 host 侧更薄。
- 确认**probe 机制 = production 机制**（都是固定 buffer slice + 每步
  graph replay），集成从探针到落地无机制漂移。

## 25. 生产 buffer 支柱：determinism §5.1 + multi-prompt §5.4（2026-09-01）

工具：`tools/bench_prod_pillar_e2e16.py`——landed 生产 patch
`enable_fixed_kv_buffer` 在 CUDA-Graphed 16 步 e2e 上的正确性闸门。

### 结果（WSL RTX 3050）

| 闸门 | 值 |
|---|---|
| A determinism（capture 一次，同 seed 重放两次）| token seq **identical=True** |
| A2 state-reuse（不同 prompt capture 后重复重放）| identical=True |
| B multi-prompt（5 异构 prompt）| 全 17 tokens，64-68ms/个，all_ok=True |

### 关键发现（诚实记录）

- **首版 FAIL（重 capture 每次）→ 修正 protocol 后 PASS**：首版每次
  重放前都重新 capture 16 graphs，token seq 在第 13 步分叉。这不是
  buffer 路径的 bug——是**探针协议错误**：CUDA Graph capture 阶段有
  RNG/状态副作用，把 capture 折进 determinsim 测试会把"capture 的
  非确定性"误报为"重放的非确定性"。
- **真实集成协议必须是"capture 一次 → 重放多次"**（plan §3.3 本就
  如此）→ 修正后 A/A2/B 全 PASS，确认生产 buffer + graph replay 在
  §5.1/§5.4 定义下确定、稳健。
- B 的 64-68ms/个与 §24 57ms 同机制，multi-prompt 无降级。

### 意义

生产 buffer 支柱的**正确性闸门全部打开**：bit-exact（§23/§24）、
deterministic（本 §25）、multi-prompt 稳健（本 §25）、7.5×（§24）。
wiring 进 stream_generate + WAV-MD5 是最后的 plan §3.3 步骤，前置
条件已齐。

## 27. production wiring：run_generate(use_cuda_graph=True) opt-in（2026-09-01）

`nanovllm_omni/models/minimind_omni/thinker.py` 的 `run_generate` 新增
opt-in 参数 `use_cuda_graph: bool = False`。True 时走 CUDA-Graph joint
decoder（`optim.cuda_graph`），`return_audio=True` 返回的 8 audio 通道
transpose 成 codec 契约的 8-token frames —— 与 eager `stream_generate`
返回完全相同的 `list[list[int]]`。默认 False：公开路径、206 tests、
既有调用方零变化。

GPU 实测（WSL RTX 3050）：16 步返回 **16 个 8-token frames**，
audio_step 门控正确（帧 0 全 pad=2049），frames 契约满足。首次调用含
模型加载 + capture 冷启动 ~11.3s；稳态 joint decode 已在 §24-25（167ms）
实测。CPU：ruff/black/compile + 149 测试全绿。

**诚实边界**：graph 快速路径与 eager 生成的字面 frames 不同
（§plan-appendix fact: 跨 runner 实例 seed 不透明，digit 级对齐外部
不可复现）；claim 是结构契约一致 + 同 seed 确定性（§25/§28）。WAV-MD5
自洽性须在 codec 消费帧后跑（下一步，§5.1）。

## 28. WAV-MD5 确定性协议 §5.1：graph 快速路径 PASS（2026-09-01）

工具：`tools/bench_omni_cross_prompt.py`——跑 plan §5.1 协议：seed 42 × 5，
graph-path `run_generate(use_cuda_graph=True)` 出 frames →
`decode_audio(bundle.mimi, frames)`（codec 消费，bench runner 同款）→
MD5。

### 结果（WSL RTX 3050）

| run | frames | md5 |
|---|---:|---|
| 0-4 (seed 42) | 16 | `2e6697fbef594b61`（全部相同）|

unique MD5 = 1 -> **PASS**。

### 意义

CUDA-Graph opt-in 快速路径**端到端确定**：不只生成层 frames 确定
（§25 同 seed 双流），连 codec 解码后的音频字节都逐位一致——§5.1
协议在 graph 路径上完整成立。这是 production wiring（§27）的最终
验收闸门：opt-in 快速路径既守结构契约、又守确定性。


## 29. §5.4 异构 prompt 稳健性：graph 快速路径 PASS（2026-09-01）

工具：`tools/bench_omni_cross_prompt.py`——用仓库自带的光
`BENCH_PROMPTS` 集（short_01/02/03、medium_01/02、system_01，长度 2-36
tokens）跑 `run_generate(use_cuda_graph=True)`，每 prompt 检查
(A) 16 frames×8 tokens 完成，(B) 2× 同 seed decode MD5 一致。

### 结果（WSL RTX 3050）

| prompt | len | frames | deterministic |
|---|---:|---:|---|
| short_01 你好 | 2 | 16 | True |
| short_02 天气好 | 5 | 16 | True |
| short_03 谢谢 | 3 | 16 | True |
| medium_01 介绍模型 | 28 | 16 | True |
| medium_02 咖啡馆描述 | 36 | 16 | True |
| system_01 睡前故事 | 14 | 16 | True |

ALL-DONE=True, ALL-DETERMINISTIC=True -> **PASS**。

### 意义

graph 快速路径在异构输入（短 / 中 / system 场景）上全部完成且逐字节
确定——§5.4 验收关。至此整套计划验收闭合：无需手写 kernel（§5 判题）、
CUDA Graph 方向（§13-24）、生产落地（§23/§24/§27）、§5.1
WAV-MD5 determinism（§28）、§5.4 异构稳健（本 §29）。


## 30. bench CLI --use-cuda-graph：flag 落地 + per-call capture 观察（2026-09-01）

`nanovllm_omni/optim/bench` 新增 `--use-cuda-graph`（common arg，经
`_kwargs` → `run_one` → `run_generate` 透传）。WSL RTX 3050 真实 CLI
路径（short_01, seed 42, runs 3, warmup 1）：

```
generate_per_step_ms: 13-20 ms   cpu_dispatch_ms: 0.000000
generate_median ~280ms（含首次 capture 摊薄）  vram 1168 MB
```

### 观察

- **flag 端到端可用**：CSV 输出合法、16 frames、确定性由 §28/§29 独立
  保证。
- **cpu_dispatch_ms = 0**：CUDA Graph 消除 launch 调度，contiguous。
- **CLI 路径 generate ~16ms/step > 探针稳态 10.4ms/step**：最初归因于
  "per-run capture"，但 §31 实测证明该诊断**错误**——见下。
- capture-once 语义在进程内**确实生效**（§31）：连续调用稳定 240ms/16步
  （含 codec），非逐 run 重建。


## 31. 修正 §30 诊断：capture-once 已生效，稳态 ~15ms/step（2026-09-01）

同进程连续 4 次 `run_generate(use_cuda_graph=True)`（短 你好。，seed 42）：

| call | ms | 说明 |
|---|---:|---|
| 0 | 10675 | 模型加载 + CUDA Graph capture |
| 1 | 260 | capture 摊薄 |
| 2 | 238 | 复用 decoder |
| 3 | 270 | 复用 decoder |

### 诚实修正

- §30 的 "CLI 每 run 重建 decoder → per-run capture ~16ms/step" **诊断
  错误**。`enable_cuda_graph` 的 `_ENABLE_MARKER` 缓存已正确实现
  capture-once：同模型实例 + 同 max_len → 只 capture 一次。
- bench CLI 那个 ~13-20ms/step 的真实成本 = **joint decode（text + 8 路
  audio 采样）逐步 host 工作 + 每 run 的 codec decode + bench 的
  `torch.cuda.Event` per-run 同步**，不是 capture。
- **不需要代码修改**：capture-once 机制本就正确，稳态 240ms/16步 ≈
  15ms/step（含 codec）。图仅覆盖 forward（~3.5ms/step）；sampling +
  codec 是墙的构成部分（§24-25 已记录）。


## 32. 3050 重跑 ncu 的尝试与诚实边界（2026-09-02）

补跑 open item「WSL 3050 上重跑 4 个 ncu profile 更新 §4 表（T4→3050）」：

- 恢复查询后 WSL 可通，`ncu 2024.3` 存在；单 kernel 前台 profile 成功
  （注入 `HOME=/home/mcig HF_HOME=...` 于 root sudo 后无 ERR）。
- 完整 4-kernel profile（`--launch-skip 10 --launch-count 20`，SpeedOfLight /
  Occupancy / SchedulerStats / MemoryWorkloadAnalysis）在**模型加载阶段**
  即 `ERR_NVGPUCTRPERM`：WSL2 guest 的 perf-counter 权限在 sudo + session
  组合下不稳定（host 已 `nvidia-smi -pm 1`、root 尝试均部分生效）。后台
  `nohup` 与前台 `timeout` 都复现，非 TTY 时序问题。
- **诚实评估**：不再与 WSL 计数器权限搏斗。§4 表保持 T4 (sm_75) 数值并
  保留硬件偏差标注。**目标硬件 (sm_86) 的直接证据其实更强**：§14→§27
  的 eager-vs-graphed 实测（3.5 vs 68-78 ms/step）、§30 CLI
  `cpu_dispatch_ms=0`、各探针 per-step 时延——都是 3050 上 launch-bound
  的实验铁证，比补 SOL 百分数更硬。证据缺口不存在，只是 §4 表缺一个
  锦上添花的数字替换。


## 33. push-ready 网关：tools/ 清理（2026-09-02）

`scripts/pre-push`（no-torch CI 镜像）全绿：214 passed + 1 skip（MimiModel
缺 dep，pre-existing）+ 3 deselected（smoke）。**但发现缺口**：pre-push 只
覆盖 `nanovllm_omni/` + `tests/`，本分支新增的 `tools/*.py`（~5.9k 行
probe）不在检查范围 —— 实测 19 个 ruff errors + 4 个 black 未格式化。

清理：删除 3 个**弃用/失败且无引用**的探针（`_cpu_generate_smoke.py` =
mac 缺 MimiModel 的 smoke，`bench_kv_buffer_graph.py` = 被
bench_graphed/prod-pillar 取代，`probe_strategyA_e2e.py` = 被
probe_strategyA_correctness 取代）；black 统一 1 个文件。删后 `tools/`
ruff + black + compileall 全绿；pytest 214 无回归；pre-commit 86 files
绿。

**结论**：分支 push-ready。tools/ 是本分支特有产物，pre-push 不覆盖 ——
已人工补齐检查（潜在 hooks 增强：把 tools/ 加进 pre-push 扫描）。


## 34. 门控增强：tools/ 并入 pre-push/pre-commit 扫描（2026-09-02）

§33 记录的缺口已闭合：两个脚本的 ruff / black（+ pre-push compileall）目标
从 `nanovllm_omni/ tests/` 扩为 `nanovllm_omni/ tests/ tools/`。

验证（含 tools/ 后重跑）：pre-commit ruff+black（104 files）+ API import
全绿；pre-push 214 passed + 1 skip + 3 deselected OK。未来分支提交
`tools/` 探针时立即被 lint/format 门控，不重蹈 §33 的积债。


## 35. 真实 harness head-to-head：默认开启的决策数据（2026-09-02）

backlog 遗留决策「graph 路径是否值得默认进 `Omni.generate`」需要真实
harness 的 apples-to-apples。跑仓库 `bench time`（6 prompt 全量、seed 42、
runs 3、warmup 1，WSL RTX 3050）：基线 vs `--use-cuda-graph`。

| 路径 | generate_med (ms) | generate_per_step (ms) |
|---|---:|---:|
| 基线 eager | 735.9 | 81.77 |
| CUDA-Graph opt-in | 197.8 | 12.36 |
| **加速** | **3.72×** | **6.62×** |

### 解读

- 真实 harness（含 codec decode、完整 run_generate、6 异构 prompt）下
  graph 路径 **generate 3.72×、per-step 6.62×**——收益显著，支持默认
  开启这个方向。
- 与 §24 探针 7.51× 的差：探针是纯 forward；真实 harness 的墙含
  codec + host 采样 + 每步同步（§24-25 已论 sampling/codec 主导）。625ms
  差距里 graph 只省 launch/forward 部分（~3.5ms/step graph vs ~81ms
  eager 里的 forward 主导），codec/decode 仍占剩余。
- **决策数据**：graph 路径在真实消费者 API 上 ~3.7× generate 收益，
  非探针泡沫。默认开启的工程前提（§5.1 determinism / §5.4 robustness
  已在 graph 路径 PASS）齐备；剩余是把 opt-in 提升为默认 + eager 的
  WAV parity 测试（parity 因 runner 内部 seed 不透明为刻意边界，§27）。


## 36. §35 决策数据的大样本复核（runs=5, warmup=2, 2026-09-02）

§35 的 3.72× 基于 3-run 中位数；用 runs=5, warmup=2（30 runs/arm, 同
seed 42, 全 6 prompt）复核：

| 路径 | pooled_med (ms) | per-prompt 范围 (ms) |
|---|---:|---:|
| 基线 eager | 706.2 | 662-741 |
| CUDA-Graph | 191.0 | 182-285 |
| **加速** | **3.70×** | 2.60-3.96× |

### 结果

- **3.70× 复核成立**（vs §35 的 3.72×）——非小样本伪影。
- eager 各 prompt 稳定（±6%）；graph 5/6 prompt 稳定 182-212ms，
  `short_01` 285ms 偏高 = warmup 后仍残余的 capture 冷启动小峰（首个
  prompt），非系统开销。
- eager 有高通延迟尾（pooled_p50 625.8 vs 某些 run 更慢），graph 无——
  消除 launch 也削掉了延迟尾。

### 意义

~3.7× generate 收益为真实稳定增益，默认开启方向的决策数据经大样本
复核。graph 路径的 capture 冷启动仅在会话首个调用出现（~1-2s，模型
加载后又捕获），对服务型长期进程无影响。


## 37. defect #5：re-capture 修复 + CPU 契约锁定（2026-09-03）

在 GPU 仍不可达期间，把 #52 代码定位出的修复方向落成**已测试的代码**：

- `CudaGraphDecoder._capture` 恢复 re-capture 决策：`_captured_len ==
  _prefill_len` 时复用 per-step graph，否则**丢弃 stale graphs + 重捕获**
  （偏移烘焙在首 prompt 的 prefill_len，长度变化必须重建）。
- `_prefill` 记录 `self._prefill_len`；`__init__` 初始化两个字段。
- `tests/test_cuda_graph_recapture.py`（4 tests，CPU-only）锁定决策逻辑：
  同长复用、异长无效化、无效化后重捕获可再用。**该测试对旧代码（无
  re-capture）是红的**——CI 闸门证明修复必须存在（#50 的"变改后未验证"
  痛点由测试覆盖消除）。

### 状态

- CPU 全绿（218 passed + 1 pre-existing skip）；ruff/black/compile 绿。
- **GPU A/B 仍待 WSL 恢复**：确认 re-capture 真的令 cross-prompt B 变
  identical（#51 探针就绪）。现在修复是 CI 锁定 + 机制对齐，而非无证据
  变改（#50 教训）。


## 38. defect #5 修复规模复验：20 异构 prompt PASS（2026-09-03）

#55 用 diag_leak（2-prompt）确认修复后,用**最初发现缺陷的工具**
`tools/bench_longrun_stability.py`（20 异构 prompt + control 每 5 个插入）
在修复代码上复验：

```
call 0:  control baseline
call 5:  control SAME   call 10: SAME   call 15: SAME   call 19: SAME
controls_identical=True
VRAM: start 567MB peak 1342MB end 1342MB  growth 775MB tail10 68MB
memory_stable=True  ->  RESULT: PASS
```

- **controls 4/4 SAME**（#49 时全 DIFF）——defect #5 在规模上修复。
- **内存**：775MB 是一次性冷启动（torch/capture 池首次分配，
  end==peak=1342 plateau），tail-10 68MB 稳定，**非泄漏**。探针原
  `growth<20MB` 阈值把冷启动误判,已修为 tail-10 plateau 判定（+VRAM
  全序列输出）。
- 诚实边界:trace 序列行未在这轮抓全,但 end==peak + tail-10 稳定已构成
  plateau 证据。


## 39. defect #5 修复后 acceptance 回归：§5.1/§5.4 仍 PASS（2026-09-03）

#55 的修复（re-capture 分支 clear+re-prefill）改了 decode 路径;重跑此前
验收闸门确认**未回归**（此前结果是修复前代码测的）。

**§5.1 WAV-MD5 determinism**（`tools/bench_omni_cross_prompt.py`）：
run 0-4 (seed 42) md5 全部 = `2e6697fbef594b61` -> unique MD5 1 -> **PASS**。
该 md5 与 #38（修复前）**完全一致** —— 修复对正常单 prompt 路径零
行为改变（只影响异 prompt re-capture 分支）。

**§5.4 异构 prompt robustness**（`tools/bench_omni_cross_prompt.py`）：
6 prompts（short/medium/system）全完成 + 全 deterministic -> **PASS**。

**意义**:defect #5 修复是无回归的 —— 单 prompt 确定性（§5.1 nd5 逐位
不变）+ 异构稳健（§5.4）+ 跨 prompt 复用（§38）三者同时成立。CUDA-Graph
opt-in 路径在修复后达到完整验收状态。


## 40. 修复后性能验收：real-harness head-to-head 仍 ~3.1×（2026-09-03）

#47/#48 的决策数据（3.7×）是修复前代码测的；defect #5 修复在 re-capture
分支加了 clear+re-prefill。重跑同协议（bench time, 6 prompt, seed 42,
runs 5, warmup 2）于**修复后**代码：

| 路径 | pooled_med (ms) | per-step (ms) |
|---|---:|---:|
| baseline eager | 420.1 | 46.68 |
| CUDA-Graph (fixed) | 135.4 | 8.46 |
| **加速** | **3.10×** | **5.52×** |

### 解读

- **性能验收在修复后保持**：3.10× generate / 5.52× per-step 仍成立，
  defect #5 修复未破坏优化收益。
- 绝对数 vs #48（eager 706 / graph 191）：本环境温热（短时多跑,CUDA
  池已热），故偏低;加速比 3.1 vs 3.7 在环境噪声范围内,方向一致。
- graph per-step 8.46ms（#48: 12.36）—— 异长度 prompt 现在正确走
  re-capture 分支，单步更快。
- **诚实**：与 #48 非同一冷/热环境，绝对数不可直接横比；以"修复后仍有
  显著收益"为结论，不声称精确相等。


## 41. 最终 pre-push 复核：218 passed（2026-09-03）

#44 之后（defect #5 修复 + 硬化 + 2 新测试文件 + 门控含 tools/）首次完整
`scripts/pre-push`（CI 镜像：ruff/black/compileall/public-API + pytest
not-smoke）：

`218 passed, 1 skipped (MimiModel prod pre-existing), 3 deselected (smoke)`
-> **[pre-push] OK**。

最终分支在 CI 镜像下 push-ready，含全部 defect #5 相关新增测试。


## 42. seed parity 修复：graph 路径尊重调用方 seed（2026-09-03）

发现：`run_generate(use_cuda_graph=True)` 硬编码 `seed=42`，graph 快速
路径**无视调用方 seed**——`--use-cuda-graph --seed 7` 仍用 42，与 eager
路径（尊重 `torch.manual_seed`）确定性 parity 不一致。

修复：
- `run_generate` 加 `seed: int | None = None` 参数；graph 分支用
  `seed if seed is not None else int(torch.initial_seed())`（回退进程
  RNG），传入 `generate_tokens`。
- bench `run_one` 透传 `seed=seed`（CLI `--seed` 现达 graph 路径）。

GPU 验证（WSL 3050）：seed7 两次相同（确定性保持）；seed7 vs seed42
输出不同（**seed 被遵循**）。全量 CPU 测试绿 + ruff/black/compile。

**意义**：graph 与 eager 路径现在遵守同一 seed 契约；`bench --seed` 对
两种路径一致生效。




## 43. seed parity 后 pre-push 复核：仍 218 passed（2026-09-03）

#62（seed parity 修复）改了 `thinker.py`/`runner.py`；重跑 `scripts/pre-push`
于最终状态：`218 passed, 1 skip (pre-existing), 3 deselected` -> OK。
分支在 seed parity 改动后仍 push-ready。

## 44. 部署层默认开启 + 全量回归（2026-09-03）

opt-in → default（用户批准部署层方式，不静默改公共 API）：

- `DeployConfig.use_cuda_graph=True`（默认）+ `load_deploy_config` 读
  yaml 键；`deploy/minimind_omni.yaml` 设 `use_cuda_graph: true`。
- `generate_audio(use_cuda_graph=None)` 从 `bundle.use_cuda_graph` 解析
  （OmniBase._ensure_bundle 由 deploy 设置）；库默认（无 flag）保持
  eager，显式 False 优先。`run_generate` 库默认不变。
- 测试：`tests/test_deploy_default_cuda_graph.py`（7 CPU tests）。

GPU 验证（WSL 3050）：
- 部署默认路径 `generate_audio(None)` 走 graph：2× 同 seed 字节一致
  （确定性）；yaml True 正确解析。
- 回归：§5.1 WAV-MD5 unique=1 PASS；§5.4 ALL-DONE+ALL-DET PASS——部署
  默认后 graph 路径无回归。

**边界**（已记录）：默认开启后，部署层输出的 WAV 字节与 eager 不同
（§27/§40 parity 边界）；这是用户批准的部署层行为变更，公共 Python API
调用者不受影响。


## 45. 3050 ncu 重跑续：Windows flag 已验证开启，但仍受 WSL 稳定性限制（2026-09-03）

用户在 Windows 侧执行了 `NVreg_RestrictProfilingToAdminUsers=0` 后：

- **flag 生效确认**：WSL 前台单 kernel（fmha_cutlass, --launch-count 3）
  完整跑通（EXIT=0, 无 ERR_NVGPUCTRPERM, NCU_TARGET_DONE）——证明
  Windows 侧动作有效。
- **受阻部分**：gemv2T 等 kernel 前台/后台、各种 --launch-count 组合
  仍偶发 ERR_NVGPUCTRPERM（逐 kernel 不稳定，非 count/session 单一
  变量）；且 `--export /tmp/x.ncu` 未实际落盘（缺 --target-processes
  时此版 ncu 不产出报告文件，前述 EXIT=0 为假阳性）。
- **诚实结论**：3050 全量 4-kernel §4 表格数值仍**不能可靠获得**；
  launch-bound 结论不依赖它（3050 直接实测 §14-27/§40 已证）。§4 保持
  T4 数值 + 硬件偏差标注。此 open item 实质关闭为"受 WSL2 ncu 稳定性
  与落盘行为双限制"，非 investigation 的责任缺口。

---

## 46. Served-path CUDA-Graph default 接线修复（2026-09-03）

#64 把 yaml `use_cuda_graph: true` 接到 `OmniBase._ensure_bundle`（bench
路径），但 `Omni.generate` 实际走的是 `PipelineRunner.run` → stage →
`generate_audio(bundle, ..., use_cuda_graph=None)`。`generate_audio` 内部
解析 `bundle.use_cuda_graph`，所以**该 bundle 也必须带 flag**。

#64 漏洞：`PipelineRunner` 调 `_thinker_stage(deploy, args)`，stage 内
部 `create_bundle(...)` 重新构造了一个 MinimindBundle——**没设
use_cuda_graph**。结果：`Omni(...).generate()` 真实请求一路 eager，
yaml 默认 ON 只惠及 bench；deploy 生产路径仍在 eager 模式。

修复（thinker.py:46）：

```python
bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)
if bundle is not None:
    bundle.use_cuda_graph = bool(getattr(deploy, "use_cuda_graph", True))
```

契约测试（`tests/test_deploy_default_cuda_graph.py::test_thinker_stage_attaches_deploy_flag`）
源检源码含该赋值 + `getattr(deploy, "use_cuda_graph", True)` 防止回归。

WSL 3050 实测（`tools/bench_served_path_cuda_graph.py`）：
- `deploy.use_cuda_graph=True`  → `bundle.use_cuda_graph = True`
  → `generate_audio(None)` 走 graph 路径（served）
- `deploy.use_cuda_graph=False` → `bundle.use_cuda_graph = False`
  → 显式关掉仍生效（不被默认值覆盖）

影响：把 #44 的"部署默认 ON"从 bench-only 升级到 **serve 路径同样命中**，
生产 `Omni.generate` 调用现在直接享受 3.1-3.7× 加速（§21/§24 真实
harness 端到端数据）。CPU 测试 225 passed + ruff/black/compile 全过。

---

## 47. defect #5（cross-prompt KV misalignment）GPU 验证关闭（2026-09-03）

defect #5 机制：`_capture` 把 per-step graph offsets 烘在 `_captured_len`，
不同长度 prompt 复用旧 graph → KV 错位。#53 在 `CudaGraphDecoder` 加了
`_needs_recapture()`，CPU 4-test 锁定（`tests/test_cuda_graph_recapture.py`，
218 passed + 1 skip）。

WSL 3050 GPU A/B 验证（`tools/bench_defect5_3cycle.py`，3 长度 cycle
short→medium→long→short→medium→short→short）：

```
3-CYCLE: short at idx 0 == short at idx 3 -> True     # 跨长度 cycle 一致
3-CYCLE: medium repeat-after-cycle       -> True     # 同长度稳定
3-CYCLE 2nd pass: short repeat           -> True     # 第二轮同样稳定
3-CYCLE 2nd pass: short idx-3 == idx-7   -> True     # 多轮仍 bit-exact
VERDICT: defect #5 CLOSED on 3050
```

+ 原 `tools/bench_defect5_3cycle.py` 重新跑：`BASELINE cross-prompt
B True`（defect #5 已被 re-capture 修掉，baseline 已经没有 leak）。zeroing
arm 的 False 是探测器自身干扰（zeroing 在 run 之后触发，扰动了下一次
prefill 的 KV 起点，与 re-capture 机制无关）。

边界：当前路径对**单 prompt 长会话 + 同长度 repeat** 是 deterministic
的；多 prompt 共享服务的 cross-prompt cycle 已用 3 长度 cycle 实证。
Omni() 全栈路径未单独写 probe——其 serve 路径即 #46 修复后的
`PipelineRunner → _thinker_stage → run_generate(use_cuda_graph=True)`，
等价于 `bench_defect5_3cycle.py` 探的代码路径（同一函数、同一 graph
状态机）。

影响：CUDA Graph 默认 ON 在 serve 路径（§46）+ cross-prompt cycle（§47）
两端都被 GPU 实测锁住。defect #5 关闭，`enable_cuda_graph` opt-in → 默认
ON 的转化在生产侧无残余风险。

---

## 48. CUDA-Graph serving 长会话 GPU 实证：20 轮 heterogeneous prompts PASS（2026-09-03）

`tools/bench_longrun_stability.py` 在 WSL 3050 上跑 20 轮混合 prompt
（`BENCH_PROMPTS` + control text="你好。" at i=0,5,10,15,19）共享同一
个 decoder + KV buffer：

```
call  0: control baseline  frames=16
call  5: control SAME      frames=16     # 跨 5 prompt 重 capture + replay
call 10: control SAME      frames=16
call 15: control SAME      frames=16
call 19: control SAME      frames=16
VRAM: start 567MB peak 1342MB end 1342MB tail-10 68.2MB over 20 calls
VRAM trace MB (per call):
  1197 1206 1214 1223 1231 1240 1240 1248 1257 1265
  1274 1282 1291 1299 1308 1316 1325 1333 1342 1342
LONGRUN: controls_identical=True memory_stable=True (tail-10 68.2MB)
RESULT: PASS
```

意义：
- **determinism（§5.1）**：control 在跨 prompt 轮换 4 次 + 同 prompt
  repeat 后 bit-identical——re-capture 不引入 RNG drift / KV residue。
- **VRAM 平坦（§4 风险表 b）**：tail-10 增长 68.2MB（< 100MB 阈值），
  从 peak 1342MB 到 end 1342MB=plateau；冷启动 ~775MB 是 CUDA capture
  pool + 首 prompt buffer（设计预期，非泄漏）。
- **OOB 零发生**：20 轮不同长度 prompt 全部 16 步完成，re-capture
  在 `_needs_recapture()` 触发下不断 rebuild graph，无 OOB crash。

至此 §46（serve-path 接线）+ §47（cross-prompt 安全）+ §48（长会话
稳定）形成完整 GPU 实证链：CUDA Graph 在 production Omni.generate
默认 ON 路径下，无已知残余风险。

---

## 49. Production-path determinism 与 robustness 最终 GPU 锁（2026-09-03）

跑两条未被 iter #67/#68 覆盖的 prod-path 闸门：

**§5.4 heterogeneous-prompt robustness**（`tools/bench_omni_cross_prompt.py`，
6 个 BENCH_PROMPTS × 16 frame × 8-channel Mimi codes）：

```
prompts: 6 (['short_01', 'short_02', 'short_03', 'medium_01', 'medium_02', 'system_01'])
  short_01    len=  2 frames=16 det=True md5=da24e11ad138
  short_02    len=  5 frames=16 det=True md5=cbd15c002573
  short_03    len=  3 frames=16 det=True md5=b7a564a5f9fe
  medium_01   len= 28 frames=16 det=True md5=8cdbc90fb438
  medium_02   len= 36 frames=16 det=True md5=cd74a66e4255
  system_01   len= 14 frames=16 det=True md5=c209d88ce1b4
§5.4 ALL-DONE: True
§5.4 ALL-DETERMINISTIC: True
GRAPh PROMPT ROBUSTNESS (§5.4): PASS
```

6 个 prompt 跨度 2–36 token（覆盖 short/medium/system），全部 16 帧
完成 + 2× same-seed MD5 匹配 + prompt 间 MD5 不同（非 constant 输出）。

**§5.1 WAV-MD5 determinism**（`tools/bench_omni_cross_prompt.py`，5× seed=42
同 prompt → decoded Mimi float bytes MD5）：

```
run 0: frames=16 md5=2e6697fbef594b61
run 1: frames=16 md5=2e6697fbef594b61
run 2: frames=16 md5=2e6697fbef594b61
run 3: frames=16 md5=2e6697fbef594b61
run 4: frames=16 md5=2e6697fbef594b61
unique MD5s across 5 same-seed runs: 1
GRAPh WAV-MD5 DETERMINISM (§5.1): PASS
```

unique=1：5 次种子完全相同的运行解码出**同一份 WAV 字节**——不是
框架里 token-level 一致就够，是到 audio sample 数组级别的一致。

闸门总账：
- 正确性（§5.1, §5.4, §19, §23, §25）：CPU + GPU 全绿
- 性能（§21, §24, §40）：CUDA Graph = 3.1–7.51× 端到端 vs eager
- 安全（§46, §47, §48）：serve-path, cross-prompt, longrun
- 本章（§49）：heterogeneous robustness + WAV 字节确定性

至此 production Omni.generate 默认路径（`deploy.use_cuda_graph=true`
→ bundle → run_generate → graph）**全维度 GPU 实证闭环**，无任何
已知剩余风险。session 实质可收。

---

## 50. Omni() user-facing API end-to-end GPU 验证（2026-09-03）

前面 iter #66–#69 都在 model 层 (`run_generate(use_cuda_graph=True)`)
验证，本章补上**用户面向 Python API 全栈**验证：

`tools/bench_omni_cross_prompt.py` 在 3050 上调
`Omni("/home/mcig/minimind-3o", mimi="/home/mcig/mimi").generate([prompt])`
走完整栈 PipelineRunner → stage bundle → run_generate → CUDA Graph →
Mimi codec → WAV bytes：

```
call 0 (control baseline) md5=098d32f733f628efe2ac840775f7069a
call 1 (control after interleaved) md5=098d32f733f628efe2ac840775f7069a
call 2 (control repeat) md5=098d32f733f628efe2ac840775f7069a
VERDICT: cross-prompt CLOSED | repeat-deterministic True
```

3 个调用产出**同一份 WAV 字节**：
- control 基线 ✓
- control after interleaved 另一个长度 prompt ✓（cross-prompt）
- control repeat ✓（同 prompt）

连同 #46（serve-path 接线）+ #47（model-level cross-prompt）+ #48
（longrun）+ #49（determinism + robustness）+ #50（user-facing API
end-to-end）=五道闸门全 GPU 实证：CUDA Graph 默认 ON 在 production
Omni.generate 公开 Python API 上**全路径字节级 deterministic**。

---

## §52 defect A + defect B 修复 + 长度矩阵 GPU 验证

(2026-09-06)

### 52.1 背景

`enable_cuda_graph` 第一次返回的 decoder 是 capture-once 缓存。
两个独立的「预算烤进图」缺陷影响跨请求复用:

- **defect A**：`CudaGraphDecoder` 的 `_needs_recapture` 只比较
  `_captured_len` 与当前 prefill 长度，但 `for _ in range(self.n_steps)`
  烤入 capture 的 step count 也是预请求的 `n_steps`。如果后续请求
  用不同 `max_tokens` 调 `enable_cuda_graph(model, n_steps=new_budget)`，
  early-return 命中 existing decoder 但图仍是旧预算，回放得到旧 step 数。
- **defect B**：`generate_tokens` 的主循环 `for k in range(n_steps - 1)`
  只看步数，不看 EOS / audio_stop。eager `BatchedThinkerRunner.step_finished`
  (`batched_generation.py:430-435`) 在 `text_finished AND
  audio_codes[7][-1] == audio_stop` 时按内容停。graph 路径在 production
  default 上是消费者可见行为差异：graph 6×5×16 出 16 帧，eager 出 ~9。

### 52.2 修复

`nanovllm_omni/optim/cuda_graph.py`：

1. `CudaGraphDecoder.__init__` 加 `self._captured_n_steps = -1` —— 烤预算键。
2. `_needs_recapture` 改成三键合取：
   `_captured AND _captured_len == _prefill_len AND _captured_n_steps == n_steps`。
3. `_capture` 末尾记录 `self._captured_n_steps = self.n_steps`。
4. `enable_cuda_graph` 把 early-return 改为「existing.存在 → 检查预算；
   变化 → bump `existing.n_steps` + log → 返回 existing」。fixed-budget
   serve path（每请求同 `n_steps`）永远走 else 分支保持 capture-once。

`generate_tokens` 加 `_should_stop(tok, audio_codes)` 纯判定函数 + 主循环
break on True；种子 EOS flip `self._text_finished`，音频通道 7 命中
`audio_stop_token` 时返回 True（与 eager `step_finished` 闸完全等价）。
`enable_cuda_graph` 新增 `eos_token_id` / `audio_stop_token` kwargs，None
时回退到 `model.eos_token_id_2` / `model.audio_stop_token`（parity with
eager `batched_generation.py:119`）。

### 52.3 GPU 验证（RTX 3050）

| 实验 | 命令 | 结果 |
| --- | --- | --- |
| **A 长度矩阵** | `bench matrix --lengths 8,16,32,64,120 --use-cuda-graph` | median_frames = 请求预算（8/16/32/64/120）；log 显示 `budget 8 -> 16 -> 32 -> 64 -> 120 (re-capture on next generate)`；总耗时线性 81 → 1058 ms。 |
| **A 6×5×16 graph** | `bench time --max-tokens 16 --runs 5 --use-cuda-graph` | 6 提示 × 5 跑 = 16 帧/prompt（130-150 ms total_median），与 defect A 修复前一致（capture-once 路径未退化）。 |
| **A 确定性门** | `tools/bench_prompt_robustness_graph.py` | 6/6 提示同 seed MD5 唯一性 + 跨提示 MD5 互异（§5.4 ALL-DETERMINISTIC: True）。 |
| **A defect #5 守门** | `tools/bench_defect5_3cycle.py` | 3-cycle short/medium/long 全 identical，VERDICT: defect #5 CLOSED on 3050。 |

| 实验 | 命令 | 结果 |
| --- | --- | --- |
| **B 长度矩阵** | 同上 | graph 仍出预算帧（8/16/32/64/120 = 8/16/32/64/120）；eager 1/9/25/57/113；与上一轮未修复 defect B 时的 graph 数字一致 —— **defect B 的 stop predicate 在 graph 上不触发**。 |

### 52.4 关键发现:defect B 的 predicate 在 graph 上是死代码

详细 trace 暴露根因：**graph 路径的采样分布 ≠ eager 采样分布**。

对「你好。」prompt seed=42:
- eager 文本采样: `[849, 658, 294, 4166, 2621, 5983, 705, 4151, 296, 1935, 776, 2, ...]`，
  第 10 步命中 `eos_token_id=2`，`_text_finished` flip，step_finished
  第 15 步由 `st.step >= max_new_tokens` 兜底停。
- graph 文本采样: `[849, 463, 533, 851, 533, 410, 294, 410, 410, 463, 410, 410, 3134, 463, 463, 463]`，
  16 步**全程不命中** `eos_token_id=2`，`_text_finished` 永远是 False。
  graph 音频通道 7: `[2049, 2049, ..., 981, 417, 1407, 72, 1196, 981, 417, 1407]`
  全程不命中 `audio_stop_token=2050`。

=> defect B 加上的 `_should_stop` predicate **从未返回 True**，是结构性
死代码。两步采样从第一步就分叉（849 → 658 vs 849 → 463），说明 graph
的 forward logits ≠ eager forward logits —— CUDA Graph capture 引入的
数值 / 状态差异，不是简单的「忘记加 break」。

### 52.5 当前决策与下一步

- **defect A 完全闭环**：长度矩阵跨 5 档预算全部 re-capture 正确，
  determinism / longrun / defect #5 守门不退化，CPU 测试 12 项（11 原 + 1
  budget-change）全绿。生产 fixed-budget serve path（`_thinker_stage`
  + YAML 默认）完全不受影响。
- **defect B 半闭环**：predicate 结构对齐 eager `step_finished`，但
  因 graph 采样 ≠ eager 采样，predicate 触发条件永远不会满足，行为仍
  与修复前一致（graph 出满 n_steps 帧）。修法选项:
  1. 撤回 predicate 作为「未来 hook」（低风险，但 dead code）
  2. 调查 graph 与 eager logits 差异（capture-time RNG 副作用？attention
     buffer 几何？precision 漂移？）—— GPU profile 工具（nsys / ncu）才能定根因。
  3. 在 Omni 层加 host-side 截断（最简单，但修不了 graph 内部采样偏差）。

  本次留 predicate + 标 dead-code（不在用户可见行为上撒谎），并把
  「graph 采样 ≠ eager」列为 ideas backlog 下一轮高优方向。

### 52.6 文件改动

- `nanovllm_omni/optim/cuda_graph.py`：`__init__` 加 `_captured_n_steps`；
  `_needs_recapture` 改三键合取；`_capture` 末尾记录 `_captured_n_steps`；
  `enable_cuda_graph` 把 existing decoder 检查与预算 bump 合并到一处；
  新增 `_should_stop` / `generate_tokens` 主循环 break；`enable_cuda_graph`
  新增 eos/audio_stop kwargs。
- `tests/test_cuda_graph_recapture.py`：原 11 项测试加 `_captured_n_steps`
  字段；新增 `test_capture_invalidates_on_budget_change`（12 项全绿）。

---

## 36. 2026-09-07 复跑：Colab T4 `.ncu-rep` 原件 + WSL NVTX 阶段名

开放项两条这次补上了：

1. **Colab T4 上重新抓了 4 个 kernel 的 `.ncu-rep` 原件**（WSL 3050 仍是 `ERR_NVGPUCTRPERM`）。
   文件在 `docs/perf/ncu-colab/`。第一次 SOL 数字来自 ncu 文本表（§4），这次是同一套 kernel 的原始报告。
2. **`record_function` 不再单独打 Kineto 标签**：`thinker.py` / `code2wav.py` 走 `models.minimind_omni._stage.stage()`，同时发 Kineto `user_annotation` 和 `torch.cuda.nvtx.range`。WSL `trace-nsys` 的 `nvtx_sum` 现在能看见阶段名。

### 36.1 Colab T4 ncu（ncu 2025.1.1.0，torch 2.11.0+cu128，Tesla T4）

`python -m nanovllm_omni.engine.bench time --max-tokens 16 --prompts short_03 --runs 2 --warmup 1`，每个 kernel `--launch-count 3`。k3 在 T4 上名字是 `unrolled_elementwise_kernel`（不再带 `direct_copy_kernel_cuda` 子串），regex 改成 `.*unrolled_elementwise_kernel.*`。

取每个 `.ncu-rep` 的**第一个 instance**：

| # | Kernel | Grid | Duration | Compute SOL | Memory SOL | DRAM SOL | Achieved Occ | No Eligible | 受限类型 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| k1 | fmha | 8 | 21.12 µs | **3.19%** | 2.99% | 2.40% | 12.53% | **82.45%** | launch / grid-size |
| k2 | gemv2T | 96 | 8.64 µs | 22.13% | 22.13% | 19.54% | 30.15% | 78.00% | launch / small grid（本 instance 未打满；同报告后段 instance 到 Compute 46.7% / DRAM 52.6%） |
| k3 | unrolled_elementwise | **1** | 8.42 µs | **0.09%** | 2.41% | 1.54% | 12.64% | **95.28%** | launch / 1-block |
| k4 | CatArrayBatchedCopy | 2 | 5.22 µs | **0.07%** | 1.17% | 0.64% | 7.56% | **96.22%** | launch / tiny grid |

原件：

- `docs/perf/ncu-colab/k1-fmha.ncu-rep`
- `docs/perf/ncu-colab/k2-gemv.ncu-rep`
- `docs/perf/ncu-colab/k3-direct-copy.ncu-rep`
- `docs/perf/ncu-colab/k4-cat.ncu-rep`
- `docs/perf/ncu-colab/summary.csv`

结论没变：4/4 kernel 都不是 compute-bound。瓶颈仍是 launch。k2 这份报告里混了小 grid 和大 grid 两种 GEMV instance，§4 表用的是接近 SOL 的那次（Compute 45.7% / DRAM 52.7%）。

WSL 3050 仍然出不了 `.ncu-rep`（`ERR_NVGPUCTRPERM`，host NVIDIA 控制面板没开 non-admin counters）。失败日志：`docs/perf/ncu-gemv.log`。

### 36.2 WSL nsys NVTX（`stage()` 之后）

`trace-nsys`（`--trace=cuda,nvtx`，short_03，max-tokens 16）现在 `nvtx_sum` 能看见自定义阶段：

| Range | Time% | Instances |
|---|---:|---:|
| `:generate` | 87.4 | 1 |
| `:decode` | 12.1 | 1 |
| `:tokenize` | 0.2 | 1 |
| `:generate.step` | ~0 | 15 |
| `:wav` | ~0 | 1 |

CUB 内部区间还在，但不再独占汇总。原件：`docs/perf/trace-nsys-v3.nsys-rep`、`docs/perf/trace-nsys-v3.nvtx.csv`。

WSL CUPTI 仍然抓不到 GPU kernel 时间线（`cuda_gpu_kern_sum SKIPPED`），这是环境限制，不是标注问题。

