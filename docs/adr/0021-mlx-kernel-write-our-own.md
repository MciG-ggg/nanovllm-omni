# ADR 0021: MLX 的 varlen kernel 自研最小集，输出一致定义为 token 级

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0015（MLX 要 varlen）、0019（两轨并行）

## 问题

ADR 0015 定了"MLX 必须有 varlen"，并把路线写成"移植 `vllm-metal` 的 attention 子系统"
（`pagedattention.metal` 125 KB + block table + `cu_seqlens` + KV 写入 + rope，
外加 52 个相关 Python 文件）。

复核后这条路偏重了：`vllm-metal` 的 attention 子包里绝大多数是**我们用不到的东西**
——MLA、GDN、bidi、linear attention、Mamba2、KDA、turboquant、M5 NAX。
我们的模型只有 4 层、768 hidden、单个 head 组。

同时，ADR 0015 当时留了一个未定的验收标准：「要求输出与 CUDA 一致」。

## 决定

### 一、kernel 自研，只写自己需要的

不移植 `vllm-metal`，自己写一个**最小集**。需要的能力：

| 能力 | 说明 |
|---|---|
| paged decode attention | q 是每序列 1 个 query；k/v 按 block table 从 KV 里读；按各序列的 context_len 屏蔽 |
| paged prefill attention | q 是本步所有 scheduled token 拼成的扁平序列，用 `cu_seqlens` 划界 |
| KV 写入 | 把新算出的 k/v 写进 block table 指定的 slot（对应 fork 的 `store_kvcache`） |
| RoPE | 优先用 MLX 自带的（`mx.fast.rope` 一类），不自写 |
| reshape / 布局转换 | 把 `[B, H, T, D]` 与 `[T, H, D]` 之间来回搬 |

**明确不做**：MLA、GDN、bidi、linear attention、Mamba2、KDA、turboquant、NAX。

这个规模比移植小一个量级，也让「教学项目」的名副其实——kernel 的每一行都能解释。

### 二、「输出一致」的定义

**逐位一致做不到**：Metal 与 CUDA 的浮点归约顺序不同，fp16/bf16 的舍入也不同，
kernel 的分块策略还不同。同一份权重、同一 seed，两边生成**不同的 token** 是可能的。

可执行的标准定为：

> **token 级一致**：相同权重、相同 seed、相同采样配置下，MLX 与 CUDA 生成的
> token 序列**相同**；数值差异在事先声明的容差内。

这落在 ADR 0011 的 **Compatible** 层（任务效用与安全不劣化），不是 Exact 层。
Exact 层的逐位一致只在**同一后端内部**成立（重写版 vs vendor，ADR 0017）。

**这不是妥协，这正是要教的东西**：跨后端 serving 的一致性只能到 token 级，
比同后端低一档。想守住这一档，靠的是采样参数与停止条件的精确对齐，
而不是指望两边浮点一样。

## 后果

- **ADR 0015 的路线要改**：从"移植 `vllm-metal`"改成"自研最小集"。
  ADR 0015 的**结论不变**（MLX 必须有 varlen），变的是实现路径与成本量级。
- **自研 kernel 的验收要对着 CUDA 轨的 token 序列**，不是对着数值。
  这要求两边的采样配置严格对齐（temperature / top-k / top-p / seed / max_tokens），
  否则连 token 级都保证不了。
- **MLX 轨第一站仍是 SmolVLM**（没有 talker 依赖，能独立成链）——
  但 SmolVLM 的视觉塔是 SigLIP（第三方库），MLX 版要走 mlx-vlm 的移植。
- **两轨的对比口径**：MLX 轨出数字后，与 CUDA 轨比较的是**趋势**
  （ragged 批 vs 齐长批的相对变化），不是绝对值（ADR 0014：不跨机相减）。

## 边界

- 本 ADR **不**承诺 Metal kernel 的性能能对上 `vllm-metal`。对标是"功能等价"，
  不是"速度等价"。
- 本 ADR **不**处理 M5 NAX 路径（M2 Air 没有 tensor 单元）。
- 本 ADR **不**改 ADR 0014 的双轨结论，也不改 ADR 0019 的分工。
