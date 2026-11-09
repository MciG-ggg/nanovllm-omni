# ADR 0019: 两条轨并行——大块归 MLX，小块归 CUDA

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0014（双轨目标）、0015（MLX 要 varlen）、0017（talker 重写）、
  0018（offload 三步）

## 问题

ADR 0014 定了双轨，两条轨的工作量极不对称：

- **MLX 轨**的大块是 ADR 0015 的 varlen attention 子系统移植：
  `pagedattention.metal` 125 KB + block table 管理 + `cu_seqlens` 构造 +
  KV 写入协议 + rope，以及它们的 Python 侧胶水。这是一个多周的工程，
  而且中间没有可交付的中间产物（kernel 没通就没有 varlen，没有 varlen就没有真 CB）。
- **CUDA 轨**的小块是 ADR 0017 的 talker 重写 + 窗口化 KV、ADR 0012 遗留的
  slot 复用。每一块都是几十到几百行，做完就有数字。

如果串行，先做哪条都有问题：先 CUDA 轨，MLX 轨的启动被推后；先 MLX 轨，
中间几个月拿不出任何可对比的结论。

## 决定

**并行推进。分工按"块的大小"分，不按轨分：**

| 谁做 | 做什么 | 为什么归它 |
|---|---|---|
| MLX 轨 | varlen attention 子系统移植（ADR 0015）；MLX backend 的 paged KV 与批 decode | 这是唯一的大块，晚开工就晚完成 |
| CUDA 轨 | talker 重写 + 窗口化 KV（ADR 0017）；thinker 侧 slot 复用（ADR 0012 遗留） | 每块做完就有数字，能持续产出结论 |

**交汇点是 ADR 0017**：talker 重写出来的代码是两条轨共用的资产。
CUDA 轨写它的时候要按"将来能有一个 MLX 版"的结构写——不依赖 vendor 的 HF 风格，
attention 的接口留出 varlen 的位置。

## 后果

- **两条轨的代码所有权要分清**，否则会出现两个方向同时改 `backend/` 与 stage 层。
  建议：`nanovllm_omni/backend/mlx_backend.py` 与 MLX 的 kernel 子目录归 MLX 轨；
  `backend/base.py`（协议）与 `engine/` 归两边共同维护，改协议要两边都同意。
- **协议是共享边界，不是共享实现**。ADR 0013 的 `StageRunner` 协议在两条轨上
  都成立，但 MLX 侧因为要 varlen，会需要协议加成员。加协议成员是一次显式的协商，
  不是随手改。
- **数字要分开记**。ADR 0014 已定"不跨机相减"；这里再加一条：
  **不跨轨相减**。CUDA 轨的 talker 吞吐不能拿来推 MLX 轨的预期。
- **CUDA 轨的产出要能独立讲**。ADR 0016 定了 serving 是教学对象，
  所以 CUDA 轨的每个结论都要有自己的 ADR，不能依赖 MLX 轨也做完才成篇。

## 边界

- 本 ADR **不**承诺两条轨同时有进展。人力上如果只能推进一条，先 MLX 轨
  （大块早开工），CUDA 轨的 slot 复用可以随时插队——它足够小。
- 本 ADR **不**决定 MLX 轨是移植 `vllm-metal` 的 kernel 还是自研
  （ADR 0015 已记为待评估）。
