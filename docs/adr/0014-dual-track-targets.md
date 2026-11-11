# ADR 0014: 双轨目标——3050 与 Mac 都做，都要优化

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0006（四族显存）、0007（minimind 吞吐）、0010（并发不是杠杆）、
  0012（批处理）、0013（backend 接缝）

## 问题

backend 抽象做完之后，两条路摆在面前：

- **3050（WSL，4 GB 独显）**：fork 的 flash_attn 已经是 varlen，CUDA Graph 能捕获，
  ADR 0007/0012 的全部实测都在这台。它能给的是**真 continuous batching**（调度器
  每步重选，请求可中途加入、完成即离场，ADR 0026）**+ bs=1 图捕获**。CB 默认关闭
  （`max_num_seqs: 1`），原因见 ADR 0012/0026。
- **M2 Air（16 GB 统一内存）**：有 `mx.compile`，但 MPS 与 MLX 都没有 flash attention，
  没有 varlen；triton 无 macOS wheel，所以连 torch backend 都 import 不了。
  它能给的是**统一内存的容量**（11.84 GiB 工作集）和**无 CUDA 依赖的本地推理**。

两台机器的能力不重叠。选一台就等于放弃另一台的一半。

## 决定

**都做，都要优化。** 不选边。

这不是"两边都做一点"，而是两条轨道各自跑到能出性能结论：

- **CUDA 轨**：talker 批处理 → thinker 侧 slot 复用 → 真 CB。量的是吞吐与 TTFT。
- **MLX 轨**：varlen attention 子系统 → MLX backend 的批 decode → 真 CB。
  量的是同一组指标在 Apple Silicon 上的数。

## 后果

- **两套 backend 都要过同一批测试**（`tests/test_backend_wiring.py` 是起点，不够）。
- **性能结论按机器分开记，不跨机相减。** ADR 0006 已经有一次教训：
  同一族两次测量差 803 MiB 没定位，原因是口径不同。跨机相减更不成立。
  每条性能结论必须写清：机器、backend、batch 形状、是否 graph/capture、采样配置、runs、warmup。
- **一条轨道的进展不能默认另一条也成立。** ADR 0012 的「bs=8 是净亏」是 CUDA 轨的结论；
  MLX 轨上 talker 仍然没有实现（vendor 的 `TalkerModule` 没有 MLX 版本），
  所以 MLX 轨的 CB 数字要重新测，不能引用。
- **ADR 0013 的 backend 接缝是两条轨道的前提，不是其中一条的。** 它的价值在这里才兑现。

## 边界

- 两条轨道**不要求同时进行**，但每条都要有可复现的结论。
- Mac 轨的模型覆盖面天然更小：4 族里只有 MiniMind-O 的 thinker 与 SmolVLM 的语言侧
  有 MLX 等价物；talker、code2wav（Mimi）、SD-Turbo（diffusers）、SmolVLA（lerobot）没有。
- "都要优化"意味着 Mac 轨不接受 dense padded 的降级方案，见 ADR 0015。
