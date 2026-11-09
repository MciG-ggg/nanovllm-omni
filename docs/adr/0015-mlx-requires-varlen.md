# ADR 0015: MLX backend 必须有 varlen——移植 vllm-metal 的 attention 子系统

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）

## 问题

MLX 的 `mx.fast.scaled_dot_product_attention` 有 `mask` 参数，但**没有 `cu_seqlens`**。
即 MLX 没有 variable-length attention：批内长度不齐时只能补齐到最长。

上游 `vllm-metal` 的实测给出了这个选择的代价（Qwen3.6-35B-A3B 4bit，8 请求 ~6000 prompt token）：

| 批 | mlx_lm（padded） | oMLX | llama.cpp | vllm-metal（varlen） |
|---|---:|---:|---:|---:|
| 750 × 8（长度齐） | 4.52 | 7.32 | 5.29 | **3.87** |
| 3000 + 430 × 7（长度不齐） | 10.99 | 7.22 | 5.33 | **3.64** |
| 变化 | **+143%** | −1% | +1% | −6% |

padded 路径在 ragged 批上崩 143%，varlen 路径基本不受影响。

所以 MLX backend 有两个水位可选：

- **降级**：接受 dense padded，接口变成真有两个实现，但这个 backend 没有 continuous
  batching 的意义（batch 一旦不齐就退化）。
- **到位**：给 MLX 一条 varlen 路径。

## 决定

**必须有 varlen。** MLX backend 的水位是「有 varlen 的真 CB」，不降到 dense padded。

理由是这个仓库的目的。如果 MLX backend 只能吃 padded 批，那它和"没有 continuous
batching 的 serving"是同一件事，而 ADR 0014 已经定了 Mac 轨也要出性能结论——
一个不出真 CB 结论的 backend 对这条轨没有价值。

代价要写清楚：`vllm-metal` 的 Metal 源码规模是

| 文件 | 大小 |
|---|---:|
| `vllm_metal/metal/kernels_v2/pagedattention.metal` | **125 KB** |
| `pagedattention_nax.metal` | 40 KB |
| `pagedattention_tiled.metal` | 32 KB |
| `reshape_and_cache.metal` | 6 KB |
| `rope.metal` | 2 KB |
| `sampling.metal` | 4 KB |

17 个 `.metal` 文件合计 304 KB，`vllm_metal/` 下 204 个 Python 文件。

**"复用 vllm-metal"不是接一个函数，是移植一个 kernel 家族**：block table 管理、
`cu_seqlens` 构造、KV 写入协议、rope 实现，以及它们的 Python 侧胶水。

## 后果

- `vllm-metal` 是 **Apache-2.0**（已查，仓库根 LICENSE），法律上可以复用；
  移植时必须保留署名与 NOTICE。
- 这是一笔**独立的大工程**，要拆成子任务，不能塞进 backend 抽象里顺手做：
  1. 抽出 kernel（Metal 源码 + 编译 + 加载）
  2. paged KV 的物理层（block table、`slot_mapping`、block 分配）
  3. `cu_seqlens` 构造（prefill / decode 两条路径）
  4. KV 写入协议（对应 fork 的 `store_kvcache`）
  5. 接到 `MlxStageRunner.prepare_*` / `forward`
- **`mlx_lm.generate.BatchGenerator` 不作为答案。** 它自带一套增量批处理
  （`PromptProcessingBatch` 支持 extend / filter / split），看起来能省事，但那样
  continuous batching 的实现就不在这个仓库里了——被讲的东西变成依赖。
  可以读它做对照，不可以直接用。
- MLX 轨的 talker 仍然缺席：vendor 的 `TalkerModule` 没有 MLX 版本。
  这不影响本 ADR——本 ADR 只管 AR 阶段的 varlen。

## 边界

- 本 ADR **不**决定 MLX backend 移植自研还是直接依赖 `vllm-metal` 作为 Python 包。
  两种都可行（前者可控，后者省事但绑上游版本），需要单独评估。
- 本 ADR **不**保证 M5 NAX 路径可用——`pagedattention_nax.metal` 依赖 M5 的 tensor 单元，
  M2 Air 上没有。
