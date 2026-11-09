# 文档地图

这个仓库的文档分四类，各有各的用途：

| 位置 | 放什么 | 什么时候写 |
|---|---|---|
| `docs/adr/` | **决策记录**：定了什么、为什么、代价是什么、边界在哪 | 任何"以后会有人问为什么"的选择 |
| `docs/dev/` | **实施文档**：某个迁移/子系统怎么做的，接口长什么样 | 动手做之前或做完之后 |
| `docs/perf/` | **实测结果**：硬件、配置、runs、warmup、wall、显存、阶段分解 | 每次重跑性能测量 |
| `docs/glossary.md` | **术语表**：这个词在这条线上指什么 | 引入一个新概念 |

规则来自 `AGENTS.md`：改了模型 pipeline、stage contract、batch 形状或性能结论，
**同一变更里**就得更新对应文档。`docs/perf/` 里的大 trace 可以留在本地产物，
但可复现的结论必须进版本库。

## 三条阅读路线

**路线 A — 只想知道现在能跑什么**
`README.md` → [ADR 0013](adr/0013-backend-abstraction.md)（backend 接缝）
→ [ADR 0014](adr/0014-dual-track-targets.md)（双轨目标）
→ [`docs/dev/mlx-track.md`](dev/mlx-track.md)（Mac 轨怎么装、怎么跑）

**路线 B — 想读 serving 的结论（这个仓库最有信息量的部分）**
[ADR 0010](adr/0010-concurrency-is-not-the-lever-2026-10-30.md)（并发不是杠杆）
→ [ADR 0007](adr/0007-minimind-throughput-2026-10-30.md)（talker 成本不在 O(n²)）
→ [ADR 0012](adr/0012-batched-requests-2026-10-31.md)（批处理是净亏，为什么）
→ [ADR 0013](adr/0013-backend-abstraction.md)（接缝画在哪）
→ [ADR 0021](adr/0021-mlx-kernel-write-our-own.md)（跨后端一致性怎么定义）

**路线 C — 想读设计决策的来龙去脉**
[ADR 0016](adr/0016-serving-is-a-teaching-subject.md)（serving 是教学对象）
→ [ADR 0014](adr/0014-dual-track-targets.md)（双轨）
→ [ADR 0017](adr/0017-rewrite-talker-in-repo.md) + [0020](adr/0020-rewrite-scope-vendor-only.md)（talker 重写与边界）
→ [ADR 0021](adr/0021-mlx-kernel-write-our-own.md) + [0023](adr/0023-mlx-track-order.md)（MLX 轨）
→ [ADR 0022](adr/0022-talker-rewrite-parity-gate.md)（CUDA 轨这一轮）

## ADR 清单

### 一致性标准

| ADR | 标题 | 一句话 |
|---|---|---|
| [0011](adr/0011-parity-standard.md) | 推理一致性（parity）标准 | P0 自洽 / P1 行为 / P2 数值三层；P2 不是 bitwise |
| [0021](adr/0021-mlx-kernel-write-our-own.md) | MLX varlen kernel 自研 + 输出一致定义 | 跨后端一致 = 三档判据 + greedy 优先 |
| [0022](adr/0022-talker-rewrite-parity-gate.md) | talker 等价重写的范围、门槛与验证环境 | 重写版的验收标准与返工量对照表 |

### 性能与瓶颈

| ADR | 标题 | 一句话 |
|---|---|---|
| [0003](adr/0003-profile-results-2026-09-13.md) | Profile 结果（2026-09-13） | 最早的 profile 基线 |
| [0006](adr/0006-four-family-vram-2026-10-30.md) | 四族显存占用对比 | 四个族的显存/耗时，同机同口径 |
| [0007](adr/0007-minimind-throughput-2026-10-30.md) | MiniMind-O 多请求吞吐 | talker 的成本分解；窗口化 KV 的由来 |
| [0008](adr/0008-smolvlm-dedup-and-throughput-2026-10-30.md) | smolvlm 去重、图片路径与吞吐 | 重复模型副本的修复 |
| [0009](adr/0009-smolvla-throughput-2026-10-30.md) | SmolVLA 吞吐 | CUDA Graph 被 lerobot 的 H2D 拷贝挡住 |
| [0010](adr/0010-concurrency-is-not-the-lever-2026-10-30.md) | 并发不是这里的吞吐杠杆 | 线程并发实测 1.8× 恶化并崩一个请求 |
| [0012](adr/0012-batched-requests-2026-10-31.md) | 请求生命周期与批处理 | stage 自持队列；bs=8 是净亏（+32%） |

### 架构与接缝

| ADR | 标题 | 一句话 |
|---|---|---|
| [0001](adr/0001-profile-analysis-protocol.md) | Profile 分析协议 | 测量的口径 |
| [0004](adr/0004-model-and-engine-boundaries.md) | 模型与引擎的边界 | |
| [0005](adr/0005-comparison-code-lives-in-bench.md) | 对比代码放 bench 不放 core | |
| [0013](adr/0013-backend-abstraction.md) | stage 层的 backend 接缝 | 接缝画在「一步执行」，不画在 tensor API 层 |
| [0014](adr/0014-dual-track-targets.md) | 双轨目标 | 3050 与 Mac 都做都要优化；不跨机相减 |
| [0019](adr/0019-two-tracks-in-parallel.md) | 两条轨并行 | 大块归 MLX，小块归 CUDA |

### 模型实现

| ADR | 标题 | 一句话 |
|---|---|---|
| [0002](adr/0002-smolvlm-fork-ar-integration.md) | smolvlm 的 fork AR 集成 | |
| [0017](adr/0017-rewrite-talker-in-repo.md) | talker 的模型代码自己写 | 窗口化 KV 一步到位 |
| [0020](adr/0020-rewrite-scope-vendor-only.md) | 「自己写」的边界 | 只重写 vendor 自定义代码，第三方库照用 |
| [0023](adr/0023-mlx-track-order.md) | MLX 轨的执行顺序 | thinker-only 起步，kernel 先 decode 后 prefill |

### 项目定位与目标

| ADR | 标题 | 一句话 |
|---|---|---|
| [0015](adr/0015-mlx-requires-varlen.md) | MLX backend 必须有 varlen | dense padded 不接受（上游实测 ragged 批 +143%） |
| [0016](adr/0016-serving-is-a-teaching-subject.md) | serving 本身就是教学对象 | 教学对象从"omni 推理链路"扩到整条链路 |
| [0018](adr/0018-offload-three-steps.md) | offload 三步走 | mmap → 量化 → KV SSD |

## 实施文档（`docs/dev/`）

> **注意**：`docs/dev/` 整个目录在 `.gitignore` 里（`.gitignore:103`），**这些是本机的工作文档，不进版本库**。ADR 0002 / 0005 / 0013 / 0023 引用的 `docs/dev/*.md` 在新 clone 上会缺。`docs/perf/` 相反：README 与 `aligned/` 是被跟踪的。

| 文件 | 内容 |
|---|---|
| [`mlx-track.md`](dev/mlx-track.md) | Mac 轨怎么装 mlx / mlx-lm、怎么跑、怎么装 Metal Toolchain |
| [`mlx-varlen-kernel-interface.md`](dev/mlx-varlen-kernel-interface.md) | 自研 Metal varlen kernel 的接口设计（尚未实现） |
| [`nanovllm-omni-paged-engine.md`](dev/nanovllm-omni-paged-engine.md) | paged 引擎的形态 |
| [`nanovllm-omni-smolvlm-ar-migration.md`](dev/nanovllm-omni-smolvlm-ar-migration.md) | SmolVLM 迁到 fork AR |
| [`nanovllm-omni-smolvla-arflow-migration.md`](dev/nanovllm-omni-smolvla-arflow-migration.md) | SmolVLA 的 ARFlow 迁移 |
| [`nanovllm-omni-diffusion-mirror.md`](dev/nanovllm-omni-diffusion-mirror.md) | diffusion 侧的镜像实现 |
| [`nanovllm-omni-sdturbo-diffusion-migration.md`](dev/nanovllm-omni-sdturbo-diffusion-migration.md) | SD-Turbo 的 diffusion 迁移 |
| [`vllm-omni-diffusion-reference.md`](dev/vllm-omni-diffusion-reference.md) | 上游 vllm-omni 的 diffusion 参考 |

## 性能结果（`docs/perf/`）

- [`bench-protocol-aligned.md`](perf/bench-protocol-aligned.md) —— 测量协议
- [`README.md`](perf/README.md) —— 该目录的说明
- `aligned/` —— 对齐后的原始结果
- `*.csv` —— 各次测量的原始数据

**注意**：`docs/perf/` 下的 CSV 与 trace 是**本地产物**，不进版本库；
可复现的结论在 ADR 里，原始数据的路径写在 ADR 的实施记录中。

## 术语

新概念先查 [`docs/glossary.md`](glossary.md)。里面每条都给了定义和"在本仓库哪儿"，
包括 `varlen` / `cu_seqlens` / `paged KV` / `slot 复用` / `窗口化 KV` / `vendor layer` /
`dispatch-bound` / `EXACT / TOP_K_MATCH / FAIL` 等。
