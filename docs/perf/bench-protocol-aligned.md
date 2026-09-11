# Apples-to-apples bench protocol

> **TL;DR**: 之前仓库里的 baseline 数字（`~928 ms`、`846 ms`）和当前
> `628–1038 ms` 数字**不是同一条件下测的**，没法直接做 before/after。
> 本文给出一个固定的 4-cell 测量矩阵，每个 cell 用同一 prompt set、
> 同一 runs 数、同一脚本，落成同一格式的 CSV / markdown。

## 1. 为什么需要这个协议

仓库里有两组 bench 数字：

| 数字 | scope | 来源 |
|---|---|---|
| `~928 ms` E2E | 项目目标 statement 里写的“起点”，**不是实际测量** | `docs/perf/minimind-omni-under-500ms.md` line 3 |
| `846.38 ms` thinker-decode primitive | 实测，e8df750 commit，torch 2.13 | 同上表 line 13 |
| `320 ms` thinker-decode primitive | 实测，融合栈跑完后 | 同上 |
| `~42 ms` thinker-decode primitive | 实测，CUDA Graph 跑完后 | `optim/cuda_graph.py` docstring |
| `628–1038 ms` E2E | 实测，`8f6d522` commit，torch 2.14，**仅 fusion on、CUDA Graph off** | `docs/perf/tk005-rtx3050.md` |

这些数字：

1. **measurement scope 不同**：thinker-decode primitive vs full E2E
2. **torch 版本不同**：2.13 vs 2.14
3. **CUDA Graph 状态不同**：628–1038 ms 是 fusion on + graph off，42 ms 是 graph on
4. **baseline 928 ms 没有 CSV / 没有测量脚本**，只是目标 statement

要写一份 before/after，**必须用同一脚本在同一 torch 上重测 baseline**。

## 2. 4-cell 测量矩阵

每个 cell = 同一 prompt set（6 prompts）× 20 runs × max_tokens=16，
落在 `docs/perf/aligned/<cell>.csv`。

| Cell | fusion | thinker CUDA Graph | talker CUDA Graph | 旧术语 |
|---|---|---|---|---|
| **A** baseline | off | off | off | “vanilla unoptimized” |
| **B** fusion-only | on | off | off | “628–1038 ms 的对照” |
| **C** fusion + thinker graph | on | on | off | “32 layer capture 不全” |
| **D** full stack | on | on | on | “production 当前态” |

> talker CUDA Graph 默认由 deploy YAML `use_talker_cuda_graph: true` 控制，
> bench CLI **目前不直接 expose**。要么临时改 deploy YAML、要么后续给
> bench 加 `--use-talker-cuda-graph` 镜像 thinker 那条。本次协议默认
> **deploy YAML 不动**——Cell D 的 talker graph 行为以现有 deploy 配置为准，
> cell A / B / C 都没启用。

### Cell A 命令（baseline）

```bash
python -m nanovllm_omni.bench time \
    --pipeline full \
    --runs 20 --warmup 2 \
    --max-tokens 16 \
    --temperature 0.7 --top-p 0.9 --seed 42 \
    --enforce-eager \
    --out docs/perf/aligned/A-baseline-enforce-eager.csv
```

`--enforce-eager` 跳过 4 个 fusion monkey-patch
（`enable_sdpa_decode` / `enable_fused_rmsnorm` / `enable_fused_projections` / `enable_fused_rope`）。
不传任何 `--use-thinker-cuda-graph`，graph 默认 off。

### Cell B 命令（fusion only，与现有 628–1038 ms 对照）

```bash
python -m nanovllm_omni.bench time \
    --pipeline full \
    --runs 20 --warmup 2 \
    --max-tokens 16 \
    --temperature 0.7 --top-p 0.9 --seed 42 \
    --out docs/perf/aligned/B-fusion-only.csv
```

没有 `--enforce-eager`，bundle loader 默认应用 4 个 fusion patch；
graph 仍然 off。这条**应该**复现 `tk005-rtx3050.md` 的 `628–1038 ms`
数字（前提：torch 版本、commit、prompt set 与那次测量一致）。

### Cell C 命令（fusion + thinker graph）

```bash
python -m nanovllm_omni.bench time \
    --pipeline full \
    --runs 20 --warmup 2 \
    --max-tokens 16 \
    --temperature 0.7 --top-p 0.9 --seed 42 \
    --use-thinker-cuda-graph \
    --out docs/perf/aligned/C-fusion-thinker-graph.csv
```

启 thinker CUDA Graph，注意 `--pipeline full` 走的是 `Omni.generate`
路径，**`Omni.generate` 当前不读 `--use-thinker-cuda-graph`**——bench
代码里有 `kwargs.pop("use_thinker_cuda_graph", None)`。所以这条命令
**对 full pipeline 没有效果**，需要后续 bench 改动。**先记为
已知 gap**：Cell C 在 `--pipeline full` 下当前是空跑（等同 Cell B）；
要真正测 thinker graph on 的 E2E，需要改 bench 让 `Omni.generate`
接收这个 flag。

### Cell D 命令（full stack）

```bash
# 1. 临时改 deploy YAML 把 talker CUDA Graph 打开（默认就是 true，确认即可）
grep use_talker_cuda_graph nanovllm_omni/deploy/minimind_omni.yaml
# 期望输出: use_talker_cuda_graph: true

# 2. 跑
python -m nanovllm_omni.bench time \
    --pipeline full \
    --runs 20 --warmup 2 \
    --max-tokens 16 \
    --temperature 0.7 --top-p 0.9 --seed 42 \
    --use-thinker-cuda-graph \
    --out docs/perf/aligned/D-full-stack.csv
```

跟 Cell C 同 caveat：`Omni.generate` 目前不接 `--use-thinker-cuda-graph`，
所以 Cell D 在当前 bench 上**也会回退到 B/C 行为**。

## 3. 测量矩阵输出格式

`docs/perf/aligned/summary.md` 应该长这样：

| Cell | median (ms) | p95 (ms) | min (ms) | frames | vram (MiB) | speedup vs A |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | ? | ? | ? | 16 | ? | 1.00× |
| B fusion only | ? | ? | ? | 16 | ? | ?× |
| C fusion + thinker graph | ? | ? | ? | 16 | ? | ?× |
| D full stack | ? | ? | ? | 16 | ? | ?× |

`medium_01` / `short_02` 等 prompt 各自的 median / p95 由
`markdown_table(results)` 已经在 `bench time` 输出时打印，照搬到
summary.md 即可。

## 4. 已知 gap 与后续工作

按重要性排序：

1. **`Omni.generate` 不接收 `--use-thinker-cuda-graph`**——bench `_kwargs`
   里 `kwargs.pop("use_thinker_cuda_graph", None)` 把这个 flag 在
   `--pipeline full` 下丢了。Cell C / D 当前测不出来。
   **修复路径**：`Omni.generate` / `EngineArgs` 增加
   `use_thinker_cuda_graph` 字段，`Omni` 路由时把它传给 thinker stage。
2. **bench 不 expose `--use-talker-cuda-graph`**——Cell D 当前行为由
   deploy YAML 隐式决定，不显式。镜像 thinker 那条加个 flag。
3. **没有 torch 版本 / commit hash 自动记录到 CSV**——要 apples-to-apples，
   CSV 里得带这两栏，否则重测时不知道对齐的是什么。
4. **没有 GPU 温度 / 时钟记录**——RTX 3050 在冷态 60°C / 1725 MHz 和
   热态 80°C / 1590 MHz 相差 2×。`docs/perf/minimind-omni-under-500ms.md`
   §4.2 警告过。要在 bench 启动时 dump 一次 nvidia-smi。

## 5. 验收

完成后：

- `docs/perf/aligned/A-baseline-enforce-eager.csv` 等 4 份 CSV 落盘
- `docs/perf/aligned/summary.md` 4-cell 表填齐
- 每份 CSV 包含 `framework_torch_version` / `framework_commit` 两栏
  （**#3 的修复**）
- A cell 与 B cell 数字差距 = 4 个 fusion patch 的总收益
- C cell 与 B cell 数字差距 = thinker CUDA Graph 的 E2E 收益
  （前提：**#1 修了**）
- D cell 与 C cell 数字差距 = talker CUDA Graph 的收益
  （前提：**#2 修了**）

A→B→C→D 是单调下降的就对。如果某个 cell 比前一个还高，
说明某条 monkey-patch 实际在 E2E 上是 noise / regression，
按 `docs/perf/minimind-omni-under-500ms.md` §4 的纪律去 discard。
