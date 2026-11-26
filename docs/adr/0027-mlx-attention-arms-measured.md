# ADR 0027: MLX attention 三臂实测——paged kernel 暂不接线

- Status: Accepted
- Date: 2026-11-11
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0015（MLX 要 varlen）、
  0021（kernel 自研最小集）、0023（MLX 轨顺序）、0026（增量 step）

## 问题

ADR 0015 要求 Mac 轨有 varlen，理由是上游 `vllm-metal` 的实测：padded 路径在
ragged 批上 +143%（35B 模型，6000 prompt token）。

那条数据的两个前提我们都不成立。我们的模型是 MiniMind-O 的 thinker
（8 层，hidden 768，H=8，H_kv=4，D=96），上下文在千 token 量级。

ADR 0023 的「未做」里挂着接线任务（`TASKS.md` 的 MX-1）。接线前要先回答：
**我们自己的 paged kernel，在我们的尺寸上，值不值那笔接线。**

## 方法

脚本 `tools/bench_mlx_attention.py`。三个臂，同一批数据，同一套张量：

| 臂 | 做法 |
|---|---|
| `dense` | 现在 MLX backend 的路径：left-padded batch + `BatchKVCache` + 一次带 mask 的 SDPA |
| `paged` | 我们的 Metal kernel：block table → KV pool，每序列 1 个 query，扫 `context_len` 个 key |
| `bucketed` | `dense`，但先按长度分桶，每桶 pad 到自己的最长成员 |

每臂测两件事：attention，与 decode 每步写 1 个 token 的 KV 写入。
两条臂的数值先对齐：dense 与 paged 的最大绝对差 **3.05e-05**（fp16 噪声）。

口径：

- 硬件：M2 Air（16 GB 统一内存），共享机器
- MLX 0.32.4，代码版本 `0db97b7`（+ 本次未提交改动）
- 形状：MiniMind-O thinker，H=8 / H_kv=4 / D=96，8 层
- 批大小 1 / 4 / 8 / 16；长度均值 1024 token；四种 spread
- warmup 4、runs 12、每格重复 3 次取**最好一次**（共享 Mac 上最稳）；测量下限 0.0007 ms
- 会话漂移：固定参考格（8 × 1024 的 dense attention）开工 0.487 ms、收工 0.505 ms，差 3.7%
- 原始产物：`docs/perf/mlx-attention-2026-11-11/`（`dense-vs-paged.csv`、
  `padding.csv`、`meta.json`）

## 数字

### padding 的代价：8 序列 × 256 real token，只加 pad

| padded tokens | waste | dense attn | 8 层一步 |
|---:|---:|---:|---:|
| 256 | 0.0% | 0.328 ms | 3.102 ms |
| 512 | 50.0% | 0.367 ms | 3.185 ms |
| 1024 | 75.0% | 0.377 ms | 3.356 ms |
| 2048 | 87.5% | 0.448 ms | 3.941 ms |
| 4096 | 93.8% | 0.726 ms | 6.311 ms |
| 8192 | 96.9% | 1.085 ms | 9.591 ms |
| 16384 | 98.4% | 1.889 ms | 15.110 ms |

**64 倍的 pad 只换来 5.8 倍的时间。** MLX 的 SDPA 对被 mask 掉的列很便宜。
ADR 0015 引的 +143%（35B、6000 prompt token）在这个尺寸上不成立。

### 三臂：每层 attention + KV 写入（ms）

| batch | pattern | waste | dense | bucketed | paged (pool=batch) | paged (pool=64) | dense ns/head-key | paged ns/head-key | 赢的臂 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | uniform | 0.0% | 0.695 | 0.645 | 1.388 | 2.438 | 41.194 | 161.997 | bucketed |
| 1 | spread2 | 0.0% | 0.523 | 0.507 | 1.080 | 1.718 | 45.926 | 128.045 | bucketed |
| 1 | spread4 | 0.0% | 0.498 | 0.525 | 0.913 | 1.454 | 61.656 | 142.466 | dense |
| 1 | bimodal | 0.0% | 0.549 | 0.491 | 3.498 | 4.093 | 10.240 | 97.680 | bucketed |
| 4 | uniform | 0.0% | 0.565 | 0.609 | 1.483 | 2.130 | 11.319 | 36.462 | dense |
| 4 | spread2 | 16.7% | 0.676 | 1.125 | 1.551 | 1.813 | 11.397 | 30.349 | dense |
| 4 | spread4 | 37.5% | 0.609 | 1.887 | 2.723 | 2.816 | 8.212 | 54.403 | dense |
| 4 | bimodal | 69.5% | 0.932 | 2.185 | 3.570 | 5.719 | 5.460 | 116.700 | dense |
| 8 | uniform | 0.0% | 0.673 | 0.619 | 1.429 | 1.758 | 7.148 | 13.622 | bucketed |
| 8 | spread2 | 20.0% | 0.648 | 1.113 | 1.716 | 2.120 | 5.819 | 17.522 | dense |
| 8 | spread4 | 46.9% | 1.095 | 2.277 | 1.897 | 2.174 | 4.222 | 19.641 | dense |
| 8 | bimodal | 80.1% | 0.844 | 1.824 | 2.967 | 3.289 | 2.526 | 47.720 | dense |
| 16 | uniform | 0.0% | 0.784 | 0.827 | 2.041 | 2.139 | 4.647 | 7.104 | dense |
| 16 | spread2 | 20.0% | 0.841 | 1.257 | 2.210 | 2.141 | 4.023 | 7.502 | dense |
| 16 | spread4 | 46.9% | 0.984 | 2.814 | 3.200 | 2.652 | 2.965 | 10.408 | dense |
| 16 | bimodal | 80.1% | 1.538 | 2.115 | 5.003 | 4.927 | 2.532 | 33.013 | dense |

三条读数：

1. **paged 每个 head-key 比 MLX 的 SDPA 贵 1.5–21 倍。** waste 越大差距越大，
   因为 dense 的被 mask 列几乎不要钱，而我们的 kernel 按真实 key 逐个扫。
2. **paged 的 KV 写入是固定成本。** 64 blocks 的池每步整池拷贝，0.80–1.48 ms。
   dense 的写入随 padded 大小走，在最极端的格子（batch 16 bimodal）反而比池拷贝便宜。
3. **16 个格子里 paged 一个都没赢。** dense 赢 12 格，bucketed 赢 4 格。
   这一条在三次独立会话里都成立，是本 ADR 最稳的读数。

## 决定

### 一、暂不接线

ADR 0023「未做」里的接线任务（MX-1）继续不做，直到 kernel 本身变快。
现在接进去，Mac 轨会变慢。

### 二、接线的门槛写成数字

**paged kernel 的每 head-key 成本要低于 MLX SDPA 的一半，才值得接。**
现在是它的 1.5–21 倍。要达到，需要 flash-decode 那种写法——把 context 切成多段、
多个 threadgroup 并行扫、再合并——不是调参能解决的。

### 三、修订 ADR 0015 的一句话

ADR 0015 写的是「Mac 轨不接受 dense padded 的降级方案」。实测之后改成：

> **Mac 轨不接受无上界的 padded。** 上界由长度分桶与 max context 给，
> 并用 ADR 0026 的增量 step 保住「请求可中途加入」这件事。

分桶不是免费的：本测量里 bucketed 在 waste 小的时候和 dense 打平，
在 batch 16 spread4 时反而比 dense 慢 2.9 倍（多个小 SDPA 调用）。
所以分桶要配一个「桶数上限」，超过就不分。

### 四、prefill 的 varlen kernel（MX-2）继续不做

理由同上：没有 decode 的证据说 kernel 划算，就不写 prefill 的 kernel。

## 边界与限制

- 这是 **attention 子系统的微基准**，不是端到端 decode step。绝对值是上界
  （没有图复用，没有与周边算子融合）。两条臂用同一套口径，相对结论可用。
- 只测了 MiniMind-O 的形状与千 token 量级的上下文。更大模型或更长上下文会改变
  pad 与 scan 的比例，结论不能外推。
- 只测了 decode。prefill 的两条路都没测（我们没有 prefill kernel）。
- 机器是共享的 M2 Air，单机，min 统计。没跑多台机器。绝对值在会话之间会漂
  （同一张表在三次会话里，dense 的最大格子在 1.33–2.05 ms 之间动）。可信的读数
  是比值与「paged 一格没赢」这一条，不是绝对毫秒。
- MLX 0.32.4。`mx.fast.scaled_dot_product_attention` 的 mask 语义按当前版本。

## 未验证

- 端到端差值：把 dense / paged 两条路接进 `MlxStageRunner` 后，真实 decode step
  差多少。本测量只覆盖 attention 子系统。
- 长上下文（>8k）下 dense 的 pad 曲线是否还这么平。表里 4096 之后斜率变陡，
  但只有 3 个点。
- `vllm-metal` 的 unified paged varlen kernel 比我们快多少。它是 C++/Metal 的大工程，
  我们没有移植。
