# MiniMind-O thinker：MLX 轨接上 fork Scheduler 之后的首个数字（2026-11-10）

回答一个问题：MLX 轨的 thinker 跑上 fork `Scheduler` 之后，批的形状对它有多大影响。

结论是**批的形状影响很大**：单序列 26.83 ms，4 条不等长的请求 82.25 ms（3.1×）。
一个请求提前结束、其余继续的形状是 50.40 ms，落在两者之间——退出的那一行
确实从批里回收了，剩下三行继续算。

**这些是 MLX 轨的单机数字，不是与 CUDA 轨的对比**（ADR 0014：不跨机相减）。
本机没有 CUDA 基线，ADR 0021 的「token 级一致」尚未在两轨之间验证。

## 测量条件

| 项 | 值 |
|---|---|
| 硬件 | MacBook Air M2（Apple M2，10 核 GPU，16 GB 统一内存），macOS 27.0.1 |
| 代码版本 | `8d8ed46`（feat(mlx): bridge rows 改用整条序列重跑） |
| 运行时 | mlx 0.32.4，mlx-lm 0.32.2，Python 3.12 |
| 模型 | `jingyaogong/minimind-3o` 的 thinker，转换为 `~/models/minimind-3o/mlx-thinker`；加载为 fp16 |
| 输入 | `bench/prompts.py` 的 prompt，截到 48 token |
| 采样 | `temperature=1e-5`（fork 的 SamplingParams 禁止 0，CUDA thinker 也这样钳）、`top_p=1.0`、`repetition_penalty=1.05`、`seed=42` |
| runs / warmup | 9 / 3 |
| 计时口径 | `time.perf_counter()`，包住整个 `decode_minimind_mlx` |

命令：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
    uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/bench_mlx_scheduler.py --runs 9 --warmup 3
```

## 三个形状

| 形状 | 批行数 | prompt 长度 | 生成 | wall p50 | min / max |
|---|---|---|---|---:|---|
| single | 1 | 28 | 16 token | **26.83 ms** | 22.25 / 31.15 |
| ragged | 4 | 2+28+5+36 | 各 16 token | **82.25 ms** | 68.10 / 86.13 |
| uneven | 4 → 3 | 2+28+5+36 | 第一个 4 token，其余 16 | **50.40 ms** | 46.04 / 61.95 |

### 读法

- **ragged / single = 3.1×**。4 条请求的 wall 是单条的 3.1 倍，不是 4 倍。
  补齐让每步都按最长那条（36 token）算，所以每步比单条贵；但 4 条共用
  一次 forward，摊下来每条比单独跑便宜。
- **uneven 比 ragged 快 39%**。第一个请求只生成 4 个 token 就退出，
  剩下的 3 条从第 5 步起每步少算一行。`filter_cache_rows` 被调用了两次：
  第一次 `(4 → 3)` 删掉退出的那一行，第二次 `(3 → 0)` 在 decode 结束时清空。

## 这组数字测到了什么，没测到什么

**测到了**：批的行数怎么影响 wall；行回收这条路径确实在跑
（`filter_cache_rows` 的调用被逐次记录核对过）。

**没测到**：

- **没有真正的中途 join**。fork `Scheduler.schedule()` 一次就把 waiting 队列
  清空，所以一批请求是一起 prefill 的。`BatchKVCache.extend`——把新来的
  请求追加到正在跑的 cache 上——**目前没有调用方**，只有单测覆盖。
  「请求在 decode 中途到达」这个场景，MLX 轨现在做不到。
- **没有 varlen**。批内长度不齐仍然补齐到最长。ADR 0025 没有动这一点。
- **没有与 CUDA 轨相减**（ADR 0014）。上面每一行都是 MLX 轨自己的数字。

## 附带测到的一条硬约束

`mx.fast.rope` 的 `offset` 长度必须等于本步输入的行数：

```
cache 有 2 行, 喂 1 行 -> FAIL [rope] offset must be a scalar or vector with 1 elements but has shape (2)
cache 有 2 行, 喂 2 行 -> OK
cache 有 2 行, 喂 3 行 -> FAIL [rope] offset must be a scalar or vector with 3 elements but has shape (2)
```

这是 `BatchKVCache` 的直接后果：它的 `offset` 是一个长度等于批次的数组。
所以退役序列删 cache 行时，必须和 `schedule()` 返回的那组 `seqs` 同步。
这条约束写进了 ADR 0025。

（ADR 0023 记过「`mx.fast.rope` 的 `offset` 最多一维」，那说的是维度上限；
这里是长度上限，是另一回事。）
