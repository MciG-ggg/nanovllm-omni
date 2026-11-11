# ADR 0025 — MLX 的 thinker decode 接 fork Scheduler，不再自己数步

- 状态：Accepted
- 日期：2026-11-10
- 相关：ADR 0013（backend 接缝）、ADR 0016（serving 是教材）、ADR 0023（MLX 轨顺序）

## 问题

`decode_minimind_mlx`（`models/minimind_omni/stage_runner.py:355`）拿到一组请求后，
把所有 prompt 左补齐成一个 dense batch，然后一路 decode 到最慢的那个结束。

它**不经过 fork 的 `Scheduler`**——那 130 行里 `scheduler` 出现 0 次。CUDA 轨的
`decode_minimind`（同文件 `:99`）是 `while not scheduler.is_finished()`，走的是
`schedule() → prepare → forward → postprocess`。

两个后果：

1. **双轨的"吞吐结论"不是同一个量。** ADR 0014 要求两条轨都出性能结论、比较的是
   趋势。CUDA 轨在考调度（动态 join、chunked prefill、preemption 归还 KV），MLX 轨
   在考一个 batched generate。前者能出"加一个请求让 p99 怎么变"，后者出不了。
2. **和 AGENTS.md 的定位直接抵触。** 那里写着 serving 是教材本身（ADR 0016）。一条
   不跑调度器的轨，缺的正是教材。

ADR 0023:70 承认过"dense batch 是过渡形态"，但那是在说 batch 形状，没说这条轨根本没
有调度。

## 决定

MLX 的 thinker decode 改成和 CUDA 轨同构：`Scheduler` 同一个、`schedule()` 同一个，
两个轨只在 `forward` 的实现上分叉。

这不需要新的抽象层——`MlxStageRunner` 已经有 `prepare_prefill / prepare_decode /
prepare_sample / forward / sample`，恰好是 `schedule()` 要的全部。

### 五个必须处理的落差（均由实测确认）

**一、fork 的 block manager 在 MLX 上是纯记账。**

`schedule()` 的行为依赖 `BlockManager` 的 `can_allocate / can_append / deallocate`。
MLX 只有一份 dense KV，没有 block table。但 fork 的 `BlockManager` 142 行里只碰
`block_table` 和 `num_cached_tokens` 两个字段，不碰张量——所以它在 MLX 上**原样可用**，
不需要改写。它提供的语义（容量上限、preemption）正是要教的东西。

要让 preemption 有意义，`num_kvcache_blocks` 必须配成有限值而不是"足够大"。
`MlxConfig` 目前没有这个字段，要补 `kvcache_block_size` / `num_kvcache_blocks`。

**二、cache 的行必须每步和批次对齐，否则静默错或直接炸。**

实测（M2 Air，mlx 0.32.4 / mlx-lm 0.32.2），一个 3 行的 cache：

```
喂 1 行 -> OK，但只写到第 0 行，其余两行没更新   （静默错误）
喂 2 行 -> ValueError: [broadcast_shapes] Shapes (2,2,1,4) and (3,2,1,4) cannot be broadcast
```

所以「序列从 `scheduler.running` 里退役」和「从 `BatchKVCache` 删行」**必须同步**。
两条退役路径都要删行：

1. `postprocess` 把序列标成 `FINISHED`（EOS 或到 `max_tokens`）；
2. `preempt` 把它送回 waiting（`block_manager.deallocate` 已把 `num_cached_tokens` 归零，
   它的 KV 已经没有意义）。

删行用 `BatchKVCache.filter(batch_indices)`，实测可用：3 行筛成 2 行，`_idx` 保持，
继续 decode 正常，`keys_and_values()` 返回 `(2,2,6,4)`。

`filter`（`cache.py:981`）会顺带做一件事：算 `min_left_pad = min(left_padding)`，
把所有行共有的那段左填充永久切掉（`_idx` 同步减）。这在本轮是安全的，因为
**每轮 prefill 都重建 cache**，`filter` 只在 decode 阶段的行回收时用。不要在
「想复用 cache」的路径上调用它。

**三、新请求 join 用 `extend`，它自己会右对齐补齐。**

这是 MLX 上动态 join 的机制。实测：

```
old (2 行, _idx 5, lp [0,0])
new (1 行, _idx 3, lp [0])          # 一个 3-token 的新请求
extended -> keys (3,2,256,4)
            _idx     5              # max(5,3)
            offset   [5, 5, 3]      # 各行自己的 RoPE 起点
            lp       [0, 0, 2]      # 新行被补 2 列
```

`extend`（`cache.py:1012`）对每行算 `left = max_idx - c._idx` 再 `mx.pad`。所以新请求
单独 prefill 成只含自己的 `BatchKVCache`，再 `extend` 进正在跑的那个即可，不用手工补。

这条路径现在有调用方：`ThinkerDecodeSession` 在 decode 中途 `add()` 新请求，下一轮
`schedule()` 先 prefill 它，`MlxStageRunner.forward` 检测到 `is_prefill` 且已有
cache 时调用 `extend`。见 ADR 0026。

**四、`Scheduler` 的构造契约比 `MlxConfig` 现在给的多。**

`Scheduler.__init__` 读 `config.max_num_seqs`、`config.max_num_batched_tokens`、
`config.eos`。`MlxConfig` 现在这三个都没有；`MlxStageRunner` 也从没设过 `eos`
（它拿的是 `tokenizer`，不是 `config`）。所以本轮要一并补齐：
`MlxConfig` 加 `max_num_seqs` / `max_num_batched_tokens` / `eos`，并在
`MlxStageRunner.__init__` 里从 tokenizer 填 `config.eos`。

**五、`schedule()` 结尾有 `assert scheduled_seqs`。**

循环要防「还有 running 但全部 `is_finished`」的中间态：

```python
while not scheduler.is_finished():
    if not scheduler.waiting and not scheduler.running:
        break
    seqs, is_prefill = scheduler.schedule()
```

每轮先把手上的 `is_finished` 序列从 `scheduler.running` 摘掉、同时 `filter` 掉对应行。

### 一条硬约束：decode 的行数必须等于 cache 的行数

`mx.fast.rope` 的 `offset` 长度必须等于本步输入的行数。cache 有几行、
这次喂几行，两者必须一致，否则报错：

```
cache 有 2 行, 喂 1 行 -> FAIL [rope] offset must be a scalar or vector with 1 elements but has shape (2)
cache 有 2 行, 喂 2 行 -> OK
cache 有 2 行, 喂 3 行 -> FAIL [rope] offset must be a scalar or vector with 3 elements but has shape (2)
```

这条约束是 `BatchKVCache` 的直接后果：它的 `offset` 是一个长度等于批次的数组。

推论：**退役序列删 cache 行时，必须在同一步把那个序列从 `schedule()` 返回的
`seqs` 里去掉**。`decode_minimind_mlx` 的 `_filter_inactive_cache_rows` 在
`schedule()` 之后、`prepare_*` 之前调用，两边看到的是同一批 `seqs`，所以行数一致。

（ADR 0023 记过「`mx.fast.rope` 的 `offset` 最多一维」，那条说的是维度上限；
这里是长度上限，是另一回事。）

### bridge rows：重跑整条序列

MLX 的 bridge rows 不能靠 decode 期间累积。原因是算过的位置和 bridge row 不是
一一对应：decode 循环每步 forward 一个 token，但最后一���采样出来的 token 从来没被
forward，所以它没有 row。

CUDA 轨早就有这个问题的解法（`recapture_bridge_minimind`）：decode 结束后，对每条
请求**整条序列重跑一次 prefill**，取那一层的 hidden。MLX 侧照做。

`MlxStageRunner.recapture_bridge`（`backend/mlx_backend.py`）做这件事，关键是它用
**一份全新的 cache**，不碰 `self._cache`——正在跑的 decode 状态不受影响。

实测（M2 Air，8 层 checkpoint，prompt 22 token + 生成 16 token）：

| 路径 | 行数 | 与 vendor 的最大差 |
|---|---:|---:|
| decode 累积 | 37 | 形状就不对（少一行） |
| **recapture** | **38** | **0.25**（量级 10128，相对 2.5e-5） |

0.25 是 fp16 的精度极限，不是逻辑误差。

## 后果

- MLX 轨能出和 CUDA 轨同类的数字：动态 join、chunked prefill、preemption。
- `decode_minimind_mlx` 与 `decode_minimind` 的结构差距缩小到只剩 `forward` 与采样器。
- bridge rows 由整条序列重跑得到，与 CUDA 轨和 vendor 都对齐；
  重跑的代价是每条请求一次 prefill，ADR 不承诺这一项的吞吐数字。
- `MlxConfig` 增加 KV 与调度两组字段（`kvcache_block_size` / `num_kvcache_blocks` /
  `max_num_seqs` / `max_num_batched_tokens` / `eos`），`MlxBackend.make_config` 透传，
  `MlxStageRunner.__init__` 从 tokenizer 填 `eos`。
- decode 阶段每个退役的序列（FINISHED 或 preempt）都要 `filter` 掉它那一行，否则下一轮
  的行数和 cache 行数对不上。
- `MlxStageRunner` 多一个 `recapture_bridge` 方法：整条序列重跑一次 prefill，
  用独立的 cache，不影响正在跑的 decode。

## 实测数字（2026-11-10）

M2 Air / mlx 0.32.4 / 8 层 checkpoint，`decode_minimind_mlx` 走完整入口，
16 token，9 runs / 3 warmup：

| 形状 | 批行数 | prompt 长度 | wall p50 |
|---|---|---|---:|
| single | 1 | 28 | 26.83 ms |
| ragged | 4 | 2+28+5+36 | 82.25 ms（3.1× single） |
| uneven | 4 → 3 | 2+28+5+36 | 50.40 ms |

`uneven` 的行回收逐次核对过：`filter_cache_rows` 被调用两次，
(4 → 3) 删掉退出的那一行，(3 → 0) 在 decode 结束时清空。

**这些是 MLX 轨的单机数字，不与 CUDA 轨相减**（ADR 0014）。

完整记录在 `docs/perf/minimind-thinker-mlx-scheduler-2026-11-10.md`
（`docs/perf/` 不进 git，按仓库约定）。

### 中途 join（2026-11-11 已接上）

fork `Scheduler.schedule()` 在 decode 阶段每轮只取仍在 `running` 的序列；新 `add()`
的序列留在 `waiting`，下一轮先被 prefill（prefill 优先），随后 `extend` 进运行中的
cache，再下一轮与原来的序列同批 decode。`ThinkerDecodeSession.step()` 就是这个粒度。
ADR 0014 要的「加一个请求让 p99 怎么变」现在能测；本轮测到的方向是：CUDA 明显
变慢，MLX 比「等旧批排空」好 9–23%，但整段 wall 与串行之和的差距在波动范围内。
数字见 ADR 0026。默认部署仍是 `max_num_seqs: 1`。

## 边界

- 本 ADR **不**做 varlen attention。补齐仍然存在，省不掉的成本还在。
- 本 ADR **不**改 SmolVLM 的 `MlxVlmBackend`（`max_batch=1`，单序列没有调度可言）。
- 本 ADR **不**保证 MLX 和 CUDA 的吞吐绝对值可比（ADR 0014：不跨机相减）。
