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

### 已知的损失（写清楚，不遮）

MLX 现在拿 bridge rows 的办法是**每轮 decode 结束后把整条序列重跑一遍**
（`stage_runner.py:340-352`、`:365-368` 的 docstring）。切到调度器之后：

- 一个请求从 prefill 到结束跨多个批次，重跑时它可能已不在当前批次里；
- 当前实现把多个请求的 rows `cat` 成 `[B, T, 768]`，各序列长度不同，cat 不出来。

本轮接受**只保留"本步仍在批次内"的请求的 bridge rows**，并把 `decode_minimind_mlx`
的 docstring 改成如实描述。要在贪心 parity 上把这件事做成严格等价，需要在每次
prefill 之后立刻重跑一次 bridge（`recapture_bridge_minimind` 就是这个思路），那是
下一轮的事——先让两条轨跑同一个调度器。

## 后果

- MLX 轨能出和 CUDA 轨同类的数字：动态 join、chunked prefill、preemption。
- `decode_minimind_mlx` 与 `decode_minimind` 的结构差距缩小到只剩 `forward` 与采样器。
- bridge rows 的完整性降级，docstring 必须同步改。
- `MlxConfig` 增加 KV 与调度两组字段（`kvcache_block_size` / `num_kvcache_blocks` /
  `max_num_seqs` / `max_num_batched_tokens` / `eos`），`MlxBackend.make_config` 透传，
  `MlxStageRunner.__init__` 从 tokenizer 填 `eos`。
- decode 阶段每个退役的序列（FINISHED 或 preempt）都要 `filter` 掉它那一行，否则下一轮
  的行数和 cache 行数对不上。

## 边界

- 本 ADR **不**做 varlen attention。补齐仍然存在，省不掉的成本还在。
- 本 ADR **不**改 SmolVLM 的 `MlxVlmBackend`（`max_batch=1`，单序列没有调度可言）。
- 本 ADR **不**保证 MLX 和 CUDA 的吞吐绝对值可比（ADR 0014：不跨机相减）。
