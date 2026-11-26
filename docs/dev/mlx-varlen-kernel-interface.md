# MLX varlen / paged attention 的 kernel 接口设计

这是 ADR 0021 与 ADR 0023 的实施设计：MLX 轨要自研的最小 Metal kernel 集，
以及它如何接到 `MlxStageRunner` 的 `prepare_*` / `forward` 上。

**状态：接口设计，未实现。** 本文不声称任何性能或正确性结果。
Metal Toolchain 已在本机验证可用（`xcrun metal --version` =
Apple metal 32023.921），`mx.fast.metal_kernel` 也已用一个小 kernel 跑通；
缺的是 kernel 源码与接线。

## 目标与非目标

目标（按 ADR 0023 的顺序）：

1. **paged decode attention**：每序列 1 个 query，KV 按 block table 读取，
   按 `context_lens` 屏蔽。
2. **KV 写入**：把新算出的 k/v 写进 block table 指定的 slot。
3. **paged prefill attention**：q 是本步所有 token 的扁平序列，用
   `cu_seqlens` 划界，块内因果屏蔽。

非目标：MLA、GDN、bidi、Mamba2、KDA、turboquant、NAX（ADR 0021 已排除）。
也不做 split-K、CUDA Graph 等价物、多 stream 调度。

## 张量布局（对齐 fork）

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q`（decode） | `[B, H, D]` | fp16 / bf16 | 每序列 1 个 query |
| `q`（prefill） | `[T, H, D]` | 同上 | 本步所有 token 扁平拼接 |
| `k_cache` | `[num_blocks, block_size, H_kv, D]` | 同上 | 与 fork `k_cache` 一致 |
| `v_cache` | `[num_blocks, block_size, H_kv, D]` | 同上 | |
| `block_table` | `[B, max_blocks_per_seq]` | int32 | `-1` 为无效块 |
| `context_lens` | `[B]` | int32 | 每条序列已有的 KV 长度 |
| `cu_seqlens_q` | `[B + 1]` | int32 | prefill 的扁平 token 划界 |
| `slot_mapping` | `[T]` | int32 | 每个新 token 的 KV 写入位置 |
| 输出 | decode `[B, H, D]`，prefill `[T, H, D]` | 输入 dtype | |

`block_size` = 256（fork `Sequence.block_size`）。`D` = 96（MiniMind-O），
`H` = 8，`H_kv` = 4，GQA 组宽 2。kernel 用 template 参数接收 `D` / `H_kv`
比例，不把 96 写死；`D=96` 是启动配置里的默认值。

## 三个 kernel 的接口

```python
def paged_decode_attention(
    q: mx.array,             # [B, H, D]
    k_cache: mx.array,       # [num_blocks, block_size, H_kv, D]
    v_cache: mx.array,       # [num_blocks, block_size, H_kv, D]
    block_table: mx.array,   # [B, max_blocks_per_seq] int32
    context_lens: mx.array,  # [B] int32
    scale: float,            # D ** -0.5
) -> mx.array:               # [B, H, D]
    ...

def paged_prefill_attention(
    q: mx.array,             # [T, H, D]
    k_cache: mx.array,
    v_cache: mx.array,
    block_table: mx.array,   # [B, max_blocks_per_seq] int32
    cu_seqlens_q: mx.array,  # [B + 1] int32
    context_lens: mx.array,  # [B] int32，本步结束后每序列的 KV 长度
    scale: float,
) -> mx.array:               # [T, H, D]
    ...

def store_kv(
    k: mx.array,             # [T, H_kv, D]
    v: mx.array,             # [T, H_kv, D]
    k_cache: mx.array,
    v_cache: mx.array,
    slot_mapping: mx.array,  # [T] int32
) -> None:
    ...
```

`store_kv` 与 fork 的 `layers/attention.py::store_kvcache` 同语义：一个 token
一个 slot，slot 由 `block_table[i] * block_size + offset` 算出，`-1` 跳过。

### decode 的线程映射

- `grid = (B, H, 1)`，每个 threadgroup 负责一条序列的一个 q head。
- GQA：q head `h` 对应 kv head `h // (H // H_kv)`。
- threadgroup 内沿 context 分块（初始方案 128 token/块），每块做
  `q·k → online softmax → acc += p·v`，全用 fp32 累加。
- 取 key：`phys_block = block_table[b, pos // block_size]`，
  `k = k_cache[phys_block, pos % block_size, kv_head, :]`。
  `pos < context_lens[b]` 才参与；其余用 `-inf` 屏蔽。

### prefill 的线程映射

- `grid = (T, H, 1)`，每个 query token 一个 threadgroup。
- 第 `i` 个扁平 token 属于序列 `b`（由 `cu_seqlens_q` 二分/查表得到），
  它的绝对位置 `pos = context_start[b] + i - cu_seqlens_q[b]`。
- 只读 `[0, pos]` 的 key（块内因果），块间跨界时用
  `pos // block_size` 与 `pos % block_size` 拆两次读取。
- 输出写回 `[T, H, D]`，由调用方 reshape 成 `[B, T_b, H, D]` 的批次视图。

## 与 `MlxStageRunner` 的接线

### 上下文对象

新增一个 `PagedContext`（runner 私有，不进 `backend/base.py` 协议）：

```python
@dataclass
class PagedContext:
    k_cache: mx.array
    v_cache: mx.array
    block_table: mx.array
    context_lens: mx.array
    slot_mapping: mx.array
    cu_seqlens_q: mx.array | None
```

`set_context(is_prefill, **kwargs)` 把当前步的 `block_table` /
`slot_mapping` / `context_lens` / `cu_seqlens_q` 装进 runner；`reset_context`
清空。协议签名不变。

### `prepare_prefill`

现在返回 `(input_ids, positions)`；paged 路径需要额外产出
`cu_seqlens_q`、`slot_mapping`、`block_table`。三种落法：

1. 扩 `prepare_prefill` 返回值 —— **不改协议的前提下不行**，stage 侧会解包失败。
2. 把额外张量放到 `PagedContext`，`forward` 从 runner 读 —— 可行，协议不变。
3. 把 `block_table` 等挂到 `Sequence` 上 —— 那是 fork 的对象，不该塞后端字段。

选 2。`prepare_prefill` 仍返回两个值，但把 paged 元数据写进 runner 的
`PagedContext`；`forward` 看到 `self._paged is not None` 时走 kernel 路径。

### 模型接线

`mlx_lm.models.qwen3.Attention.__call__` 里 attention 是硬编码的
`scaled_dot_product_attention(queries, keys, values, cache=cache, ...)`。
paged 路径需要一个同权重的 `PagedAttention` 模块：

- 复用 qwen3 的 `q_proj` / `k_proj` / `v_proj` / `o_proj` / `q_norm` / `k_norm`，
  只替换 `__call__` 里的 KV 写入与 attention 调用；
- 权重从同一份 safetensors 读入，不重新转换；
- `MlxStageRunner` 在构造时用 `qwen3.ModelArgs` 建 `qwen3.Model`，再把每个
  `layer.self_attn` 换成 `PagedAttention`（共享参数数组，不复制）。

这保持了 ADR 0021 的边界：kernel 只做 attention / KV 搬运，模型其余部分
（embedding、MLP、norm、RoPE、lm_head）继续用 mlx_lm 的实现。

### RoPE 位置

paged 路径不能用 `cache.offset`（没有 `KVCache` 对象）。`prepare_*` 已经
按序列算好 `positions`，传给 `mx.fast.rope(..., offset=positions)` 即可。
decode 的位置是 `context_lens - 1`；prefill 的位置是
`context_start + arange(num_scheduled_tokens)`。这与 dense 路径的
`BatchKVCache` 逐行 offset 语义一致。

## 验收顺序

1. **kernel 单元级**：随机 q/k/v，与
   `mx.fast.scaled_dot_product_attention` 的 dense 结果比，fp32 容差
   `1e-4`（阈值在实现时按实际 dtype 定，写进测试）。
2. **dense 对照**：同一 prompt、同一采样配置，paged decode + dense
   `BatchKVCache` 路径 token 完全相同。这是本机能做的最强检查。
3. **跨机**：与 CUDA 轨比 token 序列（ADR 0021），需要 WSL 基线；
   采样器必须先对齐随机流，不能只对齐 seed。

## 未决 / 实现时再定

- threadgroup 大小与 context 分块长度：先按 128 token/块实现，再测。
- prefill 的 `cu_seqlens` 查表方式：每 token 线性扫 vs 预计算 `seq_ids`；
  先选后者（多一个 `[T]` int32 输入，换来 kernel 里无分支）。
- fp16 还是 bf16 做 KV cache：先跟输入 dtype 一致，累积用 fp32。

## 实测推翻的三个假设（2026-11-09）

下面三条是写完接口后拿 MLX 0.32.4 实测出来的，与本文原来的写法冲突。
**按实测改，不按设想改。**

### 一、`store_kv` 必须和 attention 合进同一个被编译的图

原来写的是"先分开，便于对照 fork 的 `store_kvcache`"。**实测反过来**：

- MLX 的 `mx.fast.metal_kernel` **输出永远是新 buffer，不能原地写**。把 KV pool
  同时当 input 和 output 传进去，得到的是独立的新数组，原数组不被修改。
- **MLX 没有对"仍被引用的数组"做原地 scatter 的能力**：

  | 写法 | 12 MB pool | 结果 |
  |---|---:|---|
  | `pool[idx] = new` | 0.63 ms | 正确，但分配整池新 buffer |
  | `pool.at[idx].add(new)`（不接返回值） | 0.002 ms | **静默 no-op，数据没写进去** |
  | `mx.put_along_axis(pool, idx[:,None,None], new, 0)` | 0.50 ms | 正确 |
  | `pool * 1`（纯拷贝基线） | 0.47 ms | — |

  `mx.slice_update` 不能用：它要**每个轴一个起始位置**，不是 scatter 索引表。
- 带宽约 18–20 GB/s，**scatter 和 memcpy 一样贵 → 每次 KV 写入 = 整池拷贝**。
- 唯一测得有效的缓解：**把 store 放进 `mx.compile` 图里**，12 MB 池从 eager 2.40 ms
  降到 compiled 0.60 ms（编译图内的 donation 生效）。

→ **store 与 attention 必须在同一个被编译的图里**，否则拿不到 donation。

### 二、paged KV 的池大小有硬预算

| 池大小 | 单层写入边际成本 | ×8 层/step |
|---|---:|---:|
| 8 blocks（1.5 MB） | 0.06 ms | 0.45 ms |
| 64 blocks（12 MB） | 0.46 ms | **3.7 ms** |
| 256 blocks（48 MB） | 1.48 ms | **11.9 ms** |

对照：整个 decode step 现在只要 2.3–4.0 ms。**照"每层把新 k/v 写进 block_table
指定的 slot"的直白实现，只要池超过 ~64 个 block，光 KV 写入就比整个前向还贵，
paged KV 会变成纯倒退。**

→ 池大小必须先按上表定上限，再去实现；否则会测出一个负优化还以为是 kernel 的锅。

### 三、`mx.compile` 不是无条件的加速

- **编译函数不能收 `KVCache` 对象**：`ValueError: [compile] Function arguments must be
  trees of arrays or constants`。必须把 cache 摊平成 `(keys, values, offset)` 显式数组
  （摊平后数值逐位相同，diff 0.0）。
- **`shapeless=True` 在这个模型上直接报错**：`Slice cannot infer output shapes`。
- **致命的一条：`offset` 作为 Python `int` 传进去，每一步都重新 trace。** 32 步 decode
  → `traces=32`，每个新 offset 首次调用 35–45 ms。**换成 `mx.array` 后只 trace 1 次**
  （`BatchKVCache` 的 offset 本来就是 array）。

实测收益（真实 32 步 AR 循环，fp16，同一 prompt 45 token，warmup 2 轮）：

| 形态 | eager p50 | compiled p50 | trace 次数 |
|---|---:|---:|---:|
| 单序列 decode | 2.64 ms | 2.39 ms | 1 |
| 2 行 batch decode | 4.00 ms | 3.70 ms | 1 |
| 固定 offset（不递增） | 2.29 ms | 2.32 ms | 1 |

→ **收益约 7–10%，且依赖"offset 做成 array"这个非显然前提**；否则退化成每步重编译，
**比 eager 还慢**。这与 ADR 0013 记的 4.91×（512×512 小算子链）不是同一个量级。

## 已知风险

- Metal 没有 `cu_seqlens` 原语，prefill 的边界全靠 kernel 内查表；查错会
  安静地算到隔壁序列的 KV 上，必须用第 2 项验收兜住。
- `block_table` 里的 `-1` 表示无效块；decode 的 `pos < context_lens`
  屏蔽必须发生在读块之前，否则会读到越界内存。
- 96 不是 32 的整数倍（96 = 3×32），threadgroup 内的 head_dim 切分要显式
  处理尾部；不要假设 `D % 32 == 0` 之外的向量化。
- **`pool.at[idx].add(...)` 是静默 no-op**：写 scatter 时一定要接返回值，
  否则会得到"跑得通但数据没写进去"的实现。
- 本文的所有接口都还未被调用；`backend/base.py` 的协议没有改动。
