# ADR 0023: MLX 轨的执行顺序——thinker-only 起步，kernel 先 decode

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0021（kernel 自研最小集、
  token 级一致）、0022（CUDA 轨这一轮）

## 决定

### 第一站：MiniMind-O 的 thinker（不含 talker）

用 `mlx_lm` 转出来的 MiniMind-O thinker 跑通 MLX backend 的**完整 AR 侧**：
varlen attention、批 decode、slot 复用。

**为什么不是 SmolVLM**：

- MiniMind-O 的 thinker 是 `Qwen3ForCausalLM` 契约（`thinker.py:247` 自述），
  `mlx_lm` 直接能转权重。
- SmolVLM 的视觉塔是 SigLIP（第三方库，ADR 0020 定的边界），要走 MLX 版
  得先移植它——为一个"验证 AR 侧"的目标多付一份移植。

**talker 断链不影响这个第一站**：验证 varlen kernel + backend + 批处理只需要 AR 侧完整。
talker 的 MLX 版是后面的事，它的前置是 ADR 0017 的重写（重写之后才谈得上写 MLX 版）。

### kernel 顺序：先 decode，再 prefill

- **decode** 最简单：每个序列 1 个 query，k/v 按 block table 从 KV 里读，
  按各序列的 context_len 屏蔽。没有扁平拼接、没有 `cu_seqlens`。
- **prefill** 才是 varlen 的主要收益所在（齐长批 vs ragged 批差 143%，
  ADR 0015 引的上游实测），但要处理扁平拼接 + 边界划界。

先做 decode 是为了让 kernel 有一条能验证的最短路径：decode 的正确性可以直接对着
CUDA 轨的 `audio_codes` / token 序列比（ADR 0021 的 token 级一致）。

## 后果

- **第一站的验收**：MLX 上 MiniMind-O thinker 能出与 CUDA 轨**同 token 序列**的结果
  （相同权重、seed、采样配置），并能跑一个多序列的批。
- **与 CUDA 轨的关系是趋势可比，绝对值不可比**（ADR 0014：不跨机相减）。
  比较的是"齐长批 vs ragged 批的相对变化"，不是"谁更快"。
- **第一站不覆盖** talker、code2wav、视觉、diffusers、lerobot。MLX 轨的覆盖面
  天然小于 CUDA 轨，这是 ADR 0014 已经承认的边界。
- **采样配置必须严格对齐**才能拿到 token 级一致（ADR 0021）：
  temperature / top-k / top-p / seed / max_tokens 一个都不能差。

## 实施记录（2026-11-09）

### 已落地

- **权重**：`jingyaogong/minimind-3o` 下载到 `~/models/minimind-3o`（不进 git）。
  `scripts/convert_minimind_thinker_mlx.py` 把 `model.*` + 可选 `lm_head.weight`
  转成 `mlx_lm` 的 qwen3 目录 `~/models/minimind-3o/mlx-thinker`，丢掉
  `talker.*`。转换不需要 mlx，只用项目 venv 的 torch + safetensors。
  同一 checkpoint 转两次的 `model.safetensors` sha256 相同。
- **模型复用**：`mlx_lm.models.qwen3` 的布局与 checkpoint 一致（separate
  q/k/v/o、per-head `q_norm`/`k_norm`、swiglu），不需要重映射 key。配置取
  `intermediate_size=2432`、`head_dim=96`、`rope_theta=1e6`、8 层。
  `lm_head` 与 `embed_tokens` 逐位相同，故置 `tie_word_embeddings=true` 并
  不写 `lm_head.weight`。
- **backend**：`MlxStageRunner` 走完整 AR 循环（prefill → 采样 → 逐 token
  decode）。单序列用 `KVCache`；多序列左补齐后用 `BatchKVCache`（逐行 RoPE
  offset + pad 列 mask）。`capabilities.max_batch` 从 1 改为 `None`。
  `bridge_hidden()` 返回 `None`——milestone 1 没有 talker，`bridge_layer`
  的位置留在模型侧，未接线。
- **采样**：对齐 CUDA 的
  `models/minimind_omni/stage_runner.py::_sampler_with_rp`：temperature 下限
  1e-5、full-history repetition penalty（默认 1.05）、top-p、每序列独立
  seed、`max_tokens`。`top_k` 不生效：vendor 文本路径本来就不用它，只有音频
  codebook 用。
- **dense batch**：`generate_mlx_batch` 是静态批，补齐到最长成员，整批跑到
  最慢成员结束。这是 ADR 0015 拒绝的过渡形态，只用于在 varlen kernel 之前
  拿到 MLX 轨的批数字。

### 实测（单机，不与 CUDA 相减）

硬件为 MacBook Air M2（16 GB，11.84 GiB 工作集），mlx 0.32.4 / mlx-lm 0.32.2，
代码 `698856c`，fp16，`temperature=0`（钳 1e-5）、`top_p=1`、rp 1.05、
`max_tokens=32`、seed 42，warmup 2 / runs 5：

| 形态 | TTFT p50 | decode p50 | 峰值内存 |
|---|---:|---:|---:|
| 单序列，45 token prompt | 6.06 ms | 4.33 ms/step | 145.5 MiB |
| 4 序列 dense batch，补齐到 53 | 20.77 ms | 6.63 ms/step | 214.5 MiB |

完整条件、输出 token 与限制见
`docs/perf/minimind-thinker-mlx-2026-11-09.md`。

### 验证

- 同一组 token `[1, 80, 301, 108, 80, 916]`，MLX（bf16）与 vendor
  `MiniMindOmni`（CPU fp32）的 argmax/top-5 相同。
- dense batch 与单序列在相同采样配置下逐 token 相同（2 条 prompt × 8 token）。
  这个检查抓到并修掉了一个批路径 bug：prefill 后没把首个采样 token 写回
  `Sequence.last_token`，导致后续输出整体错一位。
- top-p 分支实际执行过（`--top-p 0.9 --temperature 0.7`）。
- `pytest tests/`：8 failed / 286 passed / 6 skipped。8 个失败与改动前一致，
  全部是缺 `diffusers` / `lerobot` / `triton` 的既有失败。

### 未完成

- **没有 varlen / paged KV / Metal kernel**。ADR 0021 的 kernel 路线只做到
  接口设计：`docs/dev/mlx-varlen-kernel-interface.md`。Metal Toolchain 已装好，
  `mx.fast.metal_kernel` 可用，但 kernel 源码与接线未写。
- **ADR 0021 的 token 级一致未验证**：本机没有 CUDA 基线。另外两边 RNG 流
  不同（`torch.multinomial` vs `mx.random.categorical`），相同 seed 只对齐
  参数，不对齐随机流；真要对 token，需要先统一采样器。
- **bridge_layer / talker / code2wav 未接**。thinker-only 不需要 bridge 行。

### `mx.compile` 的实测（2026-11-09 补）

上一版写"`mx.compile` 未测"。测了，结论是**有微小收益，但依赖一个非显然前提**：

| 形态 | eager p50 | compiled p50 | trace 次数 |
|---|---:|---:|---:|
| 单序列 decode | 2.64 ms | 2.39 ms | 1 |
| 2 行 batch decode | 4.00 ms | 3.70 ms | 1 |
| 固定 offset（不递增） | 2.29 ms | 2.32 ms | 1 |

真实 32 步 AR 循环，fp16，同一 prompt 45 token，warmup 2 轮。**收益约 7–10%**，
与 ADR 0007 记的 CUDA Graph 3.8× 不是一个量级。

三个前提，缺一个就跑不起来或更慢：

1. **编译函数不能收 `KVCache` 对象**——必须把 cache 摊平成 `(keys, values, offset)`
   显式数组（摊平后数值逐位相同）。
2. **`shapeless=True` 在这个模型上直接报错**（`Slice cannot infer output shapes`）。
3. **致命的一条：`offset` 必须是 `mx.array`**。传 Python `int` 会让每一步都重新
   trace——32 步 decode → `traces=32`，每个新 offset 首次调用 35–45 ms，
   **比 eager 还慢**。`BatchKVCache` 的 offset 本来就是 array。

所以 `BackendCapabilities.graph = True` 的含义是"**这个后端有 graph 机制**"，
不是"当前路径就在用它"。MLX 侧目前的数字**全是 eager**。

MLX 的 `mx.fast.rope` 存在（`offset` 接受 int 或 array），ADR 0021 说的
"优先用自带 RoPE"成立。但 `KVCache` 用 offset 做切片（`self.keys[..., prev:self.offset, :]`），
传 array 会 `ValueError: Slice indices must be integers or None`
→ **paged 路径不能用 `KVCache` 承载 offset**，dense 对照臂要用 `BatchKVCache`。

## 边界

- 本 ADR **不**承诺 MLX 轨能跑完整 omni 管线。MiniMind-O 的完整管线需要 talker，
  而 talker 的 MLX 版尚未存在。
- 本 ADR **不**决定 kernel 源码放哪个目录、怎么编译加载——实现细节。
- 本 ADR **不**改 ADR 0019 的分工：大块归 MLX 轨，本 ADR 定的正是 MLX 轨的内部顺序。
