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
  `bridge_hidden()` 返回第 `bridge_layer` 层之后、final norm 之前的行
  （recorder 包住那一层；见下文「thinker stage 在 MLX 上跑通」）。
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

- **MlxStageRunner 尚未接入 varlen / paged KV attention**。Metal paged decode
  attention 与 KV store kernel 已实现于 `nanovllm_omni/backend/mlx_kernel/`，
  `tests/test_mlx_kernel.py` 有数值闸门；kernel 尚未接进 `MlxStageRunner`，详见
  `backend/mlx_kernel/kernels.py` 的模块说明。
- **ADR 0021 的 token 级一致未验证**：本机没有 CUDA 基线。另外两边 RNG 流
  不同（`torch.multinomial` vs `mx.random.categorical`），相同 seed 只对齐
  参数，不对齐随机流；真要对 token，需要先统一采样器。
- **code2wav 不是 MLX 实现**。thinker 与 talker 已在 MLX 上串通（见下文）；
  Code2Wav 仍由 torch Mimi 实现，Mac 上走 MPS（ADR 0024）。

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

### Metal paged decode kernel 的实施结果（2026-11-09）

kernel 本体可用。数值对照（3 条不等长序列 `context_lens=[600, 257, 300]`，
block table 非连续且带 `-1` 洞）：

| 对照 | max abs diff | 相对差 |
|---|---:|---:|
| paged decode vs `BatchKVCache` dense + SDPA（fp16） | 6.10e-05（ref 幅值 0.415） | 1.5e-04 |
| paged decode vs numpy fp32 独立参考 | 1.13e-04 | 1.86e-03 |

`store_kv` 的 `-1` 处理：有效的两个 slot 正确落位，`-1` 那一行的值在整池里出现
**0 次**，池里其他行原样保留，**传入的原池数组未被修改**（再次确认 kernel 不能原地写）。

**池大小上限定为 64 blocks**（block_size 256 → 16k token，约 12 MB/池），
依据是上面那张表里 64 blocks 那一行：单层写入 0.46 ms，×8 层 = 3.7 ms，已经顶到
整个 decode step（2.3–4.0 ms）的上沿，再大就是纯倒退。

### 上一版写的第三条修正**不成立**，按实测改口

`docs/dev/mlx-varlen-kernel-interface.md` 原来写「把 store 放进 `mx.compile` 图里 →
12 MB 池 2.40 ms 降到 0.60 ms」。**复现不出来**：

| 形态（12 MB 池，8 层 × k/v） | wall |
|---|---:|
| 同一 compiled graph，池作为参数传入传出 | 5.234 ms |
| 同一 graph 的 eager 版 | 5.344 ms |
| 同一 graph 去掉 store（对照） | 0.027 ms |

**编译只买到约 2%。** 原因是 `mx.compile(outputs=[pool])`、以及池作为参数传入传出，
**都不产生 donation** —— 调用方仍持有池的引用，MLX 只能分配新 buffer，于是每层把
12 MB 池整个拷一遍。

测到的 0.024 ms/step 是**单池 rebind 链**（调用方丢掉引用，`pool = store(pool, ...)`
逐步重绑定）的情形，**与"池要跨 step 存活"不是一回事**——原来那条结论把两件事混成了一条。
**本轮最大的测量缺口**：8 个独立池各自 rebind 的正确测量没有做。
在补上这个测量之前，不能说"store 进编译图"是可行方案。

### MLX 0.32.4 的 gotcha（写 kernel 才会撞到）

- **`grid=` 是总线程数，不是 threadgroup 数。** `grid=(8,1,1), threadgroup=(32,1,1)`
  只起 8 个线程、1 个 threadgroup，不是 8 组。早期几个探针因此全是"一个线程在跑"，
  得到过假的等价结论。
- **template 只接受 `Dtype` / `int` / `bool`，不接受 float。** 传 `("SCALE", 0.25)`
  直接 `ValueError: Invalid template argument` → 标量（如 scale）必须走 buffer。
- **只有 `input_names` 会变成 buffer**，没有别的通道传一个标量参数。
- 注入的线程属性是 `uint3`，必须 `.x`，不能当标量用。
- **`atomic_outputs=True` 编译不过**（`_atomic` 在 device 地址空间上不可赋值）。
- 单输出也返回 list（长度 1）。
- **MLX 0.32.4 没有任何 `metallib` 加载 API**（`mx.fast.precompiled_cuda_kernel` 有 CUDA
  兄弟，Metal 没有）→ 只能 JIT from source。构建脚本的作用是"在 JIT 之前先失败"。
- **`verbose=True` 打出来的源码不是完整 translation unit** —— 缺 MLX 自己的 prelude
  （`float16_t` / `simd_sum` / `metal::max` 都来自 `mlx/backend/metal/kernels/utils.h`）。
- **`verbose` 的打印走 fd 1**，绕过 `sys.stdout`；pytest 捕获下必须 `os.dup(1)`。

### KV 池跨 step 存活的真实成本（2026-11-09 补，修正上一版的担忧）

上一版担心「池必须跨 step 存活 → 每层整池拷贝 → 64 blocks 要 3.7 ms/step」。
**实测把这个担忧消掉了**：测法是 8 个独立层池，每步 `pool = store_kv(..., pool)`
**重绑定**（等价于「请求持有新数组、旧数组释放」= 跨 step 存活的真实形态），
一次 `mx.eval` 收尾。

| 池（每层 k+v，8 层合计） | eager 8 层/step | compiled 8 层/step |
|---|---:|---:|
| 4 blocks（12.6 MB） | 0.358 ms | 0.241 ms |
| 16 blocks（50.3 MB） | 0.443 ms | 0.430 ms |
| **64 blocks（201 MB）** | **1.156 ms** | **1.043 ms** |

对照「每层单独 eval」（不跨层流水）是 2.05 / 3.48 / 8.38 ms——**一次 eval 收尾这个
流水形态本身就把存储成本摊掉了**。

三条结论：

1. **跨 step 存活的 rebind 链是快的**：64 blocks、8 层、一步全部写完 **1.04–1.16 ms**，
   不是担心的 3.7 ms。
2. **`mx.compile` 在这里只买到 0–6%**，与上一版记的「约 2%」同量级。
   **没有 donation 红利可拿**——`mx.compile(outputs=[pool])` 那条路复测依旧不成立。
3. **池上限 64 blocks 的理由要换**：不再是「store 贵」，而是 **decode attention 的
   O(context) 扫描**——单池 decode attention 在 64 blocks 时 **7.48 ms**
   （1 block 0.32 ms，近似线性于 context_len）。**store 便宜，attention 贵。**
   结论（64）碰巧仍然合理，依据要改。

峰值内存：8 池 × 2 × 0.4 MiB 常驻 = 6.0 MiB，跑 20 步后 active 14.3 MiB，
**没有随步数增长**（rebind 链确实在复用 buffer）。

### 接线还没做完（未提交的半成品留在草稿区）

> **状态更新（2026-11-10）**：本节关于 `mlx_backend.py` 接线损坏的结论已被后续
> 实现取代。Metal kernel 源码现位于 `nanovllm_omni/backend/mlx_kernel/`，
> `tests/test_mlx_kernel.py` 覆盖数值闸门；这些 kernel 仍未接入 `MlxStageRunner`。
> `backend/paged_kv.py`、`backend/paged_attention.py` 与 `docs/dev/paged-kv-draft/`
> 是当时记录的本地草稿指针，当前 checkout 中均不存在。

`backend/paged_kv.py` 与 `backend/paged_attention.py` 已写完但**没有验证跑通**，
`mlx_backend.py` 的接线改到一半是坏的（`DEFAULT_MAX_BLOCKS` 的 import 缺失导致 NameError；
`PagedContext` 的所有权错：`prepare_*` 里整体替换了对象，但各层仍指向旧的那份）。
**这三样都没有进版本库**——代码留在 `docs/dev/paged-kv-draft/`（本地草稿），
接线的两个 bug 位置记在这里，供下一轮接手：

1. `MlxConfig` 引用 `DEFAULT_MAX_BLOCKS` 但 import 被删 → 必须放在函数内或用字面量 64
   （模块顶层 import 会让 `backend` 包在无 mlx 时炸，`test_registry_import_does_not_load_a_framework`
   会失败）。
2. **`PagedContext` 必须原地 mutate**（`self._paged.update(...)`），不能整体替换——
   `wrap_model()` 把 ctx 注入了每个 `PagedAttention`，替换对象后各层还指着旧的。
3. `PagedAttention.__call__` 的 decode 分支假定 `ctx.requests[row]` 与 batch 行一一对应，
   `fresh = slice(length - len(slots), length)` 在 prefill 多 token 时正确、decode 时恒为 `[-1:]`，
   **未测**。

另外记下这一轮新测到的两条 MLX 事实：

- **`mx.fast.rope` 的 `offset` 最多一维**（`[B]`）；标量 int 或 `[B]` 都行，
  `[B,1]` / `[B,1,L,1]` / `[B,L]` 全部 `ValueError`。
  **负起点可用**：`left_padding=[3,0,5]` → `offset=[-3,0,-5]` 时，三行各自的真实列与
  `rope(offset=0)` **完全一致（diff 0.00e+00）**——这是 paged prefill 左补齐能复用
  同一套 RoPE 的依据。
- **用 `nn.Module` 子类包住 `mlx_lm` 的 `Attention` 并替换 `block.self_attn` 是可行的**：
  `parameters()` 键集合不变、`set_dtype()` 仍能穿透、logits **逐位相同**。

### 未做

- **kernel 还没接进 `MlxStageRunner`**（本轮刻意不碰 `mlx_backend.py`，避免和另一个
  subagent 撞文件）。接线是下一轮。
- prefill 的 varlen kernel 没写（本轮只做 decode）。
- 上面说的「8 池各自 rebind」的测量没做。
- trace 次数没测。

### talker 的 MLX 版（2026-11-10 补）

- `backend/mlx_talker.py`：`MiniMindTalker` 的 MLX 移植（fp32 的 RMSNorm、
  GQA + 逐头 q/k RMSNorm、rotate-half RoPE、SwiGLU、8 码本的 embed/head
  适配器、`embed_proj`/`codec_proj`/`spk_proj`、两个标量 scale）加上
  `MlxTalkerRunner`。调度、发射门、top-50 + 抽样、逐码本 last-3 的
  repetition penalty 都与 `backend/torch_talker.py` 一致；`talker_kv="windowed"`
  也接上了。
- `backend/mlx_backend.py`：`MlxBackend.make_talker_runner`；`MlxStageRunner.forward`
  改成先跑 body（`model.model`）再跑 head，body 的 hidden states 留在 runner 上，
  `bridge_hidden()` 返回它们（`[B, T, hidden]`）。
- 权重：`scripts/convert_minimind_talker_mlx.py` 把 checkpoint 里的 `talker.*`
  转成 `~/models/minimind-3o/mlx-talker`（`model.safetensors` + `config.json`），
  `--check` 用随机 bridge 比对两边的 logits。

**实测**（M2 Air，mlx 0.32.4 / torch 2.11.0 CPU fp16）：八个码本的 logits 最大
绝对偏差 **0.0625**，均值 0.0033–0.0118（序列长 1 / 2 / 8 / 12 / 33 / 64）。
logits 量级到 42，fp16 的格步是 0.03125，所以这是 fp16 舍入而不是模型差异；
ADR 0022 记的 teacher-forced 基线也是 0.0625。窗口版与 dense 版在同一绝对列上
的偏差同样是 0.0625。

**两边的 frames 不可比**：`torch.multinomial`（指数竞速）与
`mx.random.categorical`（Gumbel-max）走不同的随机流，同 seed 只对齐参数、不对齐
抽样。实测同一 seed、同一 bridge：帧数相同（16 帧），逐帧相同的 0 帧。

### thinker stage 在 MLX 上跑通（2026-11-10 补）

- `bridge_hidden()` 现在取第 `bridge_layer`（=3）层之后、final norm 之前的
  hidden，与 vendor、fork 同一处：recorder 包住那一层。实测对 vendor 的最大
  绝对差 **0.25**（幅值 10128，相对 2.5e-05），见 `tests/test_mlx_bridge.py`。
- `models/minimind_omni/stage_runner.py::decode_minimind_mlx`：MLX 版的 vendor
  decode 舞步——一个 dense batch（左补齐），采样配方复用
  `backend/mlx_backend.py::_make_sampler`（temperature、full-history
  repetition penalty、top-p、每序列一个 RNG key），post-EOS 的 filler clocking
  （先 `enter_token` 再 `pad_token`）与 `talker_tail` 停止条件与 CUDA 轨一致。
  MLX 的 recorder 只留最后一次 forward 的行，所以 bridge 来自一次全序列
  forward（fresh cache）重跑，理由同 CUDA 轨的 `recapture_bridge_minimind`。
- `models/minimind_omni/stage.py::ThinkerStage`：按 `extra["backend"]` 选驱动。
  `backend="mlx"` 时加载 `<checkpoint>/mlx-thinker`（不是 torch checkpoint），
  不 import torch 的 thinker 模块（不需要 triton），也不建 NCCL 进程组；
  `args.device` 是 torch 的旋钮（没有 CUDA 时读 `"cpu"`），不传给 MLX backend。

**实测**（M2 Air，mlx 0.32.4 / mlx-lm 0.32.2，fp16）：

- 中文 prompt「今天天气怎么样？」，`temperature=0.7`、`max_tokens=16`、
  `seed=42`：文本「今天的天气情况因地而异，建议您查看当地的天气预」，
  bridge `[38, 768]`（22 个 prompt 位置 + 16 个生成位置）。
- 与 vendor 的 bridge 对比（vendor 在 CPU 上跑 fp16，同一条 token 序列）：
  最大绝对偏差 **0.25**，bridge 量级 10128，相对偏差 **2.47e-05**。
- post-EOS filler：EOS 落在第 22 个位置，`content_len=22` + `talker_tail=8`
  → 生成 30 个 token（EOS 后 1 个 `enter_token` + 7 个 `pad_token`），
  bridge `[52, 768]`。
- 批：两条 prompt（各 22 token）在一个 dense batch 里跑，各自 `[30, 768]`，
  `thinker2talker` 交接出 `start_pos=22`、`num_steps=8`。

测试：`tests/test_minimind_thinker_mlx.py`（`pytest.importorskip("mlx")`，
没有 `~/models/minimind-3o/mlx-thinker` 时 skip）。

## 边界

- 本 ADR **不**承诺 omni 管线全部由 MLX 执行。MiniMind thinker → talker 的 bridge
  已接通；Code2Wav 仍是 torch/MPS 实现（ADR 0024），所以该链路跨 MLX 与 torch。
- 本 ADR **不**决定 kernel 如何编译加载——实现细节。源码位置见
  `nanovllm_omni/backend/mlx_kernel/`。
- 本 ADR **不**改 ADR 0019 的分工：大块归 MLX 轨，本 ADR 定的正是 MLX 轨的内部顺序。
