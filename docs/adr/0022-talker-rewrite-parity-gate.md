# ADR 0022: talker 等价重写的范围、parity 门槛与验证环境

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0007（talker 吞吐）、0011（parity 标准）、0017（talker 重写 + 窗口化）、
  0020（重写边界）

## 决定

CUDA 轨的**这一轮**做到：talker 的模型代码自己写（ADR 0020 的边界）+ parity 绿。
不做窗口化 KV，不做批处理。

## parity 门槛

按 ADR 0011 的三层，不另造标准：

| 层 | 判定 | 这次的闸门 |
|---|---|---|
| **P0 自洽** | 同引擎、同输入、同 seed 跑两次 | `audio_codes` 逐位相同 + WAV sha256 相同 |
| **P1 行为** | 与参照实现的 token 序列是否一致 | 重写版 vs vendor：同 prompt 同 seed 的 `audio_codes` 序列**逐位相同** |
| **P2 数值** | 固定输入下数值是否在容差内 | teacher-forced 对照：audio logits 的 max abs diff 在容差内 |

**P2 不是 bitwise，而且这一轮明确不追求 bitwise。** 理由（ADR 0011 §2 已经写了）：
只有固定整条栈时逐 bit 才有定义。重写会改变 GEMM 的形状——最直接的是
**`lm_head` 只算最后一个位置**（ADR 0007 的分解里写的就是"全 lm_head"，
而采样只取 `logits[0, -1, :]`）。这个改动数学上等价，但 GEMM 从 `[T, H] × [H, V]`
变成 `[1, H] × [H, V]`，cuBLAS 选到不同的 kernel，**bit 就变了**。

> 修正：早前把"`lm_head` 只算最后位置"称为"逐位等价的白捡"是错的。
> 它是**数学等价**，落在 P2 层。

P2 的容差参照 ADR 0011 §4.2 实测的 talker teacher-forced 对照
（max abs diff 0.0625 / mean 0.0047 / 124 个真实采样位置），加上 margin 后写死。

**注意 ADR 0011 §4.2 的现状**：talker 目前**没有自动闸门**，那条对照是一次性 harness。
所以这一轮要顺带把它变成测试——重写版与 vendor 版的对照从此可重复执行。

## 验证环境：WSL（等它回来）

- **parity 基线必须在同一台机器上建立。** A800（SM80）与 RTX 3050（SM86）的
  fp16 GEMM kernel 不同 → 逐位基线不能跨机器。ADR 0014 已定"不跨机相减"。
- ADR 0007 / 0012 的全部基线都在 WSL 的 3050 上 → 在这台做，parity 数字能直接接上。
- **阻塞**：`mcigs-wsl` 的 tailscale 状态是 `offline, last seen 1d ago`，
  Windows 宿主也 `offline, last seen 12h ago`。
  → **这一轮在 WSL 回来之前只能写代码，不能出 parity 结论。**
- air-1（A800-80GB × 8）通着，但**没有现成环境**（系统 python 无 torch、
  无 conda env、PATH 里无 nvcc），且它是 ATG 那边在用的机器。
  要在它上面做，等于另起一条基线。

## 与 ADR 0017 的关系（返工量要说清）

ADR 0017 说"重写时窗口化 KV 一步到位，不做分桶过渡"。这一轮的范围是 dense 等价重写，
两者不矛盾，但**返工量必须诚实标注**：

| 部分 | 这一轮 | 下一轮（窗口化）要改什么 |
|---|---|---|
| embed / `embed_proj` / `codec_proj` / `spk_proj` | 写完 | 不动 |
| 层结构（RMSNorm、MLP、RoPE 的施加、residual） | 写完 | 不动 |
| 采样（8 码本、top-k、repetition penalty、交错调度） | 写完 | 不动 |
| 发射门 / 早退逻辑 | 写完 | 不动 |
| **attention** | 写 dense 版（全序列、无 cache） | **改成窗口版（cache + 最后 8 列）** |
| KV cache 管理 | 不写 | 新增 |

**返工量 = 一个 attention 子层 + 一套 cache 管理**，不是整个模型。
这是"先写 dense 版再改窗口"与"一步到位"之间真实的差别，写在这里是为了
让后面的人知道返工在哪，而不是以为要重写两遍。

## 后果

- 重写后的代码放在本仓库（`nanovllm_omni/models/minimind_omni/talker.py` 一类的新文件），
  checkpoint 目录只剩权重 + vendor 参照实现（ADR 0020）。
- **vendor 代码不删**——它是 P1/P2 的参照臂。运行时不加载它。
- 这一轮的产出是"等价 + 可重复执行的闸门"，不是性能。性能是窗口化之后的事。
- 这一轮不产出任何跨机可比的性能数字（ADR 0014）。

## 边界

- 本 ADR **不**决定重写代码的具体结构（类划分、张量布局），那是实现细节，
  按 AGENTS.md 的可读性要求写。
- 本 ADR **不**承诺这一轮结束 talker 的瓶颈——它只是把 talker 变成可改的代码。

## 实施记录（2026-11-09）

### 做了什么

| 文件 | 内容 |
|---|---|
| `nanovllm_omni/models/minimind_omni/talker.py` | dense 等价重写：`RMSNorm` / `TalkerAttention` / `TalkerMLP` / `TalkerBlock` / `TalkerEmbedding` / `TalkerHead` / `MiniMindTalker`；`load_talker` 从 `pytorch_model.bin` 取 `talker.*`，去前缀后 `strict=True` 加载 |
| `nanovllm_omni/models/minimind_omni/stage.py` | `TalkerStage` 默认 `talker_impl="rewritten"`；`extra["talker_impl"]="vendor"` 切回参照臂。驱动循环（采样、交错调度、repetition penalty、发射门、常驻 buffer）未改 |
| `tests/test_minimind_talker_parity.py` | P0（自洽）、P1（vendor vs rewritten 的 `audio_codes`）、P2（teacher-forced logits）三层常驻闸门 |

重写逐行对照的 vendor 部件：

- `TalkerModule.__init__`：layers / norm / lm_head / embed_tokens / codec_proj /
  embed_proj / text_scale / audio_scale / spk_proj / freqs 表。
- `TalkerEmbedding.forward`：8 个码本的 `base + adapter` 取平均。
- `TalkerHead.forward`：共享 base 加 8 个 rank-256 adapter。
- `MiniMindBlock.forward`：pre-norm、attention residual、MLP residual。
- `Attention.forward`：q/k 的 head-wise RMSNorm、RoPE 广播方式、
  `repeat_kv`、`seq_len > 1` 走 SDPA causal、`seq_len == 1` 走显式 softmax。
- `RMSNorm.forward`：fp32 归一化 + 乘 weight，再转回输入 dtype。
- `precompute_freqs_cis`：非 YaRN 分支（该 checkpoint `inference_rope_scaling: false`）。
- `MiniMindOmni.forward` 的 talker 段：`embed_proj(bridge)*text_scale +
  codec_proj(embeds)*audio_scale`，以及 `spk_emb` 的 codebook-0 mask。
- `stream_generate` 只取 `al[0, -1, :]`、`/0.2`、last-3 repetition penalty、
  `topk(50)` + `multinomial`：这些留在 `stage.py` 驱动里，未重写。

有意的语义差异：

- `lm_head` 只算最后一个位置（ADR 0022 正文已把它归为 P2 层，不是 bitwise）。
- talker 的架构参数从 `config.json` 读；vendor 用 `MiniMindConfig(hidden_size=...)`
  的默认值重建。该 checkpoint 上两者相等（8 heads / 4 kv heads / head_dim 96 /
  intermediate 2432 / eps 1e-6 / rope_theta 1e6 / max_position 32768）。
- 不实现 YaRN 分支、不建 dropout 模块（`dropout: 0.0`，eval 下无作用）。
- 重写版不保留 `past_key_value` / `use_cache` / `attention_mask` 参数：驱动只用
  `use_cache=False` 且无 mask，其余分支属于假想输入。

### parity 实测

| 层 | 结果 |
|---|---|
| P0 | **MPS 通过**：同 payload + seed 42 跑两次，`audio_codes` 逐位相同，24 帧；WAV sha256 相同。**MPS 数字不是 parity 结论**（ADR 0014） |
| P1 | **未跑**：WSL 仍 offline（本 ADR「验证环境」的阻塞条件），命令见测试 skip 信息 |
| P2 | **未跑**：同上 |

P2 容差写死为 `max abs diff <= 0.125`、`mean abs diff <= 0.01`，即 ADR 0011 §4.2
实测值（0.0625 / 0.0047）的 2 倍；该实测含 0.05% 的 bridge 输入差，本闸门两臂输入
完全相同，只剩 lm_head 的 GEMM 形状差（ADR 0022 正文）。

本机（MPS，仅冒烟、不是闸门）的旁证：

- 修正一处 `RMSNorm` 返回 dtype 的 bug 后，重建版与 vendor 在相同输入上逐层输出
  完全相等（embed / embed_proj / codec_proj / 4 层 / norm）。
- 公开 `forward` 因 lm_head 尾位置差：max 0.0156 / mean 1.9e-5（一次随机输入）。
- 用生产驱动做 teacher-forced 回放：对比 220 个真实采样位置，max 0.03125 /
  mean 1.3e-5，argmax 全部一致。

WSL 上首次运行需回填的参照栈（ADR 0011 §3）：RTX 3050 4GB Laptop /
驱动 591.86 / CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.16.1 /
Python 3.12.3 / fp16 / SDPA / `jingyaogong/minimind-3o` 本地目录 /
prompt `你好，请用一句话介绍你自己。` / `max_tokens=24` / seed 42。

### 返工量复核（对照 §「与 ADR 0017 的关系」）

对照表说下一轮只改「attention 一个子层 + cache 管理」。写完后这个估计**基本准确，
但少列了三处接线**：

1. `MiniMindTalker.forward` 的 RoPE 切片要从 `[:seq_len]` 改成按绝对位置
   `[start_pos:start_pos+len]`（窗口版每步 query 只有 8 列）。
2. `TalkerStage.__call__` 现在每步传 `bridge[:, :seq_len]` / `buf[:, :, :seq_len]`；
   窗口版要改成传 cache + 最后 8 列 + 当前位置。
3. cache 的生命周期要挂在 `TalkerStage` 上（每请求一份，请求结束释放）。

不动的部分与对照表一致：embed / 两个 proj / 层内 RMSNorm、MLP、residual / 采样 /
发射门 / 早退。`lm_head` 尾位置这一轮已经做完，下一轮也不再改。所以返工量仍是
「一个 attention 子层 + 一套 cache 管理 + 三处调用点」，不是重写第二遍。

### 窗口化的实施记录（2026-11-09）

窗口化 KV 已实现并接上驱动，默认仍走 dense：

| 位置 | 做了什么 |
|---|---|
| `talker.py` | `_LayerKVCache` / `TalkerKVCache`（每请求一份）；`_append_to_cache` 并入本步 k/v 后丢掉尾部 7 列；`_window_attention_mask` 显式掩码（窗口在 cache 末尾，`is_causal=True` 会挡掉缓存列）；`MiniMindTalker.forward` 的 RoPE 按 cache 长度取绝对偏移 |
| `stage.py` | `extra["talker_kv"] = dense \| windowed`，**默认 dense**；windowed 与 vendor 实现互斥；`__call__` 按 `cache.length` 切窗口 |
| `tests/test_minimind_talker_windowed.py` | 模型级闸门（CPU fp32，逐窗口比对最后一位置的 8 个头的 logits）；cache 前缀长度的性质测试 |

**实测**：fp32 CPU 上，每一步、每一层，窗口版与 dense 版在**同一个绝对列**上的
偏差 < 1e-4。

两条要记住的教训：

1. **对照必须按绝对列对齐。** dense 每步的列数（`start_pos + step`）与窗口的
   列数（8）不同，拿 `dense[:, 0]` 对 `window[:, 0]` 会把绝对列 1 和 0 比在一起——
   第一次跑就是这么得到 8.36 的假偏差。正确的对照是 `dense[:, lo:lo+n]` 对
   `window[:, :n]`，其中 `lo = cache.length`。
2. **dense 也要逐步增长前缀**，不能一次喂满整条——真实驱动每步只多一列。
   一次喂满 vs 逐步喂满不是同一件事（后者才是 stage 的形状）。

**仍未验证**：stage 级（`audio_codes` 逐位相同）在 CUDA 上验；窗口版的 P1/P2 parity；
窗口化后的吞吐数字（需要 WSL）。

### 下一轮的起点：cache 的语义已经想清楚（2026-11-09 补记）

窗口化的实现试到一半被中断（未提交，已回退）。留下的唯一有价值的东西是 cache 的
**语义**，记在这里，下一轮不用重新推：

- **窗口宽度 = 8**（= 码本数）。vendor 的发射门读的是对角线
  `audio_codes[i][step - 7 + i]`，所以只有最后 8 列还可能变，更早的列已经定型、
  可以缓存（ADR 0017）。
- **每层一份 cache**，`[B, H, T, D]` 的 keys / values。
- **每步 extend 之后要丢掉尾部 7 列**：窗口含本步新增的那一列共 8 列，
  尾部 7 列下一步会被重算，所以不进 cache。
- **cache 的长度就是下一步窗口的绝对起始位置**——RoPE 要按这个偏移取表，
  不是按 `[0:len]`（这是上面第 1 处返工点）。

为什么没有保留那半份代码：窗口版的数值正确性必须在 CUDA 上验 P1/P2
（ADR 0022 的门槛），本机验不了；留着半份实现等于留一段没人验证过的死代码。
**代码等能验的时候再写，语义先记在这里。**

### 未做 / 已知问题

- P1 / P2 尚无 WSL 实测基线；测试已入库，命令写在 skip 信息里。
- `load_talker` 只读 `pytorch_model.bin`（当前 checkpoint 的形态）；vendor 参照臂
  仍保留 safetensors 回退。若 checkpoint 改成 safetensors-only，默认路径要补。
- speaker embedding 路径保留了 vendor 语义，但它从未被赋值（ADR 0011 §5 的既有
  缺口，本轮不变）。
- P0 的 bridge 是固定合成张量：thinker 的 fork attention 需要 CUDA/triton，MPS 上
  跑不了。P0 因此只验证 talker + code2wav 的确定性，不验证端到端 thinker 输出。
