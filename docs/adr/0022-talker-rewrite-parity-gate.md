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
