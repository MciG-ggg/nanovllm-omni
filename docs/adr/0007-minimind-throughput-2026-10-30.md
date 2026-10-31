# ADR 0007: MiniMind-O 多请求吞吐（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

## 问题

`minimind_omni` 是仓库主模型族。当前唯一入口是一次一个请求的同步
`Omni.generate`，所以多请求吞吐 ≈ 1 / 单请求 wall。本文记录瓶颈、改动与
改动后的实测吞吐。

## 环境与口径

- 机器与软件同 ADR-0006：WSL2 RTX 3050 4GB Laptop（4096 MiB，driver 591.86），
  torch 2.11.0+cu130，transformers 5.16.1；代码 `perf/colab-2x` @ `78121b6` + 本文改动。
- 模型：`jingyaogong/minimind-3o`（本地目录）+ `kyutai/mimi`（HF 缓存），`HF_HUB_OFFLINE=1`。
- 命令：`python -m nanovllm_omni.bench time --pipeline full --model <dir> --device cuda
  --runs 5 --warmup 1 --max-tokens 120 --seed 7`，六个 bench prompt，warmup 在计时外。
- 计时：`perf_counter` 包住 `omni.generate(...)`，即 thinker→talker→code2wav 的整请求 wall。
- 显存：实测 1002–1016 MiB（`max_memory_allocated`）。

## 先修三处测量/复现缺陷

没有这三步，后面所有数字都不可信。

1. **`total_ms` 漏掉 talker**（`bench/runner.py`）。`run_one_full` 把
   `stage_ms["thinker"]` 覆盖到 `generate_ms`、`stage_ms["code2wav"]` 覆盖到
   `decode_ms`，而 `total_ms` 是这些字段之和——talker（占 2/3）没有字段承载。
   CSV 因此报 2.3 s，真实 wall 是 7.0 s。修法：保留被测试锁定的 per-stage 映射，
   新增 `StageTimes.wall_ms`，`total_ms` 有 wall 时直接返回它；阶段分解仍由
   `stage_ms` 列承载。
2. **bench 没把 seed 传进 `SamplingParams`**（`bench/runner.py`）。`run_one_full`
   只 `torch.manual_seed(seed)`，请求内的 talker 拿不到 seed，于是"同 seed 同音频"
   这条承诺对 E2E 路径不成立。
3. **talker 全程不播种**（`models/minimind_omni/stage.py`）。talker 权重在首次
   调用时才构造，构造期消耗 CUDA RNG，而 talker 自己从不 `manual_seed`。结果：
   同一个 `(prompt, seed)`、同一份代码，不同进程分别给出 **4 / 18 / 112 帧**。
   修法：`TalkerStage.__call__` 里按 `sampling.seed` 播种，写法与
   `models/smolvlm/stage.py` 一致。

修完之后，同一命令重复执行、跨进程结果一致（两次独立进程：`short_02` 13 帧、
`medium_02` 5 帧，逐次相同）。

## 瓶颈分解（改动前，实测）

| stage | 时间 | 占请求 |
|---|---:|---:|
| thinker | ≈2250 ms | 32% |
| talker | ≈4700 ms | 67% |
| code2wav | ≈35 ms | 0.5% |

- **thinker 不是算力问题**：120 步 × 19.3 ms，而 hidden 768 / 8 层的单 token
  算力量级 0.05 ms。84% 在 `run_model`，profile 显示每步 **306 次 kernel launch**
  加 dynamo 的每调用开销。根因是 `enforce_eager=True`（ADR-0005 对 3050 的选择）。
  每步明细见 `docs/perf/minimind-thinker-host-2026-10-30.md`（本地产物）。
- **talker 是全序列重算**：每步对 `start_pos + step` 条 token 跑完整 forward，
  120 步 → O(n²)。这是改动前的主要成本。

## 改动与实测效果

### 1. 发射门关闭后提前退出（主要收益）

talker 的发射门要求 8 个 codebook 同时 active，判据是
`idx[i] = step - 7 + i < stop_pos[i]`。`idx[i]` 随 `step` **单调递增**，
所以一旦某个 codebook 的 `idx` 越过它的 `stop_pos`，门**永久关闭**，
其后每一步都不可能再产出帧。原实现仍会跑满 `num_steps` 次全序列 forward。

改动（`models/minimind_omni/stage.py`）：把门抽成 `_can_emit_frame(...)`，
门关闭时 `break`。因为剩下的步数对 `emitted_frames` 贡献恒为空，
**同一 seed 下输出逐字节不变**，只是不再做无用计算。

测试：`tests/test_minimind_talker_early_exit.py`（门在 `stop_pos` 处关闭；
关闭后 9…119 步全部为假；无 codebook 停止时门一直开）。

同一 seed、同一输出下的对比：

| cell | 请求 wall 中位 | 说明 |
|---|---:|---|
| seeded，未提前退出 | ≈6950 ms | talker 跑满 120 步 |
| seeded + 提前退出（eager） | 2692 ms | **2.6×** |
| seeded + 提前退出 + thinker CUDA Graph | 990 ms | **对上面再 2.7×** |

### 2. thinker CUDA Graph（既有开关，**已于 2026-10-30 转为 stage 默认**）

`--use-thinker-cuda-graph` 把 thinker decode 从 eager 派发换成 graph replay：
thinker 阶段 2250 ms → 580 ms（3.9×）。

转默认的依据（ADR-0005 修订要求“测量 + parity”）：

- 测量：本节六题 `--runs 5` 的 eager 2692 ms vs graph 990 ms（2.7×，整请求）。
- parity：graph 与 eager 同 prompt/同 seed 输出音频**逐字节相同**，两个 prompt
  均验证（`tests/test_minimind_graph_parity.py`，`@pytest.mark.smoke`）。

因此 `ThinkerStage` 现在默认 `enforce_eager = False`，bench 仍可用
`extra["use_thinker_cuda_graph"] = False` 测 eager cell；`DeployConfig`
依然不带 graph 旋钮（ADR-0005 的拒绝项没被重新打开）。

> 时钟状态提醒：借默认值生效后复测同一 cell，得到 1124–1476 ms（中位 ≈1250），
> 而本文表中的 990 ms 是在更高的 GPU 时钟状态下测的（复测时 `clocks.sm` 为
> 1725/2100 MHz、62 °C）。显式 `--use-thinker-cuda-graph` 与不传 flag 在同一
> 会话内给出相同数字，证明差异来自机器状态而不是默认值。**跨会话只比同一批
> 次内的相对值，不要拿不同天的绝对 wall 相减。**

### 3. 其余（噪声级，已记录不宣称收益）

- 重复惩罚改成一次索引除法：0.330 → 0.225 ms/步（占请求 <0.5%）。
  `models/minimind_omni/stage_runner.py` + `tests/test_minimind_repetition_penalty.py`。
- bridge buffer 先切片再 clone：0.13 ms/步，不可测，纯减少拷贝。

## 最终数字（六题，`--runs 5`，seeded）

| prompt | eager wall 中位 | graph wall 中位 | 帧数 | graph 显存 |
|---|---:|---:|---:|---:|
| medium_01 | 2747.5 ms | 1017.4 ms | 12 | 1016.0 MiB |
| medium_02 | 2782.2 ms | 962.9 ms | 9 | 1015.7 MiB |
| short_01 | 2609.6 ms | 1024.6 ms | 12 | 1016.0 MiB |
| short_02 | 2575.2 ms | 837.0 ms | 5 | 1015.4 MiB |
| short_03 | 2793.1 ms | 1034.2 ms | 14 | 1016.1 MiB |
| system_01 | 2637.1 ms | 725.6 ms | **0** | 1002.7 MiB |

- eager 中位 **2692 ms ≈ 0.371 req/s**；graph 中位 **990 ms ≈ 1.01 req/s**。
- 相对改动前（7020 ms，0.142 req/s）：eager **2.6×**，graph **≈7.1×**。
- 上述 graph 数字两次独立进程复跑一致（帧数、wall 都在 ±3%）。

## 诚实边界

1. **这些收益来自"跳过不可能出帧的步数"**。改动前那次 7020 ms 的测量因为
   talker 未播种，帧数是 112 的偶然值；播种后模型稳定只出 5–14 帧
   （`system_01` 是 0 帧）。所以：
   - 请求/s 提升 ≈7× 是真实的（同样的确定性输出，更少的无用计算）；
   - **音频秒/秒没有提升**（120 帧预算下实际只出 0.4–1.1 s 音频）。
   不能把本文的数字当作"单位音频成本降低"。
2. **"只出 5–14 帧"本身仍未定因**：按 vendor 语义应接近 112 帧（预算上限 120−8）。
   已确认的一处差异是 bridge 捕获：vendor 的 `MiniMindBlock.forward` 返回
   加过残差的流，而 fork 的 `ThinkerBlock.forward` 返回 `(mlp_delta, residual)`，
   `MiniMindThinker.forward` 写进 bridge 的是 MLP 增量（少了 `+ residual`）。
   用 seed 修复后的 A/B 显示它会让部分 prompt 变长、另一部分变 0 帧，
   **不是唯一根因**，因此本轮没有改它。
   如果这个 bug 修好、音频回到 112 帧，提前退出的收益会消失，wall 会回到
   接近改动前的量级——那时真正的杠杆是下一节的 talker 增量解码。
3. `system_01` 出 0 帧时 `code2wav` 不执行，该行 wall 偏小（726 ms），
   不可与其它行直接比。
4. 单请求串行入口。`max_concurrent` / `AsyncOmni` 目前不产生真实并发：
   fork 的 `Context` 是模块级全局、`runner.sampler` 被就地替换、RNG 是全局的，
   多线程会互相污染。所以本文的"吞吐"= 1 / wall，不是服务端批处理吞吐。
5. 数字来自 WSL + 单卡 3050，不能与 ADR-0003 的 Colab T4 数字相减。
6. 原始 CSV 与 profile 是本地产物（WSL `/tmp`），不进 git。

## 下一步（按收益排序）

1. **talker 窗口化 KV**：朴素 KV 缓存不正确（音频 buffer 的第 `p` 列要等 8 个
   codebook 到齐才定型，最后 8 列在相邻步之间会被改写）。正确形态是缓存
   `[0, step-8)`、每步只重算最后 8 列，O(n²) → O(8n)。验收：同 seed 下
   `audio_codes` 逐位相同（播种已就位）。
2. **修 bridge 捕获点**，然后用同 seed 的 A/B 与 vendor 输出对齐；
   这是让"音频秒/秒"变得有意义的前提。
3. **thinker graph 转默认**：补一条 graph vs eager 的 token 逐位 parity 测试，
   再改 `deploy` 默认值并同步 ADR-0005。

## 修正（2026-10-31）：talker 的成本不在 O(n²)

上面第 1 条把 talker 的窗口化 KV 当成“真正的杠杆”。实测不成立。

WSL / RTX 3050 / fp16，55 步 decode（T=27..81），分相计时（5 次中位数）：

| 相 | ms | 占比 |
|---|---:|---:|
| 音频 buffer 构造（`stage.py` 的嵌套循环） | 481.6 | 33% |
| 模型前向（全序列，全 lm_head） | 705.4 | 48% |
| 采样（8 码本 × `topk`／`softmax`／`multinomial`／`.item()`） | 278.0 | 19% |
| 合计 | 1457.7 |  |

各相之和 1465.1 ms，与一次跑完 1457.7 ms 一致，所以分解可信。

关键事实：**单步成本对序列长度几乎不变**——T=1 的前向 12.20 ms，
而 T=27..81 的 55 步平均 12.4 ms/步。模型只有 4 层、768 hidden，
算量在 3050 上是微秒级，12 ms 全是 kernel 启动与 Python 派发开销。
所以 KV 缓存（降 FLOPs）**不是**这里的主要杠杆，CUDA graph 才是。

真正的大头是 host 开销：

1. **buffer 构造 481 ms** — `for p in range(step): for i in range(min(p,8)):`
   里逐元素写 GPU 标量，55 步累计上万次小 op。改成常驻 buffer、每步
   只写一列（`buf[0, :k, col] = codes[:k, p]`），值完全相同。
2. **采样 278 ms** — 8 个码本各一次 `.item()`，一步 8 次 host sync
   （55 步 440 次）。把 8 个结果 stack 后单次 `.tolist()`，RNG 调用顺序
   不变。

这两项加起来 760 ms（52%），**改完与原实现逐位相同**——不涉及任何数值。
剩下的 705 ms 模型前向要上 CUDA graph，那是下一步。

### 实测：buffer 常驻（2026-10-31）

已实施第 1 项。`TalkerStage.__call__` 改为分配一次
`[1, 8, start_pos + num_steps]` 的 buffer，每步只写刚定型的列，
不再每步重建。

正确性：同一 prompt + seed 7 + `max_tokens=64`，改前改后

```
audio_codes  torch.equal = True，差异位置 0 个（56 帧 × 8 码本）
WAV          sha256 前缀 d8c35d04dd57540a，均为 215084 字节
```

速度（talker 墙钟，同一请求，2 次预热后 5 次中位数）：

| | median | min | max |
|---|---:|---:|---:|
| 改前（每步重建 buffer） | 2174.7 ms | 2123.2 | 2303.1 |
| 改后（常驻 buffer，只写列） | 1379.4 ms | 1296.3 | 1399.7 |

**-795.3 ms，即 -36.6%。** 比隔离测得的 buffer 构造 481 ms 更多，
因为旧实现每步还要重新分配并填充整张 `[1,8,seq_len]` 张量（`torch.full`），
那部分未算进隔离测量。

端到端复核（`python -m nanovllm_omni.bench time --pipeline full
--prompts short_01 --max-tokens 512 --runs 3 --warmup 1`，WSL / RTX 3050）：

| | 改前 | 改后 |
|---|---:|---:|
| thinker | 442 ms | 464.9 ms |
| talker | 1881 ms | 1166.1 ms |
| code2wav | 33 ms | 34.5 ms |
| wall | 2342 ms | 1665.5 ms |
| 帧数 | — | 37 |

talker 占比从 80% 降到 70%。thinker 的 464.9 落在噪声范围内（改的是 talker）。

剩余待做：模型前向的 705 ms（隔离测量），要上 CUDA graph。

### 否定的假设：采样侧的 host sync（2026-10-31）

上面原本写着“下一笔是采样侧的 8 次 host sync”。实测不成立。

把 8 个码本的 `.item()` 合并成步末一次 `.tolist()`（`multinomial` 仍是
8 次独立调用，所以 RNG 序列不变），输出确实逐位相同：
`torch.equal`、WAV sha 与改前一致。但速度没有收益。

交替跑的 A/B（同一请求，每轮每版 5 次中位数，共 3 轮）：

| 轮 | 仅常驻 buffer | + 合并 host 读 |
|---|---:|---:|
| 1 | 1372.3 ms | 1423.5 ms |
| 2 | 1420.5 ms | 1421.2 ms |
| 3 | 1418.1 ms | 1372.0 ms |

两版落在同一噪声带内（±50 ms），无法区分。所以采样相 278 ms 里
**不是** sync 占主导，而是每个码本的 8 个小 kernel（clone、温度除、
3 次索引除、topk、softmax、multinomial）—— 8 码本 × ~8 kernel = 每步
~64 个 launch。

因为无收益且多一层间接（dict + sorted + stack + zip），该改动已回退。
若要再碰采样，得把 8 个码本的 clone／温度除／topk／softmax 合成一个
`[8, vocab]` 的批量运算（`multinomial` 仍保留 8 次调用以保 RNG 序列），
预期上限只有 ~100 ms，且需要比墙钟更细的仪器才能证明。
