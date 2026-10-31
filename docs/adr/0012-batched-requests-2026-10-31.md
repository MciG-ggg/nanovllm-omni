# ADR 0012: 请求生命周期与批处理（对齐 vllm-omni 的形状）（2026-10-31）

- Status: Accepted
- Date: 2026-10-31
- 关联: ADR 0005（图策略在 stage，不在 DeployConfig）、0007（minimind 吞吐基线）、
  0010（并发不是杠杆）、0011（一致性分层标准）

## 问题

`PipelineRunner.run()` 是"请求即控制流"：一个请求从头走到尾，下一个请求才开始。
于是任何 stage 都不可能同时看到两个请求，批处理无从谈起，多请求吞吐 = 1 / 单请求 wall。

参考实现 `vllm_omni` 的形状是：

- **每个 stage 自己持有请求队列**，批由那个 stage 的 scheduler 从队列里选，
  不是由上层拼一个"批对象"；
- orchestrator 只做路由（stage N 的输出 → stage N+1 的队列）和生命周期；
- **默认不设等待窗口**：批只来自"恰好同时在队列里"的请求，想攒批要显式开
  等待窗口并付首请求延迟；
- **AR stage 的 `max_num_seqs` 生产配置 ≥ 4**，codec 类 stage 常常是 1；
- **跨 batch size 不承诺逐位等价**（`docs/configuration/stage_configs.md`：
  "Floating-point decoder outputs can differ across batch sizes; this PR does
  not claim bitwise or quality equivalence"）。

## 改动

### 1. stage 自己持有队列

新增 `engine/stage_engine.py`：

- `StageRequest`：排队中的一个请求（`request_id` + payload + sampling）。
- `StageEngine`：队列 + `add_request` / `has_pending` / `step`。
  默认策略是一次服务一个请求；能批的 stage 覆写 `step`。

各 stage 类（`TalkerStage`、`MiniMindOmniCode2Wav`、`SmolVLMStage`、
`SmolVLAVlmStage`、`DiffusionRunner`）继承 `StageEngine`，`__call__` 不变。

### 2. runner 只做路由

`engine/runner.py` 的公开形状对齐 `vllm.LLMEngine`：

| 方法 | 作用 |
|---|---|
| `add_request(prompt, sampling, request_id)` | 把请求放进第一个 stage 的队列 |
| `step()` | 让一个 stage 服务它持有的请求，把完成的推到下一个 stage |
| `has_unfinished()` | 是否还有请求在飞 |
| `run(prompt, sampling)` | `add_request` + drain，单请求调用方的便捷形式 |

调度策略是**对持有请求的 stage 轮转**（`self._next_stage` 游标）。参考实现每个
轮次轮询所有 stage、服务就绪的那个；单卡单进程上轮转是它的对应物。早期版本
"最早 stage 优先"会让新到的请求插到在飞请求前面（完成顺序倒挂），轮转消掉了
**这个入口处的**倒挂——但批内部的请求仍然受队头阻塞，见“边界”。

`process_input`（stage 之间的桥）挪到"提交给下一个 stage 时"执行，即边界处。

### 3. thinker 的批解码

`models/minimind_omni/stage_runner.py::decode_minimind` 收一组请求：

- 一个 fork `Scheduler` 装 N 个 `Sequence`，批多大由 `max_num_seqs` 决定；
- **每个序列一个 `torch.Generator`**：有 seed 时按该请求的 seed 播种，没有时从
  全局 RNG 抽一个（新建的 generator 自带固定默认种子，否则所有无 seed 的请求
  会抽到同一条流）。请求抽到的随机数因此不依赖它和谁同批——这是并发下
  "同 seed 同输出"的前提；
- bridge 行按序列累加：prefill 时各序列的 token 在 buffer 里是打包的，
  按 `num_scheduled_tokens` 切段；decode 时每序列一行；
- `stop_hit` 只结束命中它的那条序列（把该序列的 `max_tokens` 收到当前
  completion count，让 fork 自己的 `postprocess` 收尾），不打断同批其他请求。

`models/minimind_omni/thinker.py` 的 bridge 写入合并成一个固定切片
（`_bridge_buffer[:seq_len]`），prefill 和 decode 共用；bs=1 时与改动前逐位等价。

### 4. 入口先入队再 drain

`PipelineExecutor.run_many` / `submit_many` 与 `Omni.generate` / `AsyncOmni.generate`
改为：先把全部请求放进队列，再 drain。**这是批能形成的唯一入口**：晚到的请求
不会和已经在跑的请求拼批（与参考实现默认 `request_batch_max_wait_ms == 0` 同义）。
`py_generator=True` 仍然逐条执行，因此不批。

### 5. 部署形状：能力就位，默认仍是每步一个

`nanovllm_omni/deploy/minimind_omni.yaml` 的 thinker 保持 `max_num_seqs: 1`。
能力已经接上（把它调高就能批），但**现在调高是净亏**，见实测。

参考实现的 AR stage 是 `max_num_seqs >= 4`（`.claude/skills/add-tts-model/SKILL.md:180`）；
我们暂不跟这一条，因为参考实现的 AR stage 和 codec stage 是**并发引擎**，
批起来能真正重叠，而单卡上 talker 必须等 thinker，批只是把同一份工作重新排队。

### 6. 图策略：只捕获 bs=1

fork 的 `capture_cudagraph` 把各 batch 尺寸的图捕获进**同一个 memory pool**，
先大后小。与更大的尺寸一起捕获的 bs=1 图会换归约顺序：最终 logits 仍一致
（token 相同），但 bridge 行（中间 hidden state）差最后几位，于是 graph 与 eager
不再产生逐字节相同的音频。

因此 stage 默认 `max_capture_bs = 1`：**只捕获单请求的图**，bs>1 走 eager。
这让 bs=1 路径与 ADR 0007 记录的配置逐字节一致。代价是 bs>1 没有图可用：
eager 的每步 host 派发约为 graph 的 4 倍（ADR-0007：2250 ms vs 580 ms），
所以那个配置下 thinker 批起来反而更贵（见“批处理确实摊开了”一节）。
`extra["thinker_max_capture_bs"]` 可以抬高这个上限，bench 用它对比。

## 实测

WSL2 RTX 3050 4GB，torch 2.11.0+cu130；权重 `~/models/minimind-3o`；
`HF_HUB_OFFLINE=1`。口径：`python tools/experiment.py --experiment minimind --only batch`
与 `--only batch-graphed`（`--max-tokens 120 --seed 7 --runs 5 --warmup 1`），
warmup 在计时外，wall 取 5 次 p50。

两格的区别只有 `thinker_max_capture_bs`（`batch` = 1，即默认；`batch-graphed` = 8）。

**两格都需要把 `nanovllm_omni/deploy/minimind_omni.yaml` 里 thinker 的
`max_num_seqs` 临时改成 8**，否则仓库的默认值 1 会把每个请求串行化
（`_stage_args` 让 deploy 的值优先于调用方，所以 bench 无法从命令行覆盖它）。
测完改回 1。这是下面数字的完整复现条件，少这一步就复现不了：

### 只捕获 bs=1（默认配置下批走 eager）

| batch | wall ms | 每请求 ms | req/s | 峰值显存 |
|---|---:|---:|---:|---:|
| 1 | 2810 | 2810 | 0.36 | 1090 MiB |
| 2 | 6954 | 3477 | 0.29 | 1094 MiB |
| 4 | 15019 | 3755 | 0.27 | 1136 MiB |
| 8 | 31107 | 3888 | 0.26 | 1144 MiB |

### 每个 bucket 都捕获成图

| batch | wall ms | 每请求 ms | req/s | 峰值显存 |
|---|---:|---:|---:|---:|
| 1 | 2859 | 2859 | 0.35 | 1090 MiB |
| 2 | 5656 | 2828 | 0.35 | 1094 MiB |
| 4 | 13475 | 3369 | 0.30 | 1136 MiB |
| 8 | 29696 | 3712 | 0.27 | 1144 MiB |

### 结论：今天就发是净亏

和"8 个请求各跑一次"比：`8 × 2810 = 22480 ms`。两种配置都更慢——
31107 ms（+38%）和 29696 ms（+32%）。**所以 thinker 的 `max_num_seqs`
默认留在 1**，批处理作为能力存在，等 talker 不再是瓶颈再打开。

原因是 talker：单请求 wall 2810 ms 里 talker 占 2360 ms（84%），
而 talker 一次只服务一个请求、每步重算整条音频序列。批处理只作用于 thinker，
救不了这 84%。

### 批处理确实摊开了 thinker 的派发

thinker 阶段自身的耗时（整批，`batch-graphed` 那格）：

| batch | thinker ms（整批） | 每请求 |
|---|---:|---:|
| 1 | 449 | 449 |
| 2 | 598 | 299 |
| 4 | 1170 | 292 |
| 8 | 2086 | 261 |

8 个请求一共 2086 ms，而单跑是 8 × 449 = 3592 ms。每请求 449 → 261 ms。
**这就是批处理真正的收益**，只是它被 talker 淹没了，而且在
`max_capture_bs=1` 的默认配置下 bs>1 走 eager，thinker 反而更贵（8 个请求
3737 ms）——eager 的每步 host 派发是 graph 的约 4 倍（ADR-0007：2250 vs 580 ms）。

## 一致性

按 ADR 0011 的分层：

- **P0 自洽**（同引擎、同输入、同 seed、同批组成，跑两次相同）：成立。
- **P1 行为**（thinker 的 token 序列）：批内与单跑逐个相同。这条是用
  `pipeline="minimind_o_thinker_only"` 直接比对文本测出来的，**但那个脚本没有进 git**
  （它测的是现象，不是可复现的闸门），所以它是一条本地观测，不是仓库里的闸门。
  要把它变成闸门，需要给 bench 的 `batch` 子命令加一个 thinker-only 变体，
  比对文本而不是音频。本文没有做这件事。
- **音频**：**不承诺**跨批组成逐字节相同。批处理把多个序列打包进一次 prefill，
  kernel 选择随之改变，bridge 行差最后几位；talker 对 bridge 敏感（ADR 0007 记录的
  bridge 捕获分歧），所以帧数可能不同。

bench 的 `same_audio` 列现在只能部分覆盖这件事：它拿每个请求与它自己的单跑比，
但默认只有 6 个 prompt，`batch = [prompts[i % len(prompts)]]`，所以 bs=8 里有
2 个请求是重复的 prompt。两格实测都是 `8/8`，但那是 6 个不同 prompt + 2 个重复，
不是 8 个不同 prompt —— **不要把 `8/8` 读成“8 个互不相同的请求都逐字相同”**。

结论落在 ADR 0011 已经列出的原因里（“serving 的 batch 形状会改变 GEMM 与 kernel
选择”）。bs=1 的路径仍然逐字节一致，ADR 0005 / 0007 的闸门不变
（`tests/test_minimind_graph_parity.py` 在 `max_capture_bs=1` 下为绿）。

## 边界（明确没做到的）

- **晚到的请求不拼批**：`add_request` 发生在第一个 step 之后，就只能自己跑。
  与参考实现默认不设等待窗口一致；要攒批就得引入等待窗口，代价是首请求延迟。
- **HTTP 入口不会批**：`serving/openai_adapter.py` 每个请求只带一个 prompt，
  所以服务路径永远 `bs=1`。批只出现在 `Omni.generate(list)` / `run_many`。
- **talker / code2wav 不批**：`max_num_seqs: 1`，每步一个请求。
- **默认不批**：`max_num_seqs: 1`，理由见“结论”。
- **一个 step 会跑完整批**：`decode_minimind` 循环到 `scheduler.is_finished()`，
  所以批里最先 EOS 的请求也要等到最慢的那个到达 `max_tokens`。它经历的延迟是
  `batch max`，不是它自己的。参考实现能让每个请求先跑完先出。
- **`stage_ms` 的语义变了**：每个请求记的是它经历的那个 stage 的 wall。同一个
  批里 thinker 的 N 个请求会得到**相同**的 thinker 值（整批耗时），而 talker
  因为一次一个，记的是各自的耗时。两者不能相加，也不该横向比较。
- **`AsyncOmni.generate` 现在先入队再 drain**，因此结果是成组返回而不是逐个
  产出；本仓库本来就没有 token 级流式。
- **`input_sources` 仍然没被读取**：所有现有 pipeline 都是线性的，所以它是装饰性
  字段；改成非线性的拓扑前要先实现它。

## 复现

```bash
# 复现：先把 deploy 里 thinker 的 max_num_seqs 改成 8，跑完改回 1。
python tools/experiment.py --experiment minimind --only batch
python tools/experiment.py --experiment minimind --only batch-graphed
```

原始 CSV 在 `docs/perf/experiments/minimind/`（本地产物，不进 git），
里面 `same_audio` 列两格都是 `8/8`——但读法见上面“一致性”一节：
默认只有 6 个 prompt，bs=8 里有 2 个是重复的。
