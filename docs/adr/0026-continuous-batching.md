# ADR 0026: Continuous batching——增量式 stage step

- Status: Accepted
- Date: 2026-11-11
- 关联: 0011（parity 标准）、0012（批处理实测）、0014（双轨目标）、
  0025（MLX decode 用 fork Scheduler）

## 问题

原来的批是**快照式**的：`StageEngine.step()` 一次取走队列里的全部请求，decode 循环
在一个函数里跑到结束，中途到达的请求只能等下一批。参考实现 `vllm-omni` 的语义是
「stage 自己持有请求队列，scheduler 每步重选批」——晚到的请求在下一次调度就进入
正在跑的批，完成的序列当步离场。两者行为不同，ADR 0014 承诺的 CB 数字也就出不来。

## 决定

`StageEngine.step()` 的契约改为**增量式**：

- 每次 `step()` 只推进一个单位，返回**本步完成**的请求，而不是整个批的最终输出。
- `has_pending()` 必须包含在飞工作，否则 runner 会跳过还在跑的批。
- 批处理 stage 自己持有 scheduler 跨步存在；每步调用一次 `schedule()`。prefill
  优先于 decode，所以两步之间 `add()` 的请求会在下一步 prefill，再下一步与运行中的
  decode 同批。
- 完成的序列在同一步离开 `scheduler.running`，KV 行在同一步回收。
- 两条轨共用 fork `Scheduler`（ADR 0025），只在物理层分叉：
  - **CUDA**：paged KV，按 `slot_mapping` 写入。`recapture_bridge_minimind` 在批
    中途运行时先把 paged KV 摘掉、跑完再装回，避免新请求的 prefill 覆盖仍在
    decode 的序列的 slot。
  - **MLX**：新请求单独 prefill 成 `BatchKVCache`，再用 `extend` 右对齐补进正在
    跑的 cache；退出的行用 `filter` 回收。

实现入口：`ThinkerDecodeSession`（`models/minimind_omni/stage_runner.py`）持有
scheduler 与每序列状态；`ThinkerStage.step()` 是唯一的批处理 stage 覆写。其余
5 个 `StageEngine` 子类（talker、SmolVLM、SmolVLA、diffusion、code2wav）的
正确增量就是「一次一个请求」，继承基类即可，没有代码改动。

## 后果

- **默认部署仍是 `max_num_seqs: 1`（ADR 0012）：能力在，默认不开。** 实测 CUDA
  上 join 让新请求的延迟约为单独跑的 3.3 倍；MLX 上约为 1.1–1.3 倍，但相对
  「等旧批排空」仍是 10–20% 的小幅优势（见「结论与限制」）。默认值不变。
- **parity 判据分层。** 同批组成内，greedy 输出与串行逐 token 相同（P1 EXACT）；
  随机采样不承诺跨批组成逐位一致——fp16 下 join 布局的左填充长度改变归约顺序，
  末位漂移可能翻转一个 token。这条与 ADR 0012 的既有结论一致。
- **不引入 varlen。** MLX 仍是 dense padded，补齐成本还在；CUDA 是 paged。
- 测试：`tests/test_continuous_batching.py`（中途 join、提前离场、runner 调度
  队列为空的在飞批）、`tests/test_mlx_backend.py` 的两个 join/早退用例。
  真权重验证与性能数字见下节。

## 验证

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元：中途 join 与串行逐 token 相等 | `pytest tests/test_continuous_batching.py` | PASS（greedy，EXACT） |
| 单元：完成的请求先于慢请求返回 | 同上 | PASS（完成顺序 `fast` → `slow`） |
| 单元：runner 调度队列空但批在飞 | 同上 | PASS |
| MLX join / 早退 + cache 行回收 | `pytest tests/test_mlx_backend.py` | PASS |
| 真权重 MLX thinker | `pytest tests/test_minimind_thinker_mlx.py` | PASS |
| CUDA greedy join vs 串行 | WSL 一次性脚本 | 两臂 token 完全一致 |
| CUDA 早退顺序 | WSL 一次性脚本 | `['f', 's']` |
| MiniMind-O parity 闸门 | `PYTHONPATH=... HF_HUB_OFFLINE=1 python -u tests/test_minimind_hf_parity.py` | PASS（P1 text equal；P2 cosine=1.00000000，relative_mean=0.0517%） |

一次性脚本不提交（AGENTS.md「验证脚手架」）。CUDA 场景的复现方法见「实测二」。

## 实测一：MLX 轨（Mac）

工具：`tools/bench_mlx_scheduler.py`（本轮加了 `join` shape 与 peak/active memory 输出）。

- 机器：M2 Air，macOS 27.0.1 arm64，mlx 0.32.4
- 模型：`~/models/minimind-3o/mlx-thinker`（8 层 thinker）
- 代码：`c444d06`
- 输入：`single` = medium_01（28 token）；`ragged` = short_01+medium_01+short_02+medium_02
  （2+28+5+36）；`join` = A medium_01（28），B short_01（2），A 跑 3 步后加入 B
- 采样：temperature=1e-5（greedy），max_tokens=16
- runs/warmup：12/2；wall 为整段 generate
- 显存：peak 189.2 MiB，active 121.9 MiB

| shape | rows | prompt len | wall p50 (ms) | min / max |
|---|---|---|---|---|
| single | 1 | 28 | 44.14 | 34.72 / 51.96 |
| ragged | 4 | 2+28+5+36 | 130.25 | 119.87 / 150.60 |
| uneven | 4→3 | 2+28+5+36 | 81.80 | 74.60 / 101.68 |
| join | 1→2 | 28+2 | 124.87 | 115.12 / 139.70 |

join 的新请求：**B after join p50=105.69 ms**（97.52 / 120.31），**B alone p50=79.81 ms**
（72.23 / 85.77）。同配置另外 3 轮：B alone 43.10–65.89，B after join 55.62–80.41，
join/alone 比值 1.07–1.29；全部 4 轮合计 1.07–1.32。方向一致，绝对值受无风扇
热降频影响。

原始输出：`docs/perf/continuous-batching-2026-11-11/mlx-bench.txt`（本地产物，不进 git）。

## 实测二：CUDA 轨（RTX 3050 / WSL）

场景脚本（一次性，未提交）用 `ThinkerDecodeSession(backend="torch")` 先跑 A 三步，
`add()` B，记录 B 从到达到完成的延迟；A、B 各自单独跑作为对照。

- 机器：WSL，NVIDIA GeForce RTX 3050 4GB Laptop GPU
- 代码：`c444d06`；fp16；`max_capture_bs=1`（bs=1 有图，bs≥2 走 eager）
- 配置：`max_num_seqs=8`，`max_num_batched_tokens=512`，`gpu_memory_utilization=0.6`
- 输入：A = 「你好，请用一句话介绍你自己。」（27 token），
  B = 「用一句话介绍北京。」（23 token），均走 chat template
- 采样：temperature=1e-5（greedy），repetition_penalty=1.05，top_p=1.0，max_tokens=16
- runs/warmup：7/1（每个 shape 预热一次）
- 显存：peak 715.4 MiB

| 指标 | p50 (ms) | min / max |
|---|---|---|
| serial A | 77.14 | 74.83 / 81.16 |
| serial B | 76.79 | 69.39 / 79.24 |
| join B（从到达算） | 249.94 | 226.25 / 261.34 |
| join 整段 wall | 266.92 | 246.11 / 285.05 |

同配置另外 2 轮：serial A 79.20–86.35，serial B 81.49–93.06，join B 283.13–325.81，
join wall 310.19–350.92。三轮合计：serial A 77.14–86.35，serial B 76.79–93.06，
join B 249.94–325.81，join wall 266.92–350.92。方向一致，绝对值有 10–20% 的机器波动。

原始输出：`docs/perf/continuous-batching-2026-11-11/cuda-join.txt`（本地产物，不进 git）。

## 结论与限制

1. **MLX 上 join 有小幅收益，CUDA 上明显净亏。** 正确的对照是旧快照语义下新请求
   的等待：A 剩余步数 + B 单独跑。按测得的串行数字推算，MLX 上 join 把 B 的延迟
   从约 63–116 ms 降到 56–106 ms（各轮好 9–23%），整段 wall 相对串行 A+B 之和
   为 -14% 到 +1%——轮次间波动覆盖了这点差距，**收益只是新请求更早开始，不是
   稳定的吞吐提升**。CUDA 上 join 把 B 的延迟从约 140–163 ms 抬到 250–326 ms，
   整段 wall 从约 154–180 ms 抬到 267–351 ms。这些对照是**算式推出来的，不是
   直接测量**。
2. CUDA 变慢的机制明确：`max_capture_bs=1` 下 bs=2 走 eager，A 剩余的 13 步
   全部按 eager 双行步计价；ADR 0012 已记录同一机制。这是图捕获策略的代价，
   不是调度器实现的缺陷。
3. MLX 的批步本身没有 CUDA 那样的图/eager 落差，dense padded 的批步比单行
   decode 贵，但 join 让 B 的 16 步与 A 剩余的 13 步重叠，仍然赢过「等 A 排空
   再单跑」；赢的幅度在机器波动范围内，不能当成吞吐优化。
4. **默认不改。** `deploy/minimind_omni.yaml` 保持 `max_num_seqs: 1`；CB 是
   显式开启的能力，不是新的默认行为。要把这两种结论变成收益，先解决 CUDA 的
   图捕获策略（BM-3）或 MLX 的 varlen（ADR 0015）。
5. **未测**：吞吐随并发数的曲线、p99、talker/code2wav 的批、MLX 上 join 与
   随机采样的逐位关系、CUDA 上 `thinker_max_capture_bs>1` 时的 join。本文只覆盖
   thinker 单 stage 的两个请求。
6. Mac 无风扇，连续 shape 会热降频；同一配置轮次之间 B alone 可差 85%。上表给了
   min/max，结论只用方向，不用单点差值。
