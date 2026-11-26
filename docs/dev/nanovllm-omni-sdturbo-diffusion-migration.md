# SD-Turbo 迁移：闭包 → DiffusionEngine（施工细则）

**状态：** **已落地**（C5 + C1+C2：fold diffusion client/engine/scheduler
into runner，4 方法契约在 `nanovllm_omni/diffusion/interface.py` 定义，
`runner.py` 驱动，`models/sd_turbo/stage.py:SdTurboPipeline` 与
`models/smolvla/action_stage.py:SmolVLAActionPipeline` 两个 model family
实现该契约）。Owner: engine team。受众：后续接入新 diffusion / flow matching
模型的人。

**本文与 mirror doc 的分工：**
mirror doc 定了**方案**（6 个新文件 ~520 LOC、2 处接缝、4 个 commit）。
本文补它没写的三件事：

1. 从 `vllm_omni/diffusion/` **抄什么、不抄什么**（§2 / §7）
2. 每个 ADR 的取舍理由（§3）—— mirror doc 只给了骨架代码，没给"为什么这样切"
3. SD-Turbo 1-step 的特殊性对验证策略的影响（§5.3）

方案本身不在这里重复，读 mirror doc。

三份迁移指南的顺序：SmolVLM → **本文** → SmolVLA。

---

## 1. 现状

| 项 | 位置 | 现值 |
|---|---|---|
| pipeline 拓扑 | `nanovllm_omni/models/sd_turbo/pipeline.py` | 单 stage `name="sd_turbo"`，`kind=DIFFUSION`，`final_output_type="image"` |
| stage 实现 | `models/sd_turbo/stage.py:SdTurboPipeline` | 4 方法契约实现者（`prepare_encode` / `denoise_step` / `step_scheduler` / `post_decode`），工厂返回 `DiffusionRunner(pipeline)` |
| 推理调用 | 同上 | 驱动者 `DiffusionRunner.run_sync` 调 4 方法循环，sd_turbo 的 `num_inference_steps=1` 实质是一次 `denoise_step + step_scheduler + post_decode` |
| deploy 默认值 | `nanovllm_omni/deploy/sd_turbo.yaml` | `steps=1`，`guidance=0.0`，512×512 |
| 引擎参与度 | — | **部分**。`PipelineRunner` 在 `kind=DIFFUSION` 分支接走 `DiffusionRunner.__call__`，SD-Turbo 是 `DiffusionPipeline` Protocol 的第一个实现者 |

数据侧契约（`OmniRequestOutput.from_diffusion` / `is_diffusion_output` /
`StageExecutionType.DIFFUSION` / registry 自动注册）**已经全部就位**，
详见 mirror doc §1 的钩子表。缺的只有 engine 层。

---

## 2. vllm-omni 源码地图：抄什么

参考仓库只读（`/Users/mcig/Projects/vllm-omni`）。要读的文件按优先级：

| vllm-omni 文件 | 抄什么 | 抄的程度 |
|---|---|---|
| `vllm_omni/diffusion/diffusion_engine.py:228-247` | `_resolve_execution_mode()` 的双模式判定逻辑 | **只抄 STEP_BATCH 分支**（ADR-021） |
| `vllm_omni/diffusion/diffusion_engine.py:99-110` | `supports_request_batch()` 能力探测的**形状** | 简化成一个 `supports_step_execution` ClassVar |
| pipeline 侧 4 方法契约 | `prepare_encode` / `denoise_step` / `step_scheduler` / `post_decode` 的**方法名和职责边界** | 全抄名字（对齐契约），签名简化 |
| `vllm_omni/engine/stage_client.py:23` | `stage_type: str` 类属性作判别符的**模式** | 抄模式，不抄 Protocol 树 |
| `vllm_omni/diffusion/inline_stage_diffusion_client.py:38` | `stage_type = "diffusion"` 硬编码的做法 | 直接抄 |
| `vllm_omni/entrypoints/omni.py:103-160` | `Omni.generate` **完全不分支**（AR / diffusion 统一出口） | 抄这个原则（我们的 `PipelineRunner.run` 同理） |

**方法名必须对齐**：`prepare_encode` / `denoise_step` / `step_scheduler` /
`post_decode` 是消费者可见的 pipeline 契约，属于 AGENTS.md「对齐契约」范围。
签名可以简化，名字不能改。

---

## 3. 决策（ADR）

### ADR-021：只实现 `STEP_BATCH`，不实现 `REQUEST_BATCH`

**决策：** `DiffusionEngine` 只有一条执行路径 —— 每 tick 调
`pipeline.denoise_step()` 一步。不实现 vllm-omni 的 `REQUEST_BATCH`
（`pipeline.forward(input_batch)` 一次跑完整循环）。

**理由：**
- `REQUEST_BATCH` 的存在意义是"多请求合进一次 forward 提吞吐"。本项目
  单进程单 GPU 单请求，`max_num_seqs=1`，合批对象只有一个，收益为零。
- 两条路径意味着两套代码 + 两套测试 + 一个模式判定分支，全是负债。
- 教学价值全在 step 模式（能看见 denoise 循环）；`REQUEST_BATCH` 恰好是
  "把循环藏回 pipeline 里"，和迁移目的相反。

**代价：** 失去 batch 吞吐。真出现多请求需求时，`_resolve_execution_mode`
的位置留着（`vllm_omni/diffusion/diffusion_engine.py:228` 是现成参考），
加一个分支即可。

**ponytail 注记：** `# ponytail: step-only; add REQUEST_BATCH when max_num_seqs > 1
becomes a real requirement` 写进 `diffusion/engine.py`。

---

### ADR-022：只做 inline，不做 IPC

**决策：** 只实现 mirror doc §2.3 的 `InlineDiffusionClient`。
不实现 vllm-omni 的 ZMQ client / worker 进程 / `DiffusionModelRunner`
的多进程模型。

**理由：** vllm-omni 的 IPC 层是为多副本 `StagePool` 和跨进程 GPU 分配设计的。
我们的并发上限由 `PipelineExecutor` 的 `asyncio.Semaphore(max_concurrent=1)`
决定，进程内一个线程池就够。

**代价：** 无法把 diffusion stage 放到独立进程 / 独立 GPU。4 GB 单卡本来也放不下。

---

### ADR-023：`RequestScheduler` 用 stdlib `deque`，不镜像 `BaseScheduler`

**决策：** `diffusion/scheduler.py` 只有 waiting / running 两个
`collections.deque` + `finish()`。**不**复制 vllm-omni 的 `BaseScheduler`
（它依赖 paged KV block 管理），也**不**实现 mirror doc 提到过的 `StepScheduler`。

**理由：**
- diffusion 没有 KV cache 要分页，`BaseScheduler` 的核心职责（block 分配 /
  抢占 / swap）全部不适用。
- 单请求场景下 scheduler 退化成"一个队列 + 一个 step 计数器"。用 stdlib
  能一眼看懂，自己造状态机不能。

**代价：** 没有优先级 / 抢占 / 公平调度。单请求不需要。

---

### ADR-024：`run_sync` 是主路径，`step_streaming` 是次要路径

**决策：** `DiffusionEngine` 两个入口：
- `run_sync(request) -> list[OmniRequestOutput]` —— 给 `PipelineRunner.run`
  调（同步，主路径）
- `step_streaming(request)` async generator —— 给未来的 SSE / streaming HTTP
  用（次要）

`run_sync` **不是** `step_streaming` 的 `asyncio.run()` 包装，两者共享同一个
`_denoise_loop` 私有方法，各自处理输出。

**理由：**
- 现在唯一的真实消费者是 `PipelineRunner.run`（同步）。
  `PipelineExecutor` 已经在外层用线程池做了 async 桥接，engine 内部再套一层
  event loop 是重复。
- `step_streaming` 先写但不接线，是为 SD 2.1 / SANA 这类多步模型的
  streaming 预留；mirror doc §2.3 已给骨架。

**代价：** 两个入口共享一个私有循环，多一层间接。比"同步路径绕 event loop"
的代价小。

---

### ADR-025：接受 SD-Turbo 上 step 拆分**没有性能收益**，动机是架构和教学

**决策：** 文档、commit message、代码注释统一口径：本次迁移**不承诺性能改善**。

**理由（必须诚实写明，否则后来人会误判）：**
- SD-Turbo 是 adversarial distillation，`num_inference_steps=1`。
  把 `pipe(...)` 拆成 `prepare_encode → denoise_step → step_scheduler →
  post_decode` 之后，denoise 循环**只跑一轮**。
- 单步下拆分纯粹是把一次调用摊成四次调用，理论上略慢（多几次 Python 调用）。
- 真正的收益在两处，都不是现在：
  1. **多步模型**（SD 2.1 = 50 步、SANA）接进来时，step 模式能 streaming、
     能中途 abort、能 per-step profiling
  2. **SmolVLA 的 action expert**（flow matching，N 步 Euler）复用这套
     4 方法契约 —— 见第三份指南

**代价：** 有人会问"那为什么现在做"。答案写在这里：因为 SD-Turbo 是最小的
diffusion 验证载体（1 步、确定性、512×512、fp16），用它把 engine 层调通，
比直接上 50 步模型好调试。

---

## 4. 改动清单

新增文件、接缝位置、LOC 预算见 mirror doc §2.1 / §2.2，不重复。
本文只列 mirror doc 之外需要注意的：

| 文件 | mirror doc 之外的补充 |
|---|---|
| `diffusion/engine.py` | 按 ADR-021 删掉 `REQUEST_BATCH` 分支；按 ADR-024 拆 `run_sync` / `step_streaming` 共享 `_denoise_loop` |
| `diffusion/scheduler.py` | 按 ADR-023 用 stdlib `deque`，不要状态机类 |
| `diffusion/client.py` | 按 ADR-022 只留 inline；`stage_type = "diffusion"` 硬编码 |
| `models/sd_turbo/stage.py` | 拆 `StableDiffusionPipeline` 时注意：`pipe.text_encoder` / `pipe.unet` / `pipe.vae` / `pipe.scheduler` 四个子模块，`guidance_scale=0.0` 意味着**不做 CFG**（不复制 latent、不拼 uncond embeds），比标准 SD 少一半计算 |
| `engine/runner.py` | 接缝 A：`if stage_cfg.kind == StageExecutionType.DIFFUSION` 分支。注意 `_last_stage_timings` 的计时逻辑要包住 diffusion 分支，否则 bench 拿不到该 stage 的耗时 |

**fork 改动：0。** diffusion 完全不碰 `third_party/nano-vllm/`。
这是它和另外两份指南的最大区别。

---

## 5. 验证

### 5.1 引擎层（commit 3）

mirror doc §5.1 已给 `IdentityPipeline` 的 fake 测试（3 步 × 0.1 = 0.3）。
补两条：

- `run_sync` 和 `step_streaming` 对同一 request 产出**相同结果**（ADR-024
  的两个入口不能分叉）
- scheduler 的 `finish()` 后 `has_requests()` 为 False（ADR-023 的状态机回归）

### 5.2 接入层（commit 4）：位级一致是唯一硬 gate

```
同一 sd-turbo checkpoint + 同一 seed + steps=1 + guidance=0.0
→ 迁移前 pipe(...).images[0]  vs  迁移后 DiffusionEngine 产出
→ 逐像素一致（np.array_equal），不是 allclose
```

SD-Turbo 1 步确定性，**必须位级一致**。不一致说明拆解时漏了某个
scheduler 状态（`sigmas` / `timesteps` 初始化）或 latent 缩放
（`vae.config.scaling_factor`）。这是拆 diffusers pipeline 最容易踩的坑。

**当前 gate 状态**：已跑（CPU 端 `tests/test_sd_turbo_pipeline.py` + WSL
GPU 端到端均过位级一致），subagent B 的 WSL profile 在复查 CI 路径。

### 5.3 明确不作为 gate 的

- **不比较速度**（ADR-025）。若测出略慢，记录数字即可，不算回归。
- 不测多请求并发（ADR-021 不支持）
- 不测 streaming（ADR-024 的 `step_streaming` 本次不接线，只有单测覆盖）

### 5.4 CI 门禁

mirror doc §5.3 已写。补一条：`diffusion/` 下所有文件必须能在
**没有 torch / diffusers** 的环境下 import（重依赖留在 pipeline 侧的
lazy import 里），这样 CPU-only CI 能跑引擎层单测。
加一个 `tests/test_diffusion_imports.py` 锁住。

---

## 6. 实施顺序

mirror doc §4 的 4 个 commit 状态：

| 原 commit | 实际状态 | 说明 |
|---|---|---|
| 1 — Protocol+dataclass | **已完成** | `nanovllm_omni/diffusion/interface.py` 定义 `DiffusionPipeline` Protocol / `StepState` / `DiffusionOutput` |
| 2 — Scheduler+Runner | **已完成**（合并为 runner） | `nanovllm_omni/diffusion/runner.py` `DiffusionRunner`，scheduler 用 stdlib `deque` 内联在 runner 路径 |
| 3 — Engine+Client | **已完成**（C1+C2 fold） | Client / Engine / Scheduler 被 fold 进 runner；只剩 `InlineDiffusionClient` 单路径 |
| 4a — `stage.py` 返回 `SdTurboPipeline` | **已完成** | `models/sd_turbo/stage.py` 第 224 行返回 `DiffusionRunner(pipeline)` |
| 4b — 接缝 A 分支 | **已完成** | `engine/runner.py` 接 `DiffusionRunner.__call__` |
| 4c — 删 `__call__` 兜底 | **已完成** | sd_turbo stage 不再有闭包路径 |

> **注**：项目命名上用 `DiffusionRunner` 而不是 vllm-omni 的 `DiffusionEngine`。
> 本文 §2/§3 中历史 ADR 提到的 `DiffusionEngine` 均指 `DiffusionRunner`，
> 见 `nanovllm_omni/diffusion/runner.py`。
>
> commit 4 拆 4a/4b/4c 的顺序风险论述已无关当前（commit 都已落地），保留作为
> 教学参考——后来人接入新 diffusion 家族时仍可按这个二分原则拆 commit。

---

## 7. 跳过清单

mirror doc §3 已有一份。本文明确追加"从 vllm-omni **不抄**"的部分：

| vllm-omni 组件 | 位置 | 不抄的理由 |
|---|---|---|
| `StagePool` 多副本路由 | `vllm_omni/engine/` | 单 GPU 单副本 |
| `ZmqDiffusionClient` / IPC | `vllm_omni/diffusion/` | ADR-022 |
| `DiffusionModelRunner` worker 进程 | `vllm_omni/diffusion/worker/` | 进程内直调 |
| `REQUEST_BATCH` 模式 | `diffusion_engine.py:228-247` | ADR-021 |
| `KVPrefetchJob` | `vllm_omni/diffusion/sched/interface.py` | diffusion 无 KV cache |
| diffusion metrics 累加 | `vllm_omni/metrics/stats.py:631` | `_last_stage_timings` 已够 |
| `DiffusionPipelineProfilerMixin` | `vllm_omni/diffusion/profiler/` | 已有 `utils/profiling.py:profile_range` |
| `submit_interaction_async` | `stage_client.py` | 无交互式编辑需求 |
| `regionally_compile` | `vllm_omni/diffusion/compile.py` | 未测量，不做编译优化 |

---

## 8. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| 拆 diffusers pipeline 时漏 scheduler 状态 → 图变了 | **高** | 位级一致测试（§5.2）直接抓；commit 4a/4b 拆开二分 |
| `vae.config.scaling_factor` / latent 缩放漏乘 | 高 | 同上；症状是整张图对比度异常 |
| `guidance_scale=0.0` 下误加 CFG 分支 | 中 | SD-Turbo 不做 CFG，代码里不写 uncond 路径就不会错 |
| 接缝 A 打断 `_last_stage_timings` | 中 | bench 单测 `tests/test_optim_bench.py` 已覆盖该字段 |
| `diffusion/` 引入 torch 顶层 import 破 CPU CI | 低 | `tests/test_diffusion_imports.py`（§5.4） |

**回退：** commit 1-3 是纯新增（`nanovllm_omni/diffusion/` 全新目录），
没有任何现有路径经过它们，revert 零风险。
commit 4 按 4a/4b/4c 拆开，任一步 revert 即回到闭包路径。

---

## 9. 关联文档

- `docs/dev/nanovllm-omni-diffusion-mirror.md` — **方案本体**，先读它
- `docs/dev/vllm-omni-diffusion-reference.md` — vllm-omni 侧结构梳理
- `docs/dev/nanovllm-omni-smolvlm-ar-migration.md` — 第一份（AR 路线）
- `docs/dev/nanovllm-omni-smolvla-arflow-migration.md` — 第三份，复用本文的
  4 方法契约跑 flow matching（`models/smolvla/action_stage.py:SmolVLAActionPipeline`）
- C5 commit `89a099a` / C1+C2 commit `b04e8fd` — fold diffusion client/engine/
  scheduler into runner；本文落地的两个 commit
