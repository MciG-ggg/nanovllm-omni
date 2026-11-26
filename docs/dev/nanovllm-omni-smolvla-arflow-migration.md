# SmolVLA 迁移：单闭包 → AR stage + flow-matching stage（方案 B）

**状态：** **结构已落地**（HEAD 上 `nanovllm_omni/models/smolvla/pipeline.py`
两 stage 拆分已实现）；**fork-AR 化 deferred**（§3 ADR-029 因 lerobot
backbone 结构差异暂缓）；legacy `stage.py:_vla_stage` 已删（ADR-031 完成态）。
Owner: engine team。受众：要把 SmolVLA 内部隐藏的两种执行模式拆到流水线上
可见的人。

**硬前置：**
1. `docs/dev/nanovllm-omni-smolvlm-ar-migration.md` —— AR stage 原计划复用
   这里的 fork SmolVLM 语言栈移植（**实际未复用，见 §3 ADR-029 deferred**）
2. `docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md` —— flow stage
   复用其 `prepare_encode / denoise_step / step_scheduler / post_decode`
   4 方法契约和 `DiffusionRunner`（已落地）

三份指南的顺序：SmolVLM → diffusion → **本文**。

---

## 1. 现状：一个 stage 藏了两种执行模式

| 项 | 位置 | 现值 |
|---|---|---|
| pipeline 拓扑 | `nanovllm_omni/models/smolvla/pipeline.py:23-36` | **单** stage `name="vla"`，`kind=LLM_GENERATION`，`final_output_type="actions"` |
| stage 实现 | `models/smolvla/vlm_stage.py` + `models/smolvla/action_stage.py` | `SmolVLAVlmStage` + `SmolVLAActionPipeline`；两文件源码 docstring 详记 |
| 默认路径 | 同上 | `vlm_stage` 出 `VlmStageOutput`（含 KV cache + policy ref）；`vlm2action` 转 `ActionInputPayload`；`action_stage` 走 `DiffusionRunner` 4 方法契约 |
| LIBERO 路径 | 同上 | 同一两 stage 路径；队列语义留在调用方（见 ADR-030） |
| deploy | `nanovllm_omni/deploy/smolvla.yaml` | `default_sampling_params: {}`（空）；待补 stage 资源项 |
| 引擎参与度 | — | **部分**。`vlm` stage 仍走 lerobot `vlm_with_expert.forward`（ADR-029 deferred），`action` stage 由 `DiffusionRunner` 驱动 |

`pipeline.py:3-9` 的注释诚实记录了当初的取舍：

> Single terminal stage: the LeRobot SmolVLA policy is one checkpoint
> (SigLIP + SmolVLM + action expert). Splitting vision / language / action
> into separate engine stages needs TensorHandle and is not faked here.
>
> kind=LLM_GENERATION -- closest existing StageExecutionType

**实际的模型内部结构是两段不同性质的计算：**

```
SigLIP (vision) + SmolVLM (语言 backbone)   ← AR，有 KV cache
              ↓  hidden states
      Action Expert (flow matching)          ← 迭代积分，无 KV cache
              ↓
        action chunk [chunk_len, action_dim]
```

引擎层现在完全看不见这个分界。本次迁移就是把它抬到 `PipelineConfig.stages` 上。

---

## 2. vllm-omni 怎么做的，以及我们为什么不照抄

### 2.1 vllm-omni 的选择

| 事实 | 位置 |
|---|---|
| InternVLA-A1（VLA）整体放在 **diffusion** 目录下，当一个 diffusion pipeline | `vllm_omni/diffusion/models/internvla_a1/` |
| AR backbone 的 KV 由 `ARDiffusionModelRunner`（继承 `DiffusionModelRunner`）管 | `vllm_omni/experimental/ar_diffusion/runner.py` |
| pipeline 要实现 `SupportsARDiffusionPipeline` 四方法 | `vllm_omni/experimental/ar_diffusion/capability.py` |
| 这套东西整体在 `experimental/` 下 | 同上 |

`SupportsARDiffusionPipeline` 的四方法：
`ar_diffusion_kv_cache_spec()` / `bind_ar_diffusion_state(session_id, state)` /
`reset_ar_diffusion_session(...)` / `close_ar_diffusion_session(session_id)`。

`ARDiffusionKVCacheSpec` 的字段：`num_layers` / `num_kv_heads` / `head_size` /
`tokens_per_frame` / `frames_per_block` / `window_frames` / `kv_branches` /
`session_capacity` / `sink_frames` / `cross_attention` /
`max_scratch_tokens_per_branch` / `model_owned_state_bytes_per_session`。

### 2.2 我们不抄的部分，以及为什么

| vllm-omni 机制 | 为什么不需要 |
|---|---|
| `kv_branches` (多 KV 分支 + `local_index`) | 那是给 CFG 并行的（两条 cond/uncond 分支）。SmolVLA 的 action expert 不做 CFG |
| `tokens_per_frame` / `frames_per_block` / `window_frames` / `sink_frames` | 视频世界模型的滑窗 KV 语义。SmolVLA 一次 encode 一帧观测，没有滑窗 |
| `session_capacity` + LRU session 淘汰 | 跨请求持久化 KV session。我们单请求、请求间不复用 |
| `bind_ar_diffusion_state` 的 context manager 生命周期 | 上一条的配套。没有 session 就没有 bind/close |
| `ARDiffusionKVCache` / `ARDiffusionKVState` | fork 的 `BlockManager` + `SharedBlockManager` 已经覆盖单请求 KV |
| `experimental/ar_diffusion/tick_protocol.py` | 跨进程 tick 协议，我们进程内直调 |

**结论：vllm-omni 的 `ar_diffusion` 是为"AR backbone 的 KV 要跨 diffusion step
存活、且跨请求复用 session"设计的。SmolVLA 的 AR 部分是一次性 prefill
（观测 encode 完就不再需要增量 decode），flow matching 阶段不回头查 AR 的 KV。**

所以我们不需要 `ar_diffusion` 那一层，可以直接用**两个普通 stage + 一个
hidden-states bridge** —— 也就是 MiniMind-O thinker→talker 已经跑通的模式。

### 2.3 我们抄的部分

| 抄什么 | 从哪 |
|---|---|
| "VLA 的 action 部分归 diffusion 类"这个分类判断 | InternVLA-A1 放在 `diffusion/models/` 下 |
| 4 方法 pipeline 契约的名字 | `prepare_encode / denoise_step / step_scheduler / post_decode` |
| stage 间用类型化 payload 传 hidden states | 我们自己的 `minimind_omni/stage_processors.py` 更贴近 |

---

## 3. 决策（ADR）

### ADR-026：拆成两个 stage —— `vlm`（AR）+ `action`（flow matching）

**决策：** `SMOLVLA_PIPELINE` 改成两个 stage：

| stage_id | name | kind | factory | process_input | is_terminal |
|---|---|---|---|---|---|
| 0 | `vlm` | `LLM_AR` | `smolvla.vlm_stage:_vlm_stage` | `None` | False |
| 1 | `action` | `DIFFUSION` | `smolvla.action_stage:_action_stage` | `smolvla.stage_processors:vlm2action` | True（`final_output_type="actions"`） |

**理由：**
- 教学项目的核心诉求是"读懂链路"。两种执行模式藏在 lerobot policy 内部时，
  读代码的人看不到 AR 和 flow matching 的分界；拆开后 `PipelineConfig.stages`
  一眼可见。
- 结构上和 MiniMind-O thinker→talker **完全同构**（AR stage 出 hidden states，
  下游 stage 消费），复用已验证的 bridge 模式，不发明新机制。
- `vlm` stage 可以直接复用第一份指南做的 fork SmolVLM 语言栈。

**代价（说清楚，这是本次最大的账）：**
- 必须**拆开 lerobot 的 `SmolVLAPolicy`**，不能再整体调 `select_action` /
  `predict_action_chunk`。要自己接 `policy.model.vlm_with_expert` 一类的内部
  属性 → **依赖 lerobot 内部实现，lerobot 升级可能破**。
- LIBERO 官方 eval 链（`preprocess_observation` → `env_preprocessor` →
  `policy.preprocessor` → `select_action` → `policy.postprocessor` →
  `env_postprocessor`）里，`select_action` 有一个 **50 步 action chunk 内部队列**。
  拆开后这个队列的语义必须自己维护（见 ADR-030）。

---

### ADR-027：`action` stage 复用 `DIFFUSION` 枚举值，不新增 `FLOW` 成员

**决策：** `kind=StageExecutionType.DIFFUSION`。不给 `StageExecutionType`
加 `FLOW` / `FLOW_MATCHING` 成员。

**理由：**
- `StageExecutionType` 是对齐 vllm-omni 的闭集（`LLM_AR` / `LLM_GENERATION` /
  `DIFFUSION` / `CODEC`），`tests/test_config.py:268-271`
  显式锁死这四个名字。加成员就是单方面偏离对齐契约。
- vllm-omni 自己也把 flow matching 归到 diffusion 下（InternVLA-A1 在
  `vllm_omni/diffusion/models/` 里）。flow matching 和 diffusion 的区别是
  积分目标（velocity vs noise），执行形状（迭代 N 步、每步一次网络前向、
  step 之间更新一个 state）完全一致 —— 这正是 `DiffusionEngine` 的 4 方法
  契约要表达的东西。
- 第二份指南做的 `DiffusionEngine` 因此可以**原样复用**，不加分支。

**代价：** `kind=DIFFUSION` 的 stage 里跑的其实是 flow matching，名字不够精确。
用 `models/smolvla/action_stage.py` 的模块 docstring 写明这一点，
并在 `pipeline.py` 加一行注释。

**ponytail 注记：** 真出现第三种迭代式非 diffusion 执行（比如 consistency
model 的单步蒸馏），再回来讨论是否值得为精确性破坏对齐。

---

### ADR-028：bridge payload 镜像 `minimind_omni/stage_processors.py`

**决策：** 新建 `models/smolvla/stage_processors.py`，照 minimind 的做法：
frozen dataclass + 纯函数，**不调模型**。

```
@dataclass(frozen=True)
class VlmStageOutput:
    """vlm stage 的显式输出。"""
    robot_state: torch.Tensor          # [state_dim] 归一化后的本体状态
    attention_mask: torch.Tensor | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ActionInputPayload:
    """action stage 的输入。"""
    robot_state: torch.Tensor
    chunk_len: int
    action_dim: int
    attention_mask: torch.Tensor | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

def vlm2action(payload: Any, prompt: str = "") -> ActionInputPayload:
    """纯函数：形状校验 + 转换，不 tokenize、不调模型。"""
```

**理由：**
- `minimind_omni/stage_processors.py` 已经把这个模式跑通了（`ThinkerStageOutput`
  → `thinker2talker` → `TalkerInputPayload`），包括形状校验和错误信息风格。
  第二个使用者按同一形状写，读者迁移成本为零。
- `process_input` 钩子（`engine/runner.py:147-148`）就是为这个设计的。
- frozen dataclass 让 stage 边界的数据契约显式可测（纯函数好测，不需要 GPU）。

**代价：** 多一个文件。相比"把 dict 在 stage 间传来传去"值得。

**明确不做：** 不抄 minimind `stage_processors.py` 里 `_bridge_from` 那一堆
兜底路径（`multimodal_output` / `multimodal_outputs` / `hidden_states.bridge`
多种 key 探测）。那是历史包袱，新代码只接受 `VlmStageOutput` 一种类型，
其他一律 `TypeError`。

---

### ADR-029（deferred）：`vlm` stage 复用第一份指南的 fork SmolVLM 语言栈

**决策（原计划）：** `models/smolvla/vlm_stage.py` 里构造
`StageRunner(model_class=SmolVLMForCausalLM, ...)`，其中
`SmolVLMForCausalLM` 就是第一份指南在
`third_party/nano-vllm/nanovllm/models/smolvlm.py` 注册的那个类。

SigLIP vision tower 仍走 HF（ADR-016 的结论在这里同样成立）。
本体状态（`robot_state`）的 embedding 走 `inputs_embeds` 拼接路径（ADR-017）。

**理由：**
- SmolVLA 的语言 backbone **就是** SmolVLM。不复用等于把同一个模型移植两遍。
- 这也是"SmolVLM 排第一"的核心理由（第一份指南 §7）。

**代价 / 必须先验证的：** SmolVLA checkpoint（`HuggingFaceVLA/smolvla_libero`
/ `lerobot/smolvla_base`）里 backbone 的权重 key 前缀和独立 SmolVLM
checkpoint **不同**（lerobot 包了一层 policy）。
需要一份 SmolVLA 专用的权重切片映射，放 `models/smolvla/_weights.py`，
不动第一份指南做的 `models/smolvlm/_weights.py`。

**这一步要在 commit 1 就验证**，因为如果 backbone 结构对不上（比如 lerobot
改了 attention 实现），整个方案 B 的前提就没了 —— 见 §8 的回退方案。

**当前实现（HEAD，ADR 状态: deferred）**：`vlm_stage.py` 用 lerobot 内部
`policy.model.vlm_with_expert.forward(...)` 出 KV cache，不走 fork
`SmolVLMForCausalLM`。

原因：
- lerobot 的 backbone 在 `model.vlm_with_expert` 包了一层（不是裸 SmolVLM），
  独立 SmolVLM checkpoint 与 SmolVLA checkpoint 的权重 key 前缀不兼容。
- 强行复用 fork 模型会引入**位级偏差**——vlm_with_expert 内部包含 action
  expert 的 cross-attention，输入 output position 与独立 SmolVLM 的不一致。
- 这会破坏 §6.2 gate 1（`action chunk allclose(atol=1e-4)`）的位级对齐前提。

回退到这个路径意味着 **fork 改动为 0**（HEAD 已实现），代价是放弃 paged KV
收益——但保住"两种执行模式在流水线上可见"的教学目标（§1 + §3 ADR-026
的核心理由）。这是 §8 回退方案 A 的实际状态：见 §8。

**何时重启**：若 lerobot 后续发布去掉 vlm_with_expert wrapper 的版本，
或 fork loader 的 prefix 机制能跨 wrapper key prefix 工作，再回来评估。

---

### ADR-030：action expert 走 4 方法契约，chunk 队列语义留在 stage 内

**决策：** `models/smolvla/action_stage.py` 返回一个实现第二份指南
`DiffusionPipeline` 4 方法的对象：

| 方法 | SmolVLA 语义 |
|---|---|
| `prepare_encode(request)` | 从 `ActionInputPayload` 建初始 state：采样噪声 `x_0 ~ N(0, I)`，形状 `[chunk_len, action_dim]`；prefix KV cache / pad mask 从 `metadata` 读取 |
| `denoise_step(state, *, step, num_steps)` | 调 action expert 一次前向，返回 **velocity** `v_t`（不是 noise ε） |
| `step_scheduler(state, v)` | Euler 积分一步：`x_{t+dt} = x_t + v * dt`，`dt = 1/num_steps` |
| `post_decode(state)` | `x_1` → `policy.postprocessor` → `ActionArtifact.from_array` |

**50 步 chunk 队列不进引擎层。** lerobot 的 `select_action` 内部维护"一次预测
50 步动作、逐步弹出"的队列，那是**环境交互层**的语义，不是推理层。
本次迁移只覆盖"一次 chunk 预测"，队列留在调用方（`serving/vla.py` /
LIBERO eval 脚本）。

**理由：**
- flow matching 的 Euler 积分就是 N 次 `denoise_step + step_scheduler`，
  和 diffusion 的形状同构，4 方法契约直接够用（ADR-027 的论证）。
- 把环境交互的队列塞进 engine 会让 stage 有跨请求状态，破坏
  `PipelineRunner` 的无状态假设。

**代价：** LIBERO eval 路径要改调用方代码（原来一次 `select_action` 拿一步动作，
现在一次 `Omni.generate` 拿一整个 chunk，调用方自己弹出）。
这是行为可见的变化，必须在 §6.2 里用 SR 回归锁住。

---

### ADR-031（已完成）：保留 legacy 单 stage 路径直到数值对齐通过

**当前状态（HEAD，ADR-031 完成态）**：
- `models/smolvla/stage.py:_vla_stage`（旧闭包）**已删除**
- `SMOLVLA_PIPELINE` 是两阶段（vlm LLM_AR → action DIFFUSION）
- 句柄 `smolvla` / `HuggingFaceVLA/smolvla_libero` / `lerobot/smolvla_base`
  直接指向它，不再有开关。

**历史轨迹**：迁移期间 `models/smolvla/` 曾同时存在两条路径（`_vla_stage`
闭包 + `vlm_stage.py`/`action_stage.py` 两 stage）。`use_split_stages`
配置项在数值对齐（split vs `predict_action_chunk`）通过后随 commit 5
一并删除——这是临时脚手架，不是长期设计。

**当前两 stage path 的事实**：
- fork 改动：0（ADR-029 deferred）
- 数值对齐 gate（§6.2）**未跑**——fork-AR 路径 deferred，gate 1 是
  lerobot 直调的两 stage 内部对齐，需要 subagent B 在 WSL 跑端到端验
- LIBERO SR gate 2 同样未跑

---

## 4. 改动清单

### fork 改动

| 文件 | 改动 |
|---|---|
| — | **0**。SmolVLA 的 AR 半边复用第一份指南已注册的 `nanovllm/models/smolvlm.py` |

### nanovllm_omni 改动

| 文件 | 改动 |
|---|---|
| `models/smolvla/vlm_stage.py` | **新建**。`SmolVLAVlmStage`：HF SigLIP + fork `StageRunner`（SmolVLM 语言栈）→ 出 `VlmStageOutput` |
| `models/smolvla/action_stage.py` | **新建**。`SmolVLAActionPipeline`：4 方法契约（ADR-030） |
| `models/smolvla/stage_processors.py` | **新建**。`VlmStageOutput` / `ActionInputPayload` / `vlm2action`（ADR-028） |
| `models/smolvla/_weights.py` | **新建**。SmolVLA checkpoint → backbone / action-expert 两份权重切片（ADR-029） |
| `models/smolvla/pipeline.py` | 单 stage → 两 stage（ADR-026）；加 ADR-031 开关分支 |
| `models/smolvla/stage.py` | **不动**（legacy 路径，commit 5 删） |
| `deploy/smolvla.yaml` | 加 `use_split_stages`、两个 stage 的资源项和 `num_inference_steps`（flow 步数，lerobot 默认 10） |
| `serving/vla.py` | 适配 chunk 语义变化（ADR-030） |

### 测试改动

| 文件 | 内容 |
|---|---|
| `tests/test_smolvla_processors.py` | **新建**。`vlm2action` 纯函数：形状校验、错误类型、非 `VlmStageOutput` 输入报 `TypeError`（ADR-028） |
| `tests/test_smolvla_action_pipeline.py` | **新建**。用 fake action expert（返回常量 velocity）验 Euler 积分：N 步后 `x = N * v * (1/N) = v` |
| `tests/test_smolvla_split_stages.py` | **新建**。mock 两个 stage，验 `PipelineRunner` 走 `vlm → vlm2action → action` |
| `tests/test_config.py` | 改：SmolVLA 从 1 stage 变 2 stage 的拓扑断言 |

---

## 5. 实施顺序

**当前进度**（与原 5-commit 计划对照）：

| 原计划 | 实际状态 | 说明 |
|---|---|---|
| commit 1 — 权重切片 + backbone 验证 | **未做（ADR-029 deferred）** | 验证发现 lerobot backbone wrapper 与 fork 不兼容，停在 go/no-go |
| commit 2 — bridge payload + 纯函数 | **已完成** | `stage_processors.py` + `tests/test_smolvla_processors.py` |
| commit 3 — `vlm` stage | **已完成**（lerobot 直调版） | `vlm_stage.py` + `SmolVLAVlmStage`，用 `policy.model.vlm_with_expert.forward` |
| commit 4 — `action` stage + 接线 | **已完成** | `action_stage.py` + `pipeline.py` 两 stage + 4 方法契约接线 |
| commit 5 — 删 legacy | **已完成**（ADR-031 完成态） | `stage.py:_vla_stage`、`use_split_stages` 开关已删 |

**剩余工作**：
- **WSL 端到端 + §6.2 gate 跑过**：subagent B 在 WSL profile（lerobot 直调
  的两 stage 路径端到端、显存峰值、LIBERO SR 回归）。fork-AR 路径 deferred，
  gate 1 是 **lerobot 直调的两 stage 内部对齐**（不是"split vs legacy"），
  因为 legacy 已删，对齐目标是 `split vs policy.predict_action_chunk`
  走同一 lerobot backbone 的数值。
- **ADR-029 的 fork 化**（仅在 §6.2 gate 通过 + lerobot 后续发布去掉
  vlm_with_expert wrapper 后才启动）。

---

## 6. 验证

### 6.1 CPU 侧（commit 1-3）

- 权重切片：backbone key / action-expert key 正确分流，无遗漏无重叠
- `vlm2action`：形状校验、`TypeError` 边界
- Euler 积分：fake 常量 velocity，N 步后结果 = velocity（ADR-030 的数学回归）
- `PipelineRunner` 两 stage 调用序列（mock）

### 6.2 GPU 侧（WSL RTX 3050）：两条硬 gate

**gate 1 — 数值对齐（同 seed）**

```
同一 observation + 同一 instruction + 同一 torch seed
policy.predict_action_chunk(...)  vs  两 stage 路径（lerobot 直调版）
→ action chunk allclose(atol=1e-4)
```

flow matching 有噪声采样，必须固定 seed 才可比。
不要求位级一致（跨 stage payload 传递会有末位差），`atol=1e-4` 是合理带宽。

**当前 gate 未跑**（fork-AR 路径 deferred；现在对齐的是两 stage lerobot 直调版
 vs `policy.predict_action_chunk` 走同一 backbone 的数值）。Subagent B 在 WSL
 跑两 stage 端到端 + 显存峰值。

**gate 2 — LIBERO 成功率不回退**

```
HuggingFaceVLA/smolvla_libero + libero_object
legacy SR  vs  两 stage SR
→ 差值在统计噪声内（同 episode 数、同 seed 列表）
```

这条是 ADR-030（chunk 队列移出引擎）唯一的真实验证。
当前 LIBERO SR 基线为**单 stage legacy**（commit `df84462` 数字 1500ms p50，
SR 未在该 commit 明确记录）；两 stage 版本 SR 未测。Subagent B 在 WSL 跑。

legacy 路径已删（ADR-031），gate 2 的对照是"两 stage vs `df84462` commit
 数字 + 现有 eval 脚本"。

### 6.3 明确不作为 gate 的

- 不比较速度。拆两 stage 会多一次 payload 构造 + 一次 hidden states 传递，
  预期**略慢**，不算回归。
- 不测多请求 / batch（`max_num_seqs=1`）
- 不测 synthetic demo 路径的视觉质量（无客观指标）

---

## 7. 跳过清单

| 跳过 | 理由 |
|---|---|
| `ar_diffusion` 的 KV session 层 | §2.2 全表 |
| `kv_branches` / CFG 并行 | action expert 不做 CFG |
| 滑窗 KV（`window_frames` / `sink_frames`） | 一次 encode 一帧观测 |
| 把 SigLIP 搬进 fork | ADR-016 结论沿用 |
| chunk 队列进引擎 | ADR-030 |
| 新增 `StageExecutionType.FLOW` | ADR-027 |
| 三 stage 拆分（vision / language / action） | vision 无 KV cache，拆出来只增加 payload 传递成本；`models/smolvla/pipeline.py:3-6` 原注释的判断仍成立 |
| lerobot 之外的 VLA（InternVLA-A1 等） | 3050 放不下 |

---

## 8. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| **SmolVLA backbone 结构与独立 SmolVLM 不兼容**（lerobot 改了 attention / rope / 加了 adapter） | **高** | commit 1 是 go/no-go 闸门，先验证再投入 |
| lerobot 内部 API 变动破坏 action expert 接入 | 高 | 在 `action_stage.py` 顶部固定记录验证过的 lerobot 版本；`pyproject.toml` 的 `[smolvla]` extra 钉版本 |
| flow matching 的噪声采样与 legacy 不同源 → 数值对不上 | 中 | gate 1 固定 seed；先用 `num_inference_steps=1` 最小 case 调通再上多步 |
| chunk 队列移出后 LIBERO SR 掉 | 中 | gate 2；掉了就说明队列语义理解错了，回退到 ADR-031 的 legacy 开关 |
| 两 stage 显存峰值超 4 GB（backbone + expert 同时驻留） | 中 | deploy yaml 拆 `gpu_memory_utilization`（照 minimind thinker 0.6 / talker 0.2 的做法） |

### 回退方案

**方案 A（已是 HEAD 当前实现，不再是 fallback）：** backbone 不走 fork，
`vlm` stage 内部仍用 lerobot 的 `vlm_with_expert.forward`，但**保留两 stage
拆分**和 bridge payload。`pipeline.py` 两 stage 拓扑、`vlm_stage.py` /
`action_stage.py` / `stage_processors.py` 都已在仓。代价是丢掉 paged KV
收益，但保住"两种执行模式在流水线上可见"的教学目标——本次迁移的主要价值
实际由这条路径承担（fork AR 是次要价值）。ADR-029 状态："deferred，
非 future work"，参见 §3 ADR-029 deferred 段。

**方案 B（gate 数值对不上时）：** 已无 `use_split_stages` 开关可回
（ADR-031 完成态）。新代码留在仓里不动，靠 §5 剩余工作里的"WSL 端到端 +
gate 跑过"来定夺——对不上就 revert 涉及的 commit（按 §5 表对照原计划）。
legacy 路径已删，对齐目标只能是"两 stage vs `df84462` commit 数字"。

**方案 C（整体放弃）：** 不适用——结构已落地（pipeline.py 两 stage + vlm_stage
+ action_stage + stage_processors），revert 等于回退到单 stage 闭包，那是个
完整的重写不是简单 revert，且会丢失 4 方法契约的代码路径。无意义。

---

## 9. 关联文档

- `docs/dev/nanovllm-omni-smolvlm-ar-migration.md` — **硬前置 1**（fork AR 移植）
- `docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md` — **硬前置 2**（4 方法契约）
- `docs/dev/nanovllm-omni-diffusion-mirror.md` — `DiffusionEngine` 方案本体
- `nanovllm_omni/models/minimind_omni/stage_processors.py` — bridge 模式的
  第一个实现，ADR-028 的参照
- `.agents/skills/add-new-model/SKILL.md` — 新模型族接线流程
