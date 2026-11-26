# SmolVLM 迁移：闭包 → fork AR Engine

**状态：** fork-AR 化**代码已完成**（HEAD），GPU 端到端 gate 待跑。Code path
是 `SmolVLMForConditionalGeneration`（fork-style 文本栈 + HF SigLIP 视觉塔）
+ `SmolVLMStage`（fork `StageRunner` 驱动 prefill/decode 循环）。
WSL RTX 3050 上的 §6.2 / §6.3-bis 数字 gate 待跑——这是唯一还缺的硬件 gate。
Owner: engine team。受众：要让 WSL bench 把 §6.2 gate 跑绿的人。

三份迁移指南的当前状态：
- 本文（SmolVLM）：代码已落地，GPU bench 待跑
- sd-turbo：4 方法契约已落地（见 §2 / §10 交叉引用）
- SmolVLA：两 stage 拆分已落地（lerobot 直调版，ADR-029 deferred）

---

## 1. 现状（迁移后）

| 项 | 位置 | 现值 |
|---|---|---|
| pipeline 拓扑 | `nanovllm_omni/models/smolvlm/pipeline.py` | 单 stage `name="vlm"`，`kind=LLM_AR`，`final_output_type="text"` |
| stage 实现 | `nanovllm_omni/models/smolvlm/stage.py:_vlm_stage` | 工厂返回 `SmolVLMStage` 实例 |
| stage 类 | `nanovllm_omni/models/smolvlm/stage.py:SmolVLMStage` | HF SigLIP 视觉塔 + fork `StageRunner` 文本栈 AR decode |
| 完整模型 | `nanovllm_omni/models/smolvlm/smolvlm.py` | `SmolVLMForConditionalGeneration`（vpm + connector + SmolLM2 文本栈 + lm_head），fork-style building blocks for decoder |
| deploy | `nanovllm_omni/deploy/smolvlm.yaml` | `gpu_memory_utilization: 0.85`、`max_num_seqs: 1`、`max_num_batched_tokens: 8192` |
| 引擎参与度 | fork `ModelRunner` + `StageRunner` | paged KV + CUDA graph on decode loop；scheduler 复用 minimind 的 pattern |

文本栈接 fork、视觉塔走 HF（one-shot encode 无 paged 收益）、connector 纯线性
投影——三者合并在 `SmolVLMModel` 里，submodule 命名镜像 HF checkpoint key
（`model.vision_model.*` / `model.connector.modality_projection.*` /
`model.language_model.*` / `lm_head.*`），fork loader 用默认 `prefix=""`
直接吃权重——**不需要** `/tmp` 切片 + `_weights.py` hack。

---

## 2. fork 已经就位的部分（不要重复造）

`docs/dev/nanovllm-omni-paged-engine.md` 的 ADR-007 / ADR-008 **已经落地**，
迁移前先确认这几处，别再改一遍：

| 能力 | 位置 | 状态 |
|---|---|---|
| model 注册表（不硬编码 Qwen3） | `third_party/nano-vllm/nanovllm/models/registry.py` | ✓ `register_model` / `get_model_class` / `MODEL_REGISTRY` |
| `ModelRunner` 接任意 model_class | `nanovllm/engine/model_runner.py:87,103` | ✓ `(model_class or get_model_class("qwen3"))(hf_config)` |
| `run_model` 支持 `inputs_embeds` | `nanovllm/engine/model_runner.py:258,269` | ✓ 非 None 时走 direct forward（不走 graph） |
| paged KV 分配可复用 | `nanovllm/engine/model_runner.py:allocate_paged_kv_cache` | ✓ 纯函数，外部可独立调用 |
| `Config.trust_remote_code` | fork `Config` | ✓ ADR-011 已落地 |
| `StageRunner` 适配层 | `nanovllm_omni/engine/stage_runner.py` | ✓ 288 行的纯通用 AR 生命周期；C6 commit `6fcc8c6` 已把 minimind vendor 专属拆走，C8 commit `2e15f5e` 改了 minimind bridge buffer；目前唯一使用者仍是 minimind_omni |

所以本次 fork 侧的改动只有一件事：**给 `load_model` 加 `prefix` kwarg**（详见 §3 ADR-016 改写）。

---

## 3. 决策（ADR）

### ADR-016：完整 SmolVLM 进 fork-style，submodule 命名镜像 HF checkpoint keys

**决策：** 在 `nanovllm_omni/models/smolvlm/smolvlm.py` 实现完整
`SmolVLMForConditionalGeneration`（不是仅文本栈）：

- `self.model.vision_model`：HF `SiglipVisionModel`（一次性 encode，无 paged 收益）
- `self.model.connector.modality_projection`：`nn.Sequential(GELU, Linear)` 把
  视觉 dim (768) 投影到文本 dim (576)
- `self.model.language_model`：`SmolLM2ForCausalLM`，**用 fork building blocks**
**（`QKVParallelLinear` / `Attention` / `RMSNorm` / `get_rope` /
  `VocabParallelEmbedding` / `ParallelLMHead`）**——拿到 paged KV + CUDA graph
- `self.lm_head`：fork `ParallelLMHead`

submodule 命名 (`model.vision_model.*` / `model.connector.*` /
`model.language_model.*` / `lm_head.*`) **镜像 HF checkpoint key**，fork
loader 用默认 `prefix=""` 直接吃权重——**不需要** `/tmp` 切片、不需要
`_weights.py` prefix 映射。

fork loader 的 `prefix` kwarg（`third_party/nano-vllm/nanovllm/utils/loader.py`
加了一行）仍然作为通用能力保留：未来其他多模态模型（例如 Qwen2-VL）
的 submodule 命名如果选择 `self.llm` / `self.vpm` 这种 short name
（vllm-omni 风格），那个模型会用到它。SmolVLM 自身没用上。

**理由：**
- **跨栈 dtype 转换负债消除**：原来"HF vision → fork LM"一次前向跨两个栈，
  要 `.to(device, dtype)` 对齐；现在 fork-style 单一栈，数值 gate §6.2 少
  一个排查方向。
- **submodule 命名镜像 HF 避免了 prefix 引入的混淆**：读到
  `self.model.vision_model = SiglipVisionModel(...)` 的同事不需要
  心算“`model.text_model.X` checkpoint key 会落在 `self.llm.X` 还是
  `self.model.language_model.X`”——HF key 命名是什么，submodule 就是什么。
- **不在 fork registry 注册 `SmolVLMForCausalLM`**：registry 是 decoder-only
  AR 的索引，多模态 shell 不属于。`StageRunner` 直接接
  `model_class=SmolVLMForConditionalGeneration` 就跑得起来。
- **SmolVLA 的复用路径**:SmolVLA 的 backbone 结构和完整 SmolVLM 不一定
  兼容（见 smolvla 指南 §3 ADR-029 deferred），但如果未来兼容成立，复用的
  是 `self.model.language_model` 这个 SmolLM2 子模块，不再需要额外的权重
  切片。

**代价：**
- 模型类体量：~340 行（Llama-style `SmolLM2Attention` / `SmolLM2MLP` /
  `SmolLM2DecoderLayer` / `SmolLM2Model` + `SmolVLMConnector` +
  `SmolVLMModel` + `SmolVLMForConditionalGeneration` shell）。
- 视觉塔仍走 HF（`transformers.models.siglip.modeling_siglip.SiglipVisionModel`）
  —— 是设计选择（one-shot encode 无 paged 收益），不是漏洞。

---

### ADR-017：vision embeds 走已有的 `inputs_embeds` 路径，不改 fork forward 签名

**决策：** nanovllm_omni 侧用 HF processor 出 `input_ids`，用 HF SigLIP 出
image embeds，在 **nanovllm_omni 侧**做 embedding merge（把 image token 位置
替换成 vision embeds），得到完整 `[T, H]`，再调
`stage_runner.forward(input_ids, positions, is_prefill, inputs_embeds=merged)`。

fork `smolvlm.py` 的 `forward` 签名照 ADR-008 已有约定：
`forward(input_ids, positions, inputs_embeds=None)`。**不新增 kwarg。**

**理由：**
- ADR-008 已经把这条路铺好了（`model_runner.py:269`），复用零成本。
- merge 逻辑是 SmolVLM 特有的（image token id、每图 patch 数），属于模型族知识，
  放 `models/smolvlm/` 比放 fork 合适。
- fork 保持"只认 `input_ids` 或 `inputs_embeds`"的窄接口，不长出多模态分支。

**代价：** prefill 走 `inputs_embeds` 就**不走 CUDA graph**（`model_runner.py:269`
的既有行为）。500M 模型 prefill 不是瓶颈，接受。decode 阶段仍走 `input_ids`
路径，graph 照常可用。

**ponytail 注记：** 如果将来 profile 显示 prefill 占比过高，再考虑 fork 侧
graph 支持 `inputs_embeds`；现在没有测量数据，不做。

---

### ADR-018：stage kind 从 `LLM_GENERATION` 改成 `LLM_AR`

**决策：** `pipeline.py:28` 改 `kind=StageExecutionType.LLM_AR`。

**理由：** `LLM_GENERATION` 当初是"最接近的现有成员"的凑数选择
（见 `pipeline.py:5` 注释）。迁移后它真的是 AR paged decode，和
minimind thinker 同类，`LLM_AR` 是准确描述。

**代价：** 无。`StageExecutionType` 是 `StrEnum` 闭集，`LLM_AR` 已存在，
`tests/test_config.py:268-271` 锁的四个名字不变。

---

### ADR-019：闭包 `_vlm_stage` 改成 `SmolVLMStage` 类，工厂签名不变

**决策：** `models/smolvlm/stage.py` 保留 `_vlm_stage(deploy, args)` 工厂函数
（`pipeline.py` 的 tuple factory 注册保持声明式），但它返回一个 `SmolVLMStage` 实例
而不是闭包。类结构照 `minimind_omni/_engine.py:ThinkerStage`：

- `__init__`：建 fork `Config`（`get_stage_config`）、建 `StageRunner`、
  加载 HF processor + SigLIP
- `__call__(payload, sampling)`：tokenize → vision encode → merge →
  prefill/decode 循环 → decode 出文本

**理由：**
- `PipelineRunner` 只要求 stage 是 callable，类实例满足；工厂路径字符串不变，
  registry 和 deploy yaml 都不用改。
- 模型加载（重）和单次推理（轻）分离，和 minimind 的分层一致。

**代价：** `stage.py` 从 ~180 行涨到 ~280 行（多了 decode 循环）。

---

### ADR-020（已被 C7 取消）：SmolVLM 单 stage，不接 `SharedBlockManager`

**已被 C7 commit `354e9b5` 取消**：C7 删除了 `SharedBlockManager` stub
（`nanovllm_omni/engine/block_manager.py` 或同等位置；C7 提交时一并删除）。
原因：stub 是 stored-but-never-read——fork `ModelRunner` 自管 `BlockManager`，
thinker/talker 的 KV 形状不兼容，物理上不能共享。`grep SharedBlockManager
nanovllm_omni/engine/stage_runner.py` 现在零命中。

**smolvlm stage 无需此 ADR 内的改动**：单 stage 复用 fork 自管的 block table，
与 minimind 同构（每个 stage 一个 `BlockManager` 实例，见 C7 commit message）。

原 ADR 提到的"`reset_shared_block_manager` / `reset_stage_configs` 两个
dead 钩子"也已由 C7 一并清掉。

---

## 4. 改动清单

### fork 文件改动（`third_party/nano-vllm/`）

| 文件 | 改动 | 量 |
|---|---|---|
| `nanovllm/utils/loader.py` | `load_model` 加 `prefix: str = ''` kwarg：`get_parameter(weight_name)` → `get_parameter(f"{prefix}{weight_name}")` | 1 行 |
| `nanovllm/engine/model_runner.py` | 无改动 | 0 |

fork 权重加载走 `nanovllm/utils/loader.py:load_model`。
SmolVLM checkpoint 的 key 前缀是：
- `model.vision_model.*` → `self.vpm.*`
- `model.text_model.*` → `self.llm.*`（用 `prefix="llm"` 调用 fork loader）
- `model.connector.*` → `self.resampler.*`（也用 `prefix`，名称约定按实现定）

完整模型类放 `nanovllm_omni/models/smolvlm/smolvlm.py`（fork 不需要单设
`nanovllm/models/smolvlm.py`——完整模型走 `init_vllm_registered_model` 拼接，
fork 只暴露通用 LLM 注册）。**`models/smolvlm/_weights.py` 切片方案被取消**：
不需要 thinker-only key 前缀过滤 + `/tmp` 缓存，prefix kwarg 就是分桶机制。

### nanovllm_omni 文件改动

| 文件 | 改动 |
|---|---|
| `models/smolvlm/smolvlm.py` | **新建**。完整 SmolVLM 类（`self.vpm` + `self.llm(prefix="llm")` + `self.resampler` + merge 在 `forward` 里）。照 vllm-omni `minicpmo_4_5_omni_llm.py` 模式。 |
| `models/smolvlm/stage.py` | 闭包 → `SmolVLMStage` 类（ADR-019）；加 prefill/decode 循环 |
| `models/smolvlm/pipeline.py` | `kind` → `LLM_AR`（ADR-018） |
| `deploy/smolvlm.yaml` | 加 `max_num_batched_tokens` / `max_num_seqs` / `gpu_memory_utilization`（照 minimind thinker 的 0.6） |

> 注：engine 侧 `stage_runner.py` 本次无改动——C7 commit `354e9b5` 已完成 ADR-020
> 原计划的所有 engine 侧修改（删 SharedBlockManager stub），smolvlm 现在直接复用
> fork 自管的 block table，无需新增任何 engine 接口。

### 测试改动

| 文件 | 内容 |
|---|---|
| `tests/test_smolvlm_fork_smoke.py` | **新建**。验证模型结构（submodule 命名、packed_modules_mapping、connector shape）+ pipeline kind=LLM_AR + stage 工厂导出。CPU-only，不拉 GPU。 |
| `tests/test_smolvlm.py` | 改：smolvlm stage kind 断言从 `LLM_GENERATION` 改 `LLM_AR` |

> 原计划的 `tests/test_smolvlm_weights.py`（权重切片）随 `models/smolvlm/_weights.py`
> 一起被 §3 ADR-016 改写取消：完整模型不需要 thinker-only key 过滤。

---

## 5. 实施顺序

**commit 1 已被 C7 commit `354e9b5` 吸收**（解开 SharedBlockManager + 删 dead 钩子
都已在 C7 落地）。本指南不再列 commit 1。

实际待实施的 commit（顺序可调）：

1. **commit 2 — fork loader 加 `prefix` kwarg**
   `third_party/nano-vllm/nanovllm/utils/loader.py` 一行改动。
   fork 侧已有 `prefix=''` 默认值；任何现有调用点（minimind_omni loader 路径）
   行为不变。
   测试：现有 minimind_omni 的端到端必须仍过（loader 是 minimind 的隐性依赖）。

2. **commit 3 — 完整 SmolVLM 模型类（vpm + llm + merge）**
   `nanovllm_omni/models/smolvlm/smolvlm.py` 新建，
   照 vllm-omni `minicpmo_4_5_omni_llm.py` 形态。
   测试：CPU meta device 构造 + `forward(input_ids)` 形状断言（不加载真权重）。

3. **commit 4 — `SmolVLMStage` 类 + decode 循环**
   `stage.py` 重写，`pipeline.py` 改 kind（ADR-018），deploy yaml 加资源项。
   `tests/test_smolvlm_stage.py` 用 mock runner 锁调用序列。

4. **commit 5 — GPU 端到端 + 数值对齐** (pending)
   WSL 上跑 `Omni.generate`，和迁移前的 HF `generate()` 输出对比。
   `scripts/pre-push`。**当前未跑** —— 本会话实现 fork smolvlm 后未启动 WSL
   bench。下一阶段由 owner 启动 WSL bench_smolvlm 与本文 §6.2 Gate (a/b/c)
   + §6.3-bis Profile gate。

> **commit 2/3 在 §6.3-bis Profile gate 通过之前不要启动**：若 HF bench 显示
> paged KV 收益不成立（500M 模型短序列），整套迁移降级为 doc-only。
> 当前 gate 未跑，subagent B 在 WSL profile。

---

## 6. 验证

### 6.1 CPU 侧（commit 1-3，`scripts/pre-commit`）

- `SmolVLMStage` 调用序列：mock `ModelRunner`，断言
  `set_context / run_model / reset_context` 被调用且顺序正确
- 权重切片：造一个假 safetensors（含 `model.text_model.*` +
  `model.vision_model.*`），断言输出只含文本 key
- 每个 stage 拿到独立 `SharedBlockManager` 实例（ADR-020 回归）

### 6.2 GPU 侧（commit 5，WSL RTX 3050）

greedy 数值对齐是唯一有意义的 gate：

- 同 prompt + 同图 + `temperature=0`（greedy），迁移前后生成的 token 序列
  **逐 token 一致**。不一致就是 attention / rope / merge 位置对错了。
- 显存峰值 ≤ 迁移前（paged KV 应该更省，但 SmolVLM 序列短，可能持平）
- 4 GB 卡上不 OOM，`gpu_memory_utilization` 调到跑通为止并写回 yaml

**Gate 顺序**（由低层到高层，每步必须独立通过才能进下一步）：
- (a) **prefill logits 一致**：单步 forward 后 logits 与
  `HF_model.forward(input_ids, pixel_values, ...).logits` 在 `atol=1e-5`
  内一致。无采样，定位 merge / RoPE / attention mask 错。
- (b) **decode 单步 logits 一致**：与
  `HF_model.forward(input_ids[:, -1:], ...).logits` 一致（用同样的
  KV/past 状态）。定位 paged KV / position embedding 错。
- (c) **完整 token 序列对齐（greedy）**：多步 decode，`temperature=0` 下
  迁移前后生成的 token 序列逐 token 一致。定位 sampler / top_k 错。

### 6.3 明确不作为 gate 的

**不要求速度更快。** 500M 模型、短序列、单请求，paged attention 的收益
主要在长上下文和多并发；本次迁移的目标是**架构统一**（fork AR 生命周期
能接所有多模态模型），不是性能。文档和 commit message 里都要这么写，
避免后来人误以为有性能承诺。

### 6.3-bis Profile gate（必须先于 §5 commit 2 启动）

**这是迁移决策本身的闸门**，不是 commit 验证：

WSL 上跑 smolvlm HF 现状 bench，输出三项数字：
- prefill ms（单请求）
- decode ms / token（单请求）
- 显存峰值

判定规则：
- 若 `prefill_ms ≤ decode_ms × N`（N = 目标 token 数），且
- 峰值显存 `≥ 3.5 GB`（接近 4 GB 上限），

则 **fork-AR 化的性能动机不成立**（500M 模型 paged KV 收益 < 测量误差），
整个迁移指南降级为 "doc-only，记录 vllm-omni 模式参考"，**不动 fork 代码**。

**当前 gate 未通过**：smolvlm HF bench 数字未采集，迁移冻结。
Subagent B 正在 WSL 跑。

**Gate 失败时本文的命运**：
- §3 ADR-016 改写（完整模型进 fork）作为 vllm-omni 模式参考保留
- §4 / §5 的 commit 序列不启动
- §6.2 GPU gate 自然也无从跑
- §10 加一行 "doc-only，profile gate 失败"

---

## 7. 为什么 SmolVLM 排第一

- **fork 改动最小**：只需给 loader 加一个 `prefix` kwarg（一行）
- **纯 AR**，没有 diffusion / flow matching 的第二种执行模式要设计
- **解锁后两份**：SmolVLA 的 AR 半边可能复用这里的 fork 移植（`self.llm = init_vllm_registered_model(prefix="llm")`
  的产物），但**这一复用路径受 ADR-029 deferred 影响**（见 smolvla 指南 §3）
- **C7 commit `354e9b5` 已删 `SharedBlockManager` stub**，StageRunner 现在是
  纯通用 AR 生命周期（`set_context` / `forward` / `sample` / `reset_context`，
  `engine/stage_runner.py` 288 行）。C6 commit `6fcc8c6` 已把 minimind vendor
  专属 decode 拆走。smolvlm 迁移不再为"第二个消费者"叙事服务——它的价值
  改为 **fork loader prefix 化的实证**（若 §6.3-bis 通过），以及
  **§3 ADR-016 改写后的 vllm-omni 模式一致性**。

---

## 8. 跳过清单

以下**不做**，出现真实需求再说：

| 跳过 | 理由 |
|---|---|
| vision tower 进 fork | 无 KV cache，无 paged 收益（ADR-016） |
| `inputs_embeds` 路径的 CUDA graph | 没有 profile 数据支撑（ADR-017） |
| 多图 batch | 当前 `extra["images"]` 是单请求 list，`max_num_seqs=1` |
| 连续 batching / 多请求调度 | 单 GPU 单请求，`PipelineExecutor` 的 semaphore 就够 |
| TP > 1 | 4 GB 单卡 |
| SmolVLM 2.x / 更大 variant | 500M 是 3050 上的选型结论 |

---

## 9. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| vision embeds merge 位置对不上（image token id / patch 数算错） | 高 | greedy 逐 token 对齐（§6.2）能立刻抓到；先用 1 图最小 case 调通 |
| fork `Attention` 的 head_dim 非 pow2（`store_kvcache_kernel`） | 中 | ADR-010 已修 kernel；若仍报错先查 `D_PAD` 传参 |
| 权重 key 前缀和实际 checkpoint 不符 | 中 | commit 2 的切片测试先用真 checkpoint 的 key 清单跑一遍 |
| `SharedBlockManager` 解单例破坏 minimind | 低 | commit 1 独立提交，WSL 端到端验证后再往下 |

**回退：** 每个 commit 独立可 revert。commit 3 之前 SmolVLM 仍走原闭包路径；
commit 3/4 出问题就 revert 这两个，fork 侧的 `models/smolvlm.py` 留着不影响任何人
（registry 里多一个没人用的条目，无副作用）。

---

## 10. 2026-10-20：Colab T4 上的 decode 优化

迁移之后 SmolVLM 的 decode 一直跑在 eager 模式（当时 RTX 3050 上 capture 失败）。
2026-10-20 在 Colab T4 上做了 profile，两处改动：

| 改动 | 位置 | 说明 |
|---|---|---|
| decode 模型用请求的 dtype | `engine/stage_runner.py` | fork 的 `Config` 没有 dtype 字段，它按 checkpoint 的 dtype 建模型，于是 stage 自己那份权重是 bf16、真正跑 decode 的那份是 fp32。现在 stage config mapper 把 dtype 转发过去 |
| decode 循环跑在 CUDA graph 上 | `models/smolvlm/stage.py` | fork 侧加了 `Config.graph_max_seq_len`（capture 的 KV 窗口）、context 超窗回落 eager、capture 失败回落 eager；stage 默认开 graph，`extra["use_cuda_graph"] = False` 可关 |

Colab T4 实测（5 次取中位数，同一进程内对比）：

| cell | wall p50 |
|---|---:|
| eager | 684.7 ms |
| CUDA graph | 181.6 ms |

eager 与 graph 生成的文本一致。详细 profile 见
`docs/perf/smolvlm-2026-10-20.md`。

**RTX 3050 的数字没有变化**：capture 在 3050 上仍会失败，fork 现在捕获这个失败
并回落 eager，所以行为与之前一致，只是多一条 warning。

---

## 11. 关联文档

- `docs/dev/nanovllm-omni-paged-engine.md` — ADR-007/008/010/011 的来源，本文的前置
- `docs/dev/nanovllm-omni-sdturbo-diffusion-migration.md` — 第二份，DiffusionEngine
- `docs/dev/nanovllm-omni-smolvla-arflow-migration.md` — 第三份，依赖本文 + 上一份
- `docs/perf/smolvlm-2026-10-20.md` — Colab T4 上的 profile 与优化结果
- C6 commit `6fcc8c6` / C7 commit `354e9b5` / C8 commit `2e15f5e` —
  C7 取消 ADR-020（`SharedBlockManager` 解单例 + 删 dead 钩子）；
  C6 + C8 改 minimind 内部实现，smolvlm fork-AR 化不受影响
- `.agents/skills/add-new-model/SKILL.md` — 新模型族接线流程
