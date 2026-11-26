# fork paged Engine 完成 + MiniMind-O talker 接入

> 状态:**Draft** — grilling 收齐,等 owner 拍板后执行
> 日期:2025-01-XX
> 关联:`docs/dev/nanovllm-omni-rewrite.md`(ADR-002 共享 BlockManager 设计)
> 决策来源:本仓库与用户的 grill round 1-3

---

## 1. 背景

MiniMind-O 三阶段 thinker → talker → code2wav。当前代码已经接 fork `ModelRunner`(thinker),talker 自己跑 eager full-sequence。grilling 中发现两个 latent 缺陷 + 三个架构缺口:

### 1.1 Latent 缺陷

**(L1) Thinker bridge hidden 在 CUDA graph replay 路径下 stale**

`thinker.py:223` 当前用 Python 属性赋值记录 bridge hidden:
```python
def forward(self, input_ids, positions):
    ...
    for i, layer in enumerate(self.layers):
        hidden_states, residual = layer(positions, hidden_states, residual)
        if i == self.bridge_layer:
            self._bridge_hidden = hidden_states   # ← Python 属性赋值,不在 CUDA graph 里
    ...
    return hidden_states
```

fork `ModelRunner.run_model`(third_party/nano-vllm/nanovllm/engine/model_runner.py:206-222)在 `is_prefill=False, enforce_eager=False, N≤512` 时走 CUDA graph replay。graph **只 capture CUDA op**(`aten::copy_`、`aten::addmm`、`flash_attn_*` 等),**不 capture Python 属性赋值**。

结果:decode 步的 `self._bridge_hidden` 停留在 prefill 时的值,所有 decode 步拿到的 bridge hidden 都是 prefill 步的同一个 tensor。`ThinkerStage.__call__` 把 stale bridge hidden 拼到 bridge 序列里,talker 拿到的「thinker 中间层状态」恒定不变。

当前为什么没炸:
- `OmniEngineArgs.enforce_eager` 已被移除(`config/params.py:69-71`),fork `Config.enforce_eager` 默认 False
- 各 stage 在 `__init__` 里直接设 `cfg.enforce_eager = True`（thinker 因 RTX 3050 上 `capture_cudagraph()` 挂掉；smolvlm 同样）；见 ADR-0005
- `max_num_seqs=1`(`deploy/minimind_omni.yaml:19`),decode 必走 graph replay
- **也就是说这个 bug 当前默认就在踩,只是没人去验证 bridge 序列的正确性**

**(L2) Talker attention 走 stdlib SDPA,不进 paged attention**

`talker.py` 用了 `_EagerAttention`(stdlib `F.scaled_dot_product_attention`),绕开 fork `Attention` 的 paged 路径。原因:

- fork `store_kvcache_kernel` 用 `tl.arange(0, D)`,要求 `D = num_kv_heads * head_dim = 2^n`;MiniMind talker D = 2 × 96 = **192**,不 power of 2,kernel 编译失败
- fork `Attention.forward` 强依赖 `get_context()` 全局状态(talker 不构造 Context)
- fork `Attention` 走 `flash_attn`(CUDA 依赖),CPU/MPS host 上 import 失败

后果:talker 不能用 paged KV cache,每步重算整段序列,O(T²) 总成本。

### 1.2 架构缺口

**(A1) fork ModelRunner 硬编码 Qwen3**

`third_party/nano-vllm/nanovllm/engine/model_runner.py:8`:
```python
from nanovllm.models.qwen3 import Qwen3ForCausalLM
```

不能干净地接 `MiniMindThinker` / `MiniMindTalker`。

**(A2) fork ModelRunner.run_model 只接 input_ids**

不支持 `inputs_embeds` 路径,所以 talker 这种「已经预 embed 的输入」不能走 `ModelRunner.run_model`。

**(A3) fork Config 没有 trust_remote_code**

MiniMind 用 `trust_remote_code=True` 加载 HF config,当前 `get_stage_config` monkey-patch `AutoConfig.from_pretrained`(`_engine.py` 现有)。

### 1.3 已存在的脚手架可复用

- ADR-002(`nanovllm-omni-rewrite.md`):Shared BlockManager(逻辑 block_id 共享,KV tensor 各 stage 独立)
- `StageRunner`(`_engine.py`):`set_context / forward / sample / reset_context` 多方法 API,贴合 fork 生命周期
- `_ensure_dist`(`_engine.py`):monkey-patch `dist.init_process_group` 兼容两 stage 共享进程

---

## 2. 目标

1. **修复 L1**(thinker bridge hidden graph stale):用 registered buffer 让 bridge hidden 走 CUDA op 路径进 graph
2. **修复 L2**(talker 不进 paged):扩展 fork `store_kvcache_kernel` 支持非 pow2 D,让 talker 走 fork `Attention`
3. **填补 A1-A3**:把 fork `ModelRunner` 改成通用 AR Engine,支持任意 model_class + inputs_embeds 路径 + trust_remote_code kwarg
4. **Talker 增量解码**:对齐 vllm-omni 行为,O(T) 总成本而不是 O(T²)
5. **清理脚手架**:把 `_stage.py` 和 `_engine.py` 通用件提到 `nanovllm_omni/utils/profiling.py` 和 `nanovllm_omni/engine/stage_runner.py`
6. **删 dead code**:`_sampling.py` + `tests/test_sampling_helpers.py`

---

## 3. 决策(ADR)

### ADR-004:用 registered buffer 把 thinker bridge hidden 写进 CUDA graph

**决策**:`MiniMindThinker.__init__` 加 `self.register_buffer("_bridge_buffer", torch.empty(0))`。forward 在 `bridge_layer` 处用 CUDA slice assign 写 buffer:
```python
self._bridge_buffer[: hidden_states.shape[0]] = hidden_states
```
`get_bridge_hidden()` 改读 `self._bridge_buffer`。

**理由**:
- **不改 fork**(`model_runner.py` 一行不动)
- 改 `thinker.py` ~5 行
- CUDA slice assign(`aten::copy_`)进 graph,replay 后 buffer 反映**当前 step**的真实 bridge hidden
- `register_buffer` 让 buffer 跟随模型 `.cuda()` / `.to(device)` 一起搬,无需手动管 device

**代价**:每次 forward 多一次 `aten::copy_` 到预分配 buffer(bridge_layer 的 hidden_states 复制一份)。开销 = `num_tokens × hidden_size × dtype.itemsize` 的 memcpy,可忽略。

**对比方案**:

| | 改 fork | 改 thinker | graph 加速保留 | 复杂度 |
|---|---|---|---|---|
| thinker 走 direct forward(`enforce_eager=True`) | 0 | 1 行 | ✗ | 1 |
| bridge 进 graph_vars | ~10 行 | ~3 行 | ✓ | 3 |
| **registered buffer side-output(本方案)** | 0 | ~5 行 | ✓ | 1 |

### ADR-005:talker 走 incremental decode(Q1)

**决策**:talker 第一步 prefill 全序列,后续每步只算新增 1 个 audio frame 的 hidden,从 paged KV cache 取历史。

**理由**:
- 对齐 vllm-omni 行为(`vllm_omni/model_executor/models/minimind_o/minimind_omni_talker.py:271` 注释: "Decode input at position num_computed should be conditioned on the matching thinker bridge row, not on bridge[-1] for every step.")
- O(T) 总成本 vs 当前 O(T²)
- KV cache 真的有用,不是写完就丢

**代价**:
- `MiniMindTalker.forward` 拆成 `prepare_inputs_embeds(bridge, audio_codes, spk_emb)` + `body(positions, hidden_states)`(仿 vllm-omni `preprocess` / `forward`)
- `MiniMindTalker.body(positions, hidden_states)` 接收**预 embed**的 hidden(只跑 layers + norm + lm_head),不重复 codec_proj / embed_proj
- talker stage 的 decode loop 从「每步重建完整 hidden_states」改为「每步增量 1 token」

### ADR-006:talker 不跑 CUDA graph(Q7)

**决策**:talker 的 `Config` 强制 `enforce_eager=True`,不走 graph capture 路径。

**理由**:
- talker forward 含 staggered decode(8 个 codebook 各自采样)+ top-50 + repetition penalty,全是 Python 流
- 即使 attention 部分纯 CUDA,sampling 侧仍需 Python 控制流,graph 兼容性差
- talker 单请求流式生成,CUDA graph 的批大小复用收益几乎为 0

**代价**:talker decode 失去 graph 加速。thinker 仍走 graph(verify 过)。

### ADR-007:fork ModelRunner 改成通用 AR Engine,不硬编码 Qwen3(Q2)

**决策**:
- fork `model_runner.py` 顶部删除 `from nanovllm.models.qwen3 import Qwen3ForCausalLM`
- fork 加 `nanovllm/models/registry.py` 一个 dict + `@register_model("qwen3")` 装饰器,Qwen3 在自己的模块里 register
- ModelRunner 从 registry 取 default;用户传 `model_class=...` 仍优先

**理由**:
- 保持向后兼容(纯文本 Qwen3 路径不破,registry 默认 `"qwen3"`)
- 给 MiniMind-O 留 `model_class=MiniMindThinker` / `MiniMindTalker` 入口
- 符合「第二个真实使用者出现时才抽象」(MiniMind-O 是第二个)

**代价**:fork 加 `registry.py`(~20 行)+ Qwen3 模块顶部一行装饰器。

### ADR-008:fork ModelRunner.run_model 支持 inputs_embeds

**决策**:`ModelRunner.run_model(input_ids, positions, is_prefill, inputs_embeds=None)`。当 `inputs_embeds is not None` 时强制走 direct forward 路径(不走 graph),直接调 `self.model(input_ids, positions, inputs_embeds=...)`。

**理由**:talker 需要把预 embed 的 hidden 喂给 model,不走 `embed_tokens`。

**代价**:
- fork `Qwen3ForCausalLM.forward` / `Qwen3Model.forward` 加 `inputs_embeds=None` kwarg(~10 行)
- inputs_embeds 路径不走 graph(因为 graph capture 时固定 `input_ids` 路径)

### ADR-009:fork Sampler 扩展 top_k / repetition_penalty

**决策**:fork `Sampler.forward(logits, temperatures, top_k=0, history=None, repetition_penalty=1.0)`。默认参数走原 Gumbel-max 路径(无功能变化),新 kwargs 启用时本地做 rep penalty + top-k mask,再调 `@torch.compile _gumbel_max_sample`。

**理由**:
- 默认行为不变(fork 既有用户零影响)
- thinker / talker 都可接 vendor 的 top-k + rep penalty recipe
- `@torch.compile` 只裹 Gumbel-max 尾,top-k mask 在外面做(避开 shape 错位)

**代价**:fork `sampler.py` 拆 `_gumbel_max_sample(logits)` 函数 + `Sampler.forward` 加 3 个 kwargs。

### ADR-010:fork store_kvcache_kernel 支持非 pow2 D

**决策**:kernel 加 `D_PAD: tl.constexpr = next_pow2(D)` 参数,`tl.arange(0, D_PAD)` + `mask = arange < D`。`store_kvcache` Python wrapper 算 `D_PAD` 传入。

**理由**:
- MiniMind talker D = 192,必须修才能用 paged attention
- 不浪费 KV cache 内存(分配时按 D 步长,不是 D_PAD 步长)
- 任何非 pow2 D 的模型都能跑(目前 MiniMind talker 是首个,但未来防御性)

**代价**:fork `attention.py` ~10 行改动。

### ADR-011:fork Config 加 trust_remote_code

**决策**:`Config.trust_remote_code: bool = False` 字段,`__post_init__` 透传给 `AutoConfig.from_pretrained`。

**理由**:替代 `_engine.py::get_stage_config` 里的 `AutoConfig.from_pretrained` monkey-patch(去掉 ~10 行 hack)。

**代价**:fork `config.py` ~3 行改动。

### ADR-012:talker 的 Sequence 用 frame count 当 token_ids(Q10)

**决策**:talker 每步产 1 个 audio frame,`token_ids = [0, 1, 2, ...]`,每步 `seq.append_token(step_idx)`。BlockManager 只看长度不看内容。

**理由**:
- fork `prepare_prefill` 会构造一个 `input_ids` tensor(全是 frame index),talker 丢掉不用,只用 `positions/slot_mapping/block_tables`
- 不需要改 fork `Sequence`
- 内存浪费可忽略(`max_num_seqs=1`,几十 KB)

### ADR-013:`_stage.py` 提到 `nanovllm_omni/utils/profiling.py`,改名为 `profile_range`(1A)

**决策**:`_stage.py` 删,新建 `nanovllm_omni/utils/profiling.py`,导出 `profile_range(name)` 上下文管理器(`code2wav.py` 改 import)。

**理由**:
- `_stage.py` 全文 50 行完全通用,只是包了 `torch.profiler.record_function` + `torch.cuda.nvtx.range`
- 命名冲突:`stage` 在本仓既指 pipeline 阶段又指 profiler 区间,易读乱
- 改名 `profile_range` 避免歧义

### ADR-014:`_engine.py` 通用件提到 `nanovllm_omni/engine/stage_runner.py`(2A)

**决策**:新建 `nanovllm_omni/engine/stage_runner.py`,放通用件:
- `SharedBlockManager` / `get_shared_block_manager` / `reset_shared_block_manager`
- `StageRunner`
- `_ensure_dist`(fork NCCL init 兼容)
- `allocate_paged_kv_cache`(从 fork `ModelRunner.allocate_kv_cache` 抽出,module-level)
- `get_stage_config` / `_stage_kwargs_from_args`

`_engine.py` 留 MiniMind-O 专属:`ThinkerStage` / `TalkerStage` / `TalkerOutput` / `_THINKER_KEEP_PREFIX` / `_TALKER_KEEP_PREFIX` / `_talker_only_dir` / `_thinker_only_dir` / `_filter_only_dir` / `_resolve_model_path`。

**理由**:通用件只为 minimind_omni 一家用,但 ADR-002 的 SharedBlockManager + 多方法 StageRunner 是「通用 AR Engine 适配层」的雏形,提到 engine/ 给未来第二个族用。

### ADR-015:`_sampling.py` + `tests/test_sampling_helpers.py` 删除(Q12)

**决策**:`_sampling.py` 整文件删,`tests/test_sampling_helpers.py`(16 个 match)一起删。

**理由**:
- refactor 后 `StageRunner.sample(logits, temperatures, top_k=50, history=audio_codes[i], repetition_penalty=1.05)` 接管全部采样
- `_sampling.py` 现在的两个函数 refactor 后再无 caller(原 talker inline 改用 `StageRunner.sample`,thinker 走 fork `Sampler`)
- `tests/test_sampling_helpers.py` 测的是已经死掉的代码

---

## 4. 改动清单

### Fork 文件改动

| 文件 | 改动 | ADR |
|---|---|---|
| `third_party/nano-vllm/nanovllm/config.py` | 加 `trust_remote_code: bool = False` 字段,`__post_init__` 透传 | ADR-011 |
| `third_party/nano-vllm/nanovllm/engine/model_runner.py` | 删顶部 Qwen3 import;` `run_model` 加 `inputs_embeds=None` kwarg;抽 `allocate_kv_cache` 为 module-level function | ADR-007, ADR-008, ADR-003 iii.a |
| `third_party/nano-vllm/nanovllm/layers/attention.py` | `store_kvcache_kernel` 加 `D_PAD` constexpr + mask;`store_kvcache` 算 `D_PAD` 传入 | ADR-010 |
| `third_party/nano-vllm/nanovllm/layers/sampler.py` | 抽 `_gumbel_max_sample(logits)` `@torch.compile` 函数;`Sampler.forward` 加 `top_k` / `history` / `repetition_penalty` kwargs | ADR-009 |
| `third_party/nano-vllm/nanovllm/models/qwen3.py` | `Qwen3Model.forward` + `Qwen3ForCausalLM.forward` 加 `inputs_embeds=None` kwarg | ADR-008 |
| `third_party/nano-vllm/nanovllm/models/registry.py` (新建) | `MODEL_REGISTRY` dict + `@register_model("qwen3")` 装饰器;ModelRunner 通过 registry 取 default | ADR-007 |

### MiniMind-O 文件改动

| 文件 | 改动 | ADR |
|---|---|---|
| `nanovllm_omni/models/minimind_omni/talker.py` | 删 `_EagerAttention`(用 fork `Attention`);`TalkerAttention` 改用 fork `Attention`;`MiniMindTalker` 拆 `prepare_inputs_embeds` + `body` + `compute_logits`;`forward` 兼容 `inputs_embeds=None` kwarg | ADR-005, ADR-008 |
| `nanovllm_omni/models/minimind_omni/thinker.py` | `register_buffer("_bridge_buffer")`;forward 里 CUDA slice assign 写 buffer;`get_bridge_hidden` 改读 buffer | ADR-004 |
| `nanovllm_omni/models/minimind_omni/_engine.py` | 删 `_patch_store_kvcache`(ADR-010 已修);`get_stage_config` 用 `Config(trust_remote_code=True)`(ADR-011);`TalkerStage` 改用 incremental decode(ADR-005)+ `StageRunner.sample` 接 fork Sampler(ADR-009);`TalkerStage` 强制 `enforce_eager=True`(ADR-006) | 多个 |
| `nanovllm_omni/models/minimind_omni/_sampling.py` | **整文件删** | ADR-015 |
| `nanovllm_omni/models/minimind_omni/_stage.py` | **整文件删** | ADR-013 |
| `nanovllm_omni/models/minimind_omni/bundle.py` | 不动 | — |
| `nanovllm_omni/models/minimind_omni/code2wav.py` | `from ._stage import stage` → `from nanovllm_omni.utils.profiling import profile_range`;`with stage("...")` → `with profile_range("...")` | ADR-013 |

### 新建文件

| 文件 | 内容 | ADR |
|---|---|---|
| `nanovllm_omni/utils/profiling.py` | `profile_range(name)` 上下文管理器,从原 `_stage.py` 搬过来改名 | ADR-013 |
| `nanovllm_omni/engine/stage_runner.py` | `SharedBlockManager` / `StageRunner` / `_ensure_dist` / `allocate_paged_kv_cache` 等通用件,从 `_engine.py` 搬过来 | ADR-014 |

### 测试改动

| 文件 | 改动 | ADR |
|---|---|---|
| `tests/test_sampling_helpers.py` | **整文件删**(测的是 dead code) | ADR-015 |
| `examples/offline_inference/minimind_o/batched.py` | line 32: `from ...bundle import create_bundle` → `from ...bundle import load_minimind_omni_bundle`;line 59: `bundle = create_bundle(...)` → `bundle = load_minimind_omni_bundle(...)` | 之前已记录的 bug fix |
| 新增 | thinker bridge 集成测试:prefill + 5 步 decode,断言 `get_bridge_hidden()` 每步拿到的值不同 | ADR-004 |
| 新增 | talker incremental decode 测试:full-seq vs incremental 输出 bit-equal | ADR-005 |

---

## 5. 验证

按 `CONTRIBUTING.md` 流程:

### 5.1 提交前
- `scripts/pre-commit`:lint + format + import 检查
- `pytest tests/ -x`:全测试套件通过
- 重点跑 `tests/test_pipeline_runner.py` / `tests/test_config.py` / `tests/test_bundle_resolve_snapshot.py`

### 5.2 WSL GPU 端到端
- `python -m tools.bench_decode_loop --family minimind_omni --max-tokens 512`
- 验证 thinker + talker 跑通,bridge hidden 每步正确
- 验证 talker 生成的 audio_codes 与 pre-refactor 一致(vendor 对齐)
- VRAM 不超过 `deploy/minimind_omni.yaml` 标的 4GB / 0.6+0.2+0.1 = 0.9 × 4GB ≈ 3.6GB

### 5.3 关键 gate
- [ ] thinker `get_bridge_hidden()` 在 graph replay 路径下每步返回不同值(ADR-004 验证)
- [ ] talker incremental decode 输出与 full-seq 输出 bit-equal(ADR-005 验证)
- [ ] talker attention 走 fork `Attention`,attention 输出与原 `_EagerAttention` 在 num_tokens=1 时一致(ADR-010 验证)
- [ ] fork `Sampler` top_k=0 路径与 refactor 前输出 bit-exact(ADR-009 验证)
- [ ] 所有 fork 改动在 default config 下对 fork-internal Qwen3 用户零影响(ADR-007/008/009/010/011 兼容性)

---

## 6. 风险与已知 caveat

### 6.1 CPU/MPS host 兼容性
fork `model_runner.py` 顶部 `import torch.distributed as dist` 是 module-level,CPU host 上 NCCL 不可用。当前 `_ensure_dist` 补丁只在 stage factory 调用时触发,但 `_stage_kwargs_from_args` 返回 dict 不会触发 init。

dev 在 macOS 跑测试时,如果测试不触发 `StageRunner.__init__`,OK;一旦触发就崩。**当前状态就是这样,refactor 不恶化**。

### 6.2 已知不修的 caveat
- `_sampling.py` 的 recipe 常量(`DEFAULT_TEXT_TOP_P=0.90` 等)随文件删除而消失。如果未来需要 vendor recipe 调试,可以从 git 恢复
- 旧 docstring 说「talker eager full-sequence 是 vendor 行为」,refactor 后改为 incremental decode,这是行为微调(对齐 vllm-omni)。任何依赖原行为的下游代码需要 review

### 6.3 ADR-005(增量解码)的语义保证
vllm-omni 的 talker 用 `num_computed` 索引 bridge:`bridge[num_computed:num_computed+1]`(decode 步)而不是 `bridge[-1]`(每步用整段 bridge)。我们 refactor 后也要遵循同一约定 —— 这是 vllm-omni 改的「语义微调」,不是我们改的。

如果 vendor 原 MiniMindOmni.stream_generate 是 full-seq 行为,我们的 incremental 版本在**生成 audio 的精确 token 序列**上和 vendor 可能有 diff(同分布采样 + temperature 0.2 下听感无差别,但 bit-exact 测试会失败)。**这是合规的对齐**,因为 AGENTS.md 说「对齐指行为兼容」,vllm-omni 的 incremental decode 是「行为兼容」的最高优先级参考。

---

## 7. 实施顺序

按依赖关系,严格按此顺序提交,每步跑 `scripts/pre-commit`:

1. **Fork 改动**(可一次性提交):
   - ADR-010 (`attention.py`)
   - ADR-009 (`sampler.py`)
   - ADR-011 (`config.py`)
   - ADR-007 (`registry.py` 新建 + `model_runner.py` 删 import)
   - ADR-008 (`qwen3.py` + `model_runner.py` run_model 加 inputs_embeds)
   - ADR-003 iii.a (`model_runner.py` 抽 allocate_kv_cache)
2. **MiniMind-O 改动**:
   - ADR-013 (新建 `utils/profiling.py`,`code2wav.py` 改 import,删 `_stage.py`)
   - ADR-014 (新建 `engine/stage_runner.py`,`_engine.py` 改 import)
   - ADR-004 (`thinker.py` bridge buffer)
   - ADR-005 (`talker.py` 拆 prepare/body/compute_logits)
   - ADR-006 (`_engine.py::TalkerStage` enforce_eager=True)
   - ADR-015 (删 `_sampling.py` + 测试)
3. **example fix**:`examples/offline_inference/minimind_o/batched.py` 改 import
4. **新测试**:thinker bridge 集成测试 + talker incremental bit-equal 测试
5. **WSL GPU 端到端**:跑通 + 验证 gate

---

## 8. 待 owner 拍板

| 项 | 默认 |
|---|---|
| 增量解码 vllm-omni vs vendor 原版 的 bit-equal 兼容范围 | **接受 vllm-omni 行为**(incremental),vendor 原版的 bit-equal 不作为 gate |
| `gpu_memory_utilization` 在 talker + incremental decode 后是否需要重新调参 | **先用 deploy YAML 现值**(0.6+0.2+0.1),跑不通再调 |
| ADR-014 把通用件提到 `engine/` 后,sd_t / smolvlm / smolvla 是否要 follow | **不强行 follow**;等第二个用 fork 的族出现再统一 |
| `_sampling.py` recipe 常量是否保留 | **不保留**(整文件删);有需要从 git 找 |

---

## 9. 2026-10-20:MiniMind-O 不再执行 hub 远端代码

**背景**:MiniMind-3o 的 `config.json` 带 `auto_map`,指向 hub 上的 `model_omni.OmniConfig`。ADR-011 加的 `Config.trust_remote_code` 让 fork 能加载它,代价是每次加载都执行 hub 上的 Python 文件。本仓库自己实现了 thinker 和 talker,hub 的远端代码只贡献了一个 config 类。

**决策**:仓库自己注册 `minimind-o` 的 config 类(`nanovllm_omni/models/minimind_omni/hf_config.py`),并让 thinker / talker 用 `trust_remote_code=False` 建 fork `Config`。

**理由**:transformers 在 `model_type` 已注册、且注册类不在 `transformers.*` 命名空间下时走 `explicit_local_code` 分支,直接用注册类,`auto_map` 文件不会被执行。注册类只有一个字段需要自己给默认值(`tie_word_embeddings=True`,vendor 类的默认值,checkpoint 的 `config.json` 里没有这个键)。

**改动**:

| 文件 | 改动 |
|---|---|
| `nanovllm_omni/models/minimind_omni/hf_config.py` | 新建:`MiniMindOConfig` + `register_minimind_config()` |
| `nanovllm_omni/models/minimind_omni/stage.py` | thinker / talker 注册 config 类并用 `trust_remote_code=False`;tokenizer 也去掉 `trust_remote_code` |
| `tests/test_minimind_config_registration.py` | 新建:锁定"hub 文件不执行"这条性质 |

**代价**:多一个 config 类要跟着 checkpoint 的字段走;hub 的 `auto_map` 如果以后指向非 config 类,注册类不再覆盖它。

**未完成**:MiniMind-O 在 Colab T4 上的测量仍缺一块 GPU,见 `docs/perf/minimind-omni-2026-10-20.md`。

---

## 10. 关联文档

- `docs/dev/nanovllm-omni-rewrite.md` — ADR-002 SharedBlockManager 本设计的源头
- `docs/dev/vllm-omni-diffusion-reference.md` — vllm-omni diffusion 风格参考(本文档的对齐对象是 vllm-omni minimind_o)
- vLLM-Omni PR #3796 — MiniMind-Omni 集成的上游 PR,本次 refactor 的关键参考