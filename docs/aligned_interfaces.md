# 对齐接口

`nanovllm-omni` 的公共 API 在合理范围内对齐 vllm-omni 的对应符号,目标是"消费方写一份代码能在两边跑"——不是把 vllm-omni 的全部实现搬过来。本文档是**对齐契约的单一来源**:任何对公共类、字段、HTTP 响应形状的修改,都要回来更新这里的对应位置,并加一条聚焦的契约测试(详见末尾的"对齐符号变更流程")。

| nanovllm_omni 类 | vllm-omni 对应 |
|---|---|
| `nanovllm_omni.entrypoints.Omni` | `vllm_omni.entrypoints.Omni` |
| `nanovllm_omni.entrypoints.AsyncOmni` | `vllm_omni.entrypoints.AsyncOmni` |
| `nanovllm_omni.config.params.SamplingParams` | `vllm_omni.engine_args.SamplingParams` |
| `nanovllm_omni.config.params.OmniEngineArgs` | `vllm_omni.engine_args.OmniEngineArgs` |
| `nanovllm_omni.outputs.OmniRequestOutput` | `vllm_omni.outputs.RequestOutput` |
| `nanovllm_omni.config.registry.PipelineConfig` | `vllm_omni.config.PipelineConfig` |
| `nanovllm_omni.config.registry.StageConfig` (新增) | `vllm_omni.config.StagePipelineConfig` |
| `nanovllm_omni.config.registry.DeployConfig` | `vllm_omni.config.DeployConfig` |
| `nanovllm_omni.config.registry.DeployStageConfig` | `vllm_omni.config.StageDeployConfig` |
| `nanovllm_omni.config.registry.StageExecutionType` (TK-016 phase 2) | `vllm_omni.config.StageExecutionType` |
| `nanovllm_omni.config.registry.resolve_stage_factory` (TK-016 phase 2) | `vllm_omni.config.resolve_stage_factory` |
| `nanovllm_omni.engine.runner.PipelineRunner` (新增) | 无对应(单 GPU 单副本下替代 `StagePool`) |
| `nanovllm_omni.engine.executor.PipelineExecutor` (新增) | 无对应(单进程下替代 `Orchestrator`) |

## 对齐边界

这是**消费方可见的对齐**,不是逐行实现。当前垂直支持的模型族是 MiniMind-O 音频 + Wan2.2 TI2V 扩散(后者由 TICKET-06 gate)。我们**有意省略** vllm-omni 的以下能力:

- **多副本路由、负载均衡、副本指标、动态成员管理**(`vllm_omni.engine.stage_pool` 1281 行)。单进程单副本用 `PipelineRunner` 即可。
- **跨 stage 请求生命周期、CFG 伴随派发、双工会话跟踪、错误恢复、死副本检测**(`vllm_omni.engine.orchestrator` 2428 行)。`PipelineExecutor` 的 `asyncio.Semaphore` + `ThreadPoolExecutor` 就够了。
- **分布式执行、KV 传输、Mooncake 连接器、Ray 后端**。`AGENTS.md` 锁定了"本地单进程执行"。
- **全双工 S2S API、WebSocket 端点**(`/v1/duplex`, `/v1/realtime`, `/v1/realtime/robot/openpi`)。
- **视觉理解** (LLaVA / InternVL 等单 stage 视觉-语言融合模型)。
- **FastAPI / uvicorn / Pydantic / 插件入口**。HTTP 层用 stdlib `http.server`(`serving/openai_adapter.py`)。
- **多机械臂动作策略、机器人控制环、仿真器**。
- **HF-config 谓词消歧**(`hf_config_predicate`)。我们用 one-key-per-model-family。

这些都是刻意的范围边界,不是 bug。在 ticket 层级有记录,未来要重新打开时知道去哪里找。

---

## 入口对象:`Omni` 与 `AsyncOmni`

两个类共用同一个 `OmniBase`,调用 `generate(...)` 时行为不同:

| 维度 | `Omni` | `AsyncOmni` |
|---|---|---|
| `generate` 返回类型 | `list[OmniRequestOutput]` | `AsyncIterator[OmniRequestOutput]` |
| 内部调用 | `PipelineRunner.run` 同步直接调 | `PipelineExecutor.submit` 经 `asyncio.run_in_executor` 投递 |
| 单请求粒度 | 顺序 `for prompt in prompts` | `async for` 配合 `await executor.submit(...)` |
| 多请求并行 | 否(单卡串行) | 受 `extra["max_concurrent"]` 控制(默认 1) |
| 用途 | 脚本、批处理、回归测试 | HTTP 服务、异步 pipeline、与外部 `asyncio` 代码组合 |

两个类的 `generate` 都接受 `prompts: str | list[str]` 和 `sampling_params: SamplingParams | None` 两个位置参数,后者为 `None` 时回落到 `deploy/*.yaml` 里 `default_sampling_params` 配的默认值(详见 `PipelineRunner._stage_sampling`)。

`AsyncOmni.generate` 的返回类型是 `AsyncIterator` 而不是 `list`,意味着调用方必须 `async for` 或 `await anext(...)`。这跟 vllm-omni 的对齐面一致。

---

## `SamplingParams` 字段

`nanovllm_omni.config.params.SamplingParams` 是 `frozen=True` 的 dataclass,按 vllm-omni 的形状裁剪。

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `temperature` | `float` | `1.0` | 采样温度。`0.0` 等价于贪心。 |
| `top_p` | `float` | `1.0` | nucleus 采样阈值。 |
| `top_k` | `int` | `-1` | top-k 截断;`-1` 表示关闭。 |
| `max_tokens` | `int` | `16` | 单 stage 单请求最大生成 token 数。 |
| `stop` | `list[str] \| None` | `None` | 触发停止的字符串序列。 |
| `seed` | `int \| None` | `None` | 采样随机种子;`None` 表示不固定。 |
| `n` | `int` | `1` | 每个 prompt 返回几个候选。 |
| `extra` | `dict[str, Any]` | `{}` | 扩展点。thinker 识别 `open_thinking`(见 `models/minimind_omni/thinker.py:_thinker_stage`)等约定 key。 |

`SamplingParams` 的字段是**字典透传给 stage factory 的**:stage 收到的是 `(deploy_defaults, sampling)` 合并后的 dict,field 名(`temperature` / `top_p` / ...)会被读取,但任何 stage 都可能再读自己的私有约定 key(例如 minimind_omni thinker 会读 `open_thinking`)。新增约定 key 直接放 `extra` 即可,无需扩 dataclass。

---

## `OmniRequestOutput` 形状

`nanovllm_omni.outputs.OmniRequestOutput` 是 `frozen=True` 的 dataclass,封装一次请求的最终产物。

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | `str` | 默认 `""`;`from_pipeline(..., request_id="...")` 时由调用方设。HTTP 适配器用 `chatcmpl-{uuid4().hex[:24]}`。 |
| `outputs` | `Any` | 原始 payload(例如 `AudioPayload`、`ActionArtifact`)。消费方若想直接拿字节 / 数组,从这里取。 |
| `multimodal_output` | `MultimodalPayload \| None` | 按 modality 分桶的张量 + 元数据容器。详见下方"输出 payload"小节。 |
| `error` | `str \| None` | 若该次请求失败,这里放错误描述;`outputs` 和 `multimodal_output` 为 `None`。 |

三个类构造方法分别走不同路径:

| 方法 | 何时调用 | `multimodal_output` 内容 |
|---|---|---|
| `from_pipeline(output, request_id, final_output_type="audio")` | AR 多 stage 跑完,final stage 返回 audio payload | `{<final_output_type>: <value>}`(默认 `"audio"`) |
| `from_diffusion(output, request_id)` | 单 stage 扩散返回图片 | `{"image": <output>}` |
| `from_error(error, request_id)` | 异常路径 | `None`;`error` 字段填值 |

两个便利属性帮你判定走的是哪条路径:

- `is_pipeline_output` → `multimodal_output` 含 `"audio"` 键
- `is_diffusion_output` → `multimodal_output` 含 `"image"` 键

`unwrap()` 在 `error` 为真时抛 `RuntimeError(self.error)`,否则返回 `outputs`。HTTP 适配器走的是 `output.multimodal_output["audio"].wav_bytes()`(见下方"HTTP 入口")。

---

## 输出 payload:`MultimodalPayload` / `AudioPayload` / `ActionArtifact`

这是消费方最常接触的三个数据类型。

### `MultimodalPayload`

`dataclass(eq=False)`,**实现了 `Mapping[str, Any]` 协议**,所以可以直接当 dict 用:

```python
payload = output.multimodal_output          # Mapping[str, Any]
audio = payload["audio"]                    # AudioPayload | raw bytes | ...
audio_tensor = payload.primary_tensor       # 第一个 tensor 元素,或 None
payload.is_empty                            # True 表示没有任何 modality
```

内部两个字段:

| 字段 | 类型 | 含义 |
|---|---|---|
| `tensors` | `dict[str, Any]` | 真正的张量 / payload(例如 `AudioPayload`、`ActionArtifact`) |
| `metadata` | `dict[str, Any]` | 伴随的元数据(尚未被 stage 写满,保留位) |

为什么要走 `Mapping` 而不是裸 dict:让 `MultimodalPayload["audio"]` 在类型检查和运行时都像 dict 一样工作,降低 vLLM-Omni consumer 的迁移摩擦。底层 `__getitem__` 会先查 `tensors` 再查 `metadata`。

### `AudioPayload`

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `data` | `bytes` | (必填) | 原始 PCM(默认 16-bit mono)或已经是 WAV 字节流(以 `b"RIFF"` 开头) |
| `sample_rate` | `int` | `24000` | 采样率,Hz |

方法 `wav_bytes()`:

- 若 `data[:4] == b"RIFF"`,直接返回 `data`(已经是合法 WAV)
- 否则用 stdlib `wave` 包成 16-bit mono PCM WAV 返回

HTTP 适配器就是 `base64.b64encode(audio.wav_bytes())` 塞进 OpenAI 响应的 `choices[0].message.audio.data`。

### `ActionArtifact`

SmolVLA / 控制类模型的专用输出,2-D 数组 `[chunk_size, action_dim]`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `array` | `Any` | numpy 数组或带 `.shape` 的 torch 张量 |
| `action_dim` | `int` | 由 `shape[1]` 自动填 |
| `chunk_size` | `int` | 由 `shape[0]` 自动填 |
| `dtype` | `str` | 数组 `dtype` 的字符串形式 |

`__post_init__` 会校验 `array.shape` 确实是 2-D 且与 `action_dim` / `chunk_size` 一致,不匹配直接 `ValueError`。`from_array(array)` 是推荐的构造路径——它从数组自动推导后三个字段。

---

## Pipeline 拓扑

### `StageConfig` 字段

TICKET-02 之后的最小字段集(设计记录见 `.scratch/aligned-interfaces/issues/02-omni-end-to-end.md`):

| 字段 | 类型 | 用途 | 使用者 |
|---|---|---|---|
| `stage_id` | `int` | stage 在 pipeline 中的稳定顺序 | 所有模型 |
| `name` | `str` | 人类可读标识(如 `"thinker"`、`"dit"`) | 所有模型 |
| `kind` | `StageExecutionType` | 执行类:`LLM_AR` / `LLM_GENERATION` / `DIFFUSION` / `CODEC`,`StrEnum`,对照 vllm-omni | 所有模型 |
| `factory` | `str` | stage factory 的点分路径(`"package.module:attr"`),由 `resolve_stage_factory` 惰性解析 | 所有模型 |
| `process_input` | `str \| None` | bridge 钩子的点分路径,`None` 表示恒等透传 | minimind_o (TICKET-05),后续多 stage 模型 |
| `input_sources` | `tuple[int, ...]` | 该 stage 消费的 stage id;第一个 stage 为空 | 所有模型 |
| `is_terminal` | `bool` | 标记最终 stage,其输出即用户面结果 | 所有模型 |
| `final_output_type` | `str \| None` | `"audio"` / `"video"` / `"image"` / `"text"` / `"actions"` 之一;驱动 `OmniRequestOutput.multimodal_output` 的 key。SmolVLA 用 `"actions"` | 所有模型 |
| `model_subdir` | `str \| None` | checkpoint 子目录(如 `"language_model"`) | TICKET-06 / 后续 TTS 同构 ticket |
| `tokenizer_subdir` | `str \| None` | tokenizer 子目录 | 后续 TTS 同构 ticket |
| `diffusers_class_name` | `str \| None` | diffusers 类名 | TICKET-06 (Wan2.2) |

### `PipelineConfig` 字段

| 字段 | 类型 | 用途 |
|---|---|---|
| `name` | `str` | 规范模型标识(如 `"minimind_o"`、`"smolvla"`、`"wan2_2_ti2v"`) |
| `stages` | `tuple[StageConfig, ...]` | 有序 stages;长度 1 = 单 stage(扩散),长度 N = 多 stage AR pipeline |
| `default_deploy_config_name` | `str` | `deploy/` 目录下的文件名,加载到 `DeployConfig` |
| `registration_handles` | `tuple[str, ...]` | 该 pipeline 注册时挂的备用 key(如 HF repo id `"jingyaogong/minimind-3o"`)。默认 `(name,)` |
| `hf_architectures` | `tuple[str, ...]` | HF 架构别名,用于在 `OmniBase.try_infer_model_type` 的层 6 消歧 |
| `hf_config_predicate` | `Callable[[Any], bool] \| None` | 对加载到的 HF config 的额外谓词,默认 `None` |

---

## 运行时配置

### `DeployConfig` / `DeployStageConfig` 字段

`nanovllm_omni.config.registry` 里的 deploy 层 dataclass。`load_deploy_config(path)` 解析 YAML 成这个类型。

| 字段 | 类型 | 默认 | 用途 |
|---|---|---|---|
| `DeployConfig.max_batch` | `int` | `2` | 顶层全局;目前仅作记录,实际并发由 `OmniEngineArgs.extra["max_concurrent"]` 控制 |
| `DeployConfig.stages` | `tuple[DeployStageConfig, ...]` | `()` | 按 stage name 与 `PipelineConfig.stages` 对齐的部署参数 |
| `DeployStageConfig.name` | `str` | (必填) | 与对应 `StageConfig.name` 严格一致;`merge_pipeline_deploy` 按 name 匹配 |
| `DeployStageConfig.default_sampling_params` | `dict[str, Any]` | `{}` | stage 的 `SamplingParams` 默认值;运行时与请求级 `SamplingParams` 合并(请求级覆盖顶层,deploy defaults 始终进 `extra`) |

YAML 例子参见 `deploy/minimind_omni.yaml`,里面 `default_sampling_params: { temperature: 0.7, max_tokens: 512 }` 和 `watchdog_limit: 192` 这种私有约定 key 直接以 dict 形式存进去,直到该 stage factory 自己读它。当前 dataclass **只 schema-化上面四个字段**;YAML 里其它顶层 key(`connectors`、`platforms`、`stages[].devices`、`stages[].gpu_memory_utilization` 等 vllm-omni 的 schema)目前不会被 `load_deploy_config` 解析,与"`OmniEngineArgs` 生效矩阵"的策略保持一致:schema 锁形状,行为按 ticket 推进。

### `OmniEngineArgs` 字段生效矩阵

`nanovllm_omni.config.params.OmniEngineArgs` 沿用 vllm-omni 的字段形状(让构造函数签名和 kwargs 表面保持对齐),但**只有一部分字段在运行时真生效**。这一节锁住当前的生效集合,避免用户传了一个看起来对但被悄悄忽略的 kwarg。

| 字段 | 当前生效? | 读取位置 | 说明 |
|---|---|---|---|
| `model: str \| None` | ✅ 生效 | `entrypoints/base.py:143`, `models/minimind_omni/thinker.py:_thinker_stage` | 给 `bundle._resolve_snapshot` 用来定位 checkpoint。可以是 HF repo id 或本地路径。 |
| `device: str \| None` | ✅ 生效 | `entrypoints/base.py:245`, `models/minimind_omni/thinker.py:_thinker_stage` | 传给 `load_minimind_omni_bundle(device=...)` 和 `model.to(device)`。 |
| `extra: dict[str, Any]` | ✅ 生效 | `entrypoints/base.py:151,207,222,238`, `models/minimind_omni/thinker.py:_thinker_stage` | 扩展点 bag。已识别 key:`mimi_model_id` / `mimi`(thinker 专用)、`deploy_config_path`、`max_concurrent`。未知 key 留在 `extra` 里保留前向兼容。 |
| `dtype: str \| None` | ✅ 生效 | `models/smolvla/stage.py:65`, `models/minimind_omni/bundle.py:_cast_model_dtype` | SmolVLA 的 `_vla_stage` 读它做动态量化。minimind_omni 经 `_thinker_stage` / `_ensure_bundle` 透传到 `bundle._cast_model_dtype`,支持 `float16`(默认,与原 `.half()` 行为等价)/ `bfloat16` / `float32`。未知值抛 `ValueError`;cpu 总是 no-op。SmolVLA 的 int8/qint8 路径不走流过 minimind_omni 的 helper。 |
| `enforce_eager: bool = False` | ❌ no-op | (无) | dataclass 上存着,没人读。TICKET-05 把 thinker 提升为 vLLM-AR 循环、能 opt-out cudagraph capture 时变生效。 |
| `gpu_memory_utilization: float = 0.9` | ❌ no-op | (无) | 单进程单 GPU;KV cache 大小由 bundle loader 固定。为对齐保留,plumb-up 延后。 |
| `max_num_seqs: int \| None = None` | ❌ no-op | (无) | MiniMind-O 用内置 `BatchedGenerationRunner` 批;per-stage `max_num_seqs` 应放 `DeployStageConfig`(TICKET-07)。 |
| `max_num_batched_tokens: int \| None = None` | ❌ no-op | (无) | 同上,留给 TICKET-07。 |
| `tensor_parallel_size: int = 1` | ❌ no-op | (无) | `AGENTS.md` 锁了单 GPU 范围。多 GPU 要等 TK-007 多副本回归。 |
| `trust_remote_code: bool = True` | ✅ 生效 | `models/minimind_omni/bundle.py:load_minimind_omni_bundle` | 经 `_thinker_stage` / `_ensure_bundle` 透传到 bundle 的 `from_pretrained(..., trust_remote_code=...)`,覆盖 causal LM 和 tokenizer。默认 `True` 与原硬编码行为一致。 |

#### 什么是"生效"

**生效** = 至少有一个消费方(entrypoint、engine 层、stage factory)在请求 / 加载路径上读了它。**no-op** = 仅为对齐 schema 存在,没有任何消费方引用。no-op 字段不抛错,静默忽略。这是有意为之:TICKET-02 → TICKET-07 的对齐爬坡阶段先把 schema 钉死,后续 ticket 把 no-op 提升为生效时消费方不会断。本表是当前生效集合的真理来源。

#### 为什么用矩阵而不是删字段 / 加警告

- **删字段**:回归对齐契约——按 vllm-omni 写的消费方迁过来时不得不删 kwargs。
- **运行时警告**:吵,每次 `Omni(...)` 构造都会响,包括测试。
- **文档化矩阵**:保持 schema 稳定以服务对齐路线图,同时把缺口显式化、可引代码。后续把 no-op 提升为生效时,本表一行编辑加消费方改动即可。

#### 如何验证矩阵

```bash
cd /Users/mcig/Projects/nanovllm-omni
# 直接属性读(entrypoints/base.py 路径)。
grep -rn -E '\b(args|self\.args|self\.engine_args)\.(enforce_eager|gpu_memory_utilization|max_num_seqs|max_num_batched_tokens|dtype|tensor_parallel_size|trust_remote_code|device|model|extra)\b' \
    nanovllm_omni/engine nanovllm_omni/models nanovllm_omni/entrypoints
# 防御式 ``getattr(args, "<field>", default)`` 读(stage factory 路径)。
grep -rn -E 'getattr\(args,\s*"(enforce_eager|gpu_memory_utilization|max_num_seqs|max_num_batched_tokens|dtype|tensor_parallel_size|trust_remote_code|device|model|extra)"' \
    nanovllm_omni/engine nanovllm_omni/models nanovllm_omni/entrypoints
```

两个 pattern 加起来应匹配上表的"读取位置"列。bundle 契约另有 `tests/test_bundle_loader.py`(cast helper + thinker plumbing + trust_remote_code plumbing)锁定;dataclass 层的 no-op 容忍性由 `OmniEngineArgs.__init__` 实现,由 `tests/test_config.py::test_engine_args_accepts_deploy_kwargs` 验证。

---

## 引擎分层

vllm-omni 的 `StagePool`(1281 LOC)和 `Orchestrator`(2428 LOC)是面向**多副本路由**和**跨 stage 请求生命周期**的分布式抽象。`nanovllm-omni` 面向**单进程单 GPU** 安装场景(用户工作站 + 偶发租卡跑扩散)。我们用同样的"路由 vs 跟踪"划分,但以单进程形式落地:

- **`PipelineRunner`**:同步单副本 pipeline runner。把一次请求顺序跑过 `PipelineConfig.stages`,~100 LOC。完全不知道并发的存在。
- **`PipelineExecutor`**:`PipelineRunner` 的 async 包装,给 HTTP / `AsyncOmni` 用。用 `asyncio.run_in_executor` 把同步的 `PipelineRunner.run` 投出去。`max_concurrent` 默认 `1`(单 GPU),~80 LOC。

划分跟 vllm-omni 的"pool 路由到副本,orchestrator 跟踪请求"同构,但合并到一条直线:runner 是 pipeline config 的唯一消费者,executor 是 runner 的唯一消费者。`nanovllm_omni.engine/` 下**不出现** `StagePool` 或 `Orchestrator` 类名——这两个名字**留给未来 TK-007 引入多副本时再用**。

---

## 模型路径解析

MiniMind-O 的 bundle loader(`nanovllm_omni.models.minimind_omni.bundle._resolve_snapshot`)对齐 vllm-omni 的 `_resolve_model_to_local_path`(`vllm_omni/engine/stage_init_utils.py`),保证两个引擎在离线场景下用同一种方式失败:

1. 字符串已经是磁盘上的目录 → 直接用,不查 HF cache。
2. 否则走 `huggingface_hub.snapshot_download(..., local_files_only=True)`——**只读本地 HF cache,不触发任何网络下载**。
3. 找不到的 Hub id(本地 cache 没有)→ 不抛错,记一条 `WARNING` 日志,原样回传,让下游 `from_pretrained` 自己抛更清楚的错。

之前的行为是不带 `local_files_only` 调 `snapshot_download`,结果用户在没网的机器上跑默认 Hub id 时拿到的是 `LocalEntryNotFoundError: ConnectError: Network is unreachable` 而不是清晰的离线模式提示。此契约由 `tests/test_bundle_resolve_snapshot.py` 锁住。

---

## HTTP 入口:`POST /v1/chat/completions`

面向消费者的第二个入口是 OpenAI 兼容的 HTTP 适配器,实现见 `nanovllm_omni/serving/openai_adapter.py`。**只用 stdlib**:`http.server.BaseHTTPRequestHandler` + `ThreadingHTTPServer`,没有 fastapi / uvicorn / pydantic。

### 启动

```bash
python -m nanovllm_omni.serving.openai_adapter \
    --config deploy/minimind_omni.yaml \
    --model-id jingyaogong/minimind-3o \
    --mimi-model-id kyutai/mimi \
    --device cuda \
    --host 127.0.0.1 \
    --port 8000
```

默认 `--config` 指向 `deploy/minimind_omni.yaml`。

### 请求

```http
POST /v1/chat/completions HTTP/1.1
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "你好,讲一个关于迷你小人的故事"}
    ]}
  ],
  "model": "minimind-3o"
}
```

支持的形态:

- `messages[i].content` 必须是 OpenAI multimodal 内容块数组(对齐 vllm-omni 的 `run_curl_multimodal_generation.sh`),目前只接受 `type: "text"` 块,adapter 拼接最后一个 `role=user` 消息的所有文本块
- 裸字符串 `content: "..."`、未知 `type`(如 `image_url`、`audio_url`)或缺 `user` 消息都会返回 400 `invalid_request_error`
- 图像 / 音频输入尚未实现(TICKET-06 才考虑)

### 响应

```json
{
  "id": "chatcmpl-<uuid24>",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "minimind-3o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "audio": {
        "data": "<base64 WAV bytes>",
        "format": "wav",
        "sample_rate": 24000
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": <int>,
    "completion_tokens": 0,
    "total_tokens": <int>
  }
}
```

`choices[0].message.audio.data` 是 base64 编码的 16-bit mono PCM WAV,采样率 24000 Hz(由 `AudioPayload.wav_bytes()` 产生)。

错误码:

- `400 invalid_request_error`:请求体解析失败 / `messages` 不合法 / 缺 user 消息
- `404`:路径不是 `/v1/chat/completions`
- `500 server_error`:engine 抛异常(详细错误在 `error.message`)

`Streaming`、多 `choices`(`n>1`)、tools/function_call 都不支持。Adapter 用 `ThreadingHTTPServer`,并发受 `max_concurrent=1` 默认值限制。

---

## 对齐符号变更流程

改任何对齐的公共符号或响应字段时:

1. 更新本表相应位置;
2. 在 `tests/` 加一条聚焦的契约测试(签名、字段、HTTP 响应形状);
3. 跑 `AGENTS.md` 列的 CI 等价检查:`ruff check` / `black --check` / `pytest -m "not smoke"` / 公共 API 导入 smoke / `python -m compileall`。
4. 单独 commit 一条,描述"对齐契约变更",然后 push。

如果是改动(`OmniRequestOutput` 字段、`/v1/chat/completions` 响应形状之类),把验证脚本里对应的断言也一起更新;不要留漂移的契约代码。