# vLLM-Omni diffusion engine: 结构参考

**状态：** 参考性结构梳理。Owner: engine team。受众：任何打算在
`nanovllm-omni` 里加 diffusion 阶段、或阅读 `vllm_omni/diffusion/`
源码的人。

本仓库把 `vllm-omni` 当作只读参考
（`/Users/mcig/Projects/vllm-omni`，AGENTS.md 明确）。本文档把
`vllm_omni/diffusion/` 下的代码按"读者最想知道的 5 件事"重新切片：

1. 文件骨架（哪些文件、各自负责什么）
2. 数据流（一个请求从进入到输出）
3. 两种执行模式（什么时候走哪条）
4. Pipeline 接口契约（一个模型族要实现什么）
5. 入口层 AR ↔ diffusion 怎么分发（Omni 怎么区分两种阶段）

对照镜像计划见 `docs/dev/nanovllm-omni-diffusion-mirror.md`。

## 1. 文件骨架

`vllm-omni` 的 diffusion 是一个**完整独立**的 mini-engine，**不复用**
vLLM 的 `EngineCore`。所有相关代码集中在
`vllm_omni/diffusion/`，~7500 LOC：

| 文件 | 行数 | 角色 |
|---|---|---|
| `diffusion/diffusion_engine.py` | 1271 | 编排器（engine）：拥有 scheduler + executor + worker + 后处理 |
| `diffusion/worker/diffusion_model_runner.py` | 1055 | runner：每 tick 调一次 `pipeline.denoise_step()` |
| `diffusion/worker/diffusion_worker.py` | 1597 | worker：多 GPU / 分布式 / 设备初始化 |
| `diffusion/worker/input_batch.py` | 772 | batch 输入装配（多请求合批） |
| `diffusion/sched/base_scheduler.py` | 446 | scheduler ABC：waiting/running/finish 队列契约 |
| `diffusion/sched/request_scheduler.py` | 133 | 请求级批处理（REQUEST_BATCH 模式） |
| `diffusion/sched/step_scheduler.py` | 140 | 步进式调度（STEP_BATCH 模式） |
| `diffusion/request.py` | 96 | `OmniDiffusionRequest` dataclass |
| `diffusion/inline_stage_diffusion_client.py` | 329 | **单进程客户端**：`ThreadPoolExecutor` 跑 engine |
| `diffusion/stage_diffusion_client.py` | 510 | ZMQ 子进程 IPC 客户端（多副本场景） |
| `diffusion/stage_diffusion_proc.py` | 788 | ZMQ 子进程入口（与上一项配对） |
| `diffusion/models/interface.py` | 162 | `SupportsStepExecution` / `SupportsComponentDiscovery` Protocol |
| `diffusion/registry.py` | 745 | `model_class_name → pipeline_class` 映射表 |
| `diffusion/executor/abstract.py` + `multiproc_executor.py` | — | executor 抽象 + 多进程实现 |

模型族本体落在 `diffusion/models/<family>/pipeline_<family>.py`（如
`wan2_2/pipeline_wan2_2.py` 45K 行、`flux/pipeline_flux.py` 等），
通过 `registry.py` 注册。

**关键设计选择**：AR 路径和 diffusion 路径**完全分离**。AR 走 vLLM 的
`EngineCore`（由 `vllm_omni/patch.py` monkey-patch 加入 multimodal 字段），
diffusion 走自己这套栈。两套栈在入口层
（`vllm_omni/entrypoints/omni.py`）通过统一的 client protocol 重新汇合。

## 2. 数据流

```text
Omni.generate(omni.py:103)
  └─ engine.add_request(req_id, prompt, sampling_params_list)  # 按 stage 分发
       └─ stage_client.add_request_async(...)                  # StageClientBase
            └─ [Inline] DiffusionEngine.step_streaming(req)   # AsyncGenerator
                 ├─ scheduler.add_request(req)                 # 入 waiting 队列
                 ├─ while has_requests:
                 │    sched_output = scheduler.schedule()      # → DiffusionSchedulerOutput
                 │    if REQUEST_BATCH:
                 │        output = executor.execute_batch()
                 │            └─ worker.execute_model_batch(...)
                 │                └─ runner.execute_model_batch(...)
                 │                    └─ pipeline.forward(input_batch)  # 全程
                 │    elif STEP_BATCH:
                 │        output = executor.execute_step(sched_output)
                 │            └─ worker.execute_stepwise(sched_output)
                 │                └─ runner._execute_stepwise(...)
                 │                    ├─ pipeline.denoise_step(input_batch, states)
                 │                    ├─ pipeline.step_scheduler(state, noise_pred)
                 │                    ├─ (loop until num_inference_steps)
                 │                    └─ pipeline.post_decode(state) → DiffusionOutput
                 ├─ DiffusionEngine.postprocess_output(req, output) → OmniRequestOutput
                 └─ yield formatted_outputs
```

各阶段产物的形状：

| 阶段 | 形状 | 关键字段 |
|---|---|---|
| `DiffusionSchedulerOutput` | dataclass | `scheduled_new_reqs`, `scheduled_cached_reqs`, `finished_req_ids`, `num_running_reqs`, `kv_prefetch_job` |
| `DiffusionOutput` | dataclass | `images`/`videos`/`audios`, `error`, `aborted`, `metrics`, `async_output_id` |
| `OmniRequestOutput` | 公开对齐类型 | `request_id`, `final_output_type`, `multimodal_output`, `metrics` |

## 3. 两种执行模式

`DiffusionEngine._resolve_execution_mode()`
（`diffusion_engine.py:228-247`）在 engine 初始化时根据
`od_config.step_execution` 决定走哪条路：

| 模式 | 触发条件 | runner 调什么 | 何时用 |
|---|---|---|---|
| `REQUEST_BATCH` | `step_execution=False`（默认） | `pipeline.forward(input_batch)`（pipeline 自己跑完整 denoise 循环） | 吞吐优先；多请求合进一次 forward |
| `STEP_BATCH` | `step_execution=True` 或 `streaming_output=True` | `pipeline.denoise_step()`（engine 每 tick 一步） | streaming / interaction / cache_backend |

`streaming_output=True` 会**强制开启** step_execution
（`diffusion_engine.py:233`），因为只有 step 模式能在生成中途 yield。

`REQUEST_BATCH` 还有一条隐含契约：模型族必须把
`supports_request_batch: ClassVar[bool] = True` 暴露在 pipeline 类上
（参见 `diffusion_engine.py:99-110` 的 `supports_request_batch` 函数）。
否则 engine 会拒绝 `max_num_seqs > 1` 的配置。

## 4. Pipeline 接口契约

每个模型族（`vllm_omni/diffusion/models/<family>/pipeline_<family>.py`）
实现 `vllm_omni/diffusion/models/interface.py:49` 的
`SupportsStepExecution` Protocol：

```python
class SupportsStepExecution(Protocol):
    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: StepRequestState, **kwargs) -> StepRequestState:
        """prompt / 条件编码 → 初始 state（如文本 embedding、噪声 latent）"""

    def denoise_step(self, input_batch: InputBatch, *,
                     states: Sequence[StepRequestState] | None = None,
                     **kwargs) -> torch.Tensor | None:
        """跑一步去噪 forward；返回 noise_pred；返回 None 表示中断"""

    def step_scheduler(self, state: StepRequestState,
                       noise_pred: torch.Tensor, **kwargs) -> None:
        """按 σ schedule 更新 state.latents（Euler / DPM++ / FlowMatch 等）"""

    def post_decode(self, state: StepRequestState, **kwargs) -> DiffusionOutput:
        """VAE / vocoder decode → 最终 DiffusionOutput（图像 / 视频 / 音频）"""
```

外加可选的 `SupportsComponentDiscovery` Protocol
（`interface.py:96`），声明 `_dit_modules` / `_encoder_modules` /
`_vae_modules` 子模块路径，用于 CPU offload 和 HSDP 分片发现 —
**3050 用不到，可不实现**。

## 5. 入口层 AR ↔ diffusion 怎么分发

`vllm_omni/engine/stage_client.py:23` 定义 Protocol 树：

```text
StageClient(Protocol)
├─ StagePoolClient(Protocol)
│   ├─ StagePoolLLMClient        # AR 用
│   │   ├─ add_request_async(req: EngineCoreRequest)
│   │   ├─ get_output_async() -> EngineCoreOutputs
│   │   ├─ set_engine_outputs(...)
│   │   └─ get_kv_sender_info(...)
│   │
│   └─ StagePoolDiffusionClient  # diffusion 用
│       ├─ add_request_async(req_id, prompt, sampling_params, kv_sender_info)
│       ├─ get_diffusion_output_nowait() -> OmniRequestOutput | None
│       ├─ submit_interaction_async(...)
│       └─ (公共) abort_requests_async / collective_rpc_async
```

所有 client 类都有 `stage_type: str` 类属性作为判别符：

- AR 客户端：`StageEngineCoreClient.stage_type = metadata.stage_type`
  （"llm"，从 config 拿，见
  `vllm_omni/engine/stage_engine_core_client.py:138`）
- diffusion 客户端：`InlineStageDiffusionClient.stage_type = "diffusion"`
  （硬编码，见
  `vllm_omni/diffusion/inline_stage_diffusion_client.py:38`）

`Omni.generate`（`entrypoints/omni.py:103-160`）**完全不分支**：

```python
self.engine.add_request(request_id=req_id, prompt=prompt,
                        sampling_params_list=req_sp_list, ...)
...
while active_reqs:
    msg = self.engine.try_get_output()       # 统一拿输出
    ...
```

唯一一处 AR 特定的逻辑在
`Omni._maybe_force_final_only_for_llm_stages`
（`entrypoints/omni.py:42-58`）：把 LLM stage 的
`output_kind` 强制成 `FINAL_ONLY`，diffusion stage **完全不动**。

调度层在 `vllm_omni/engine/stage_init_utils.py:441-446` 按
`stage_type == "diffusion"` 选择构造哪种 `StageMetadata`（含默认
`OmniDiffusionSamplingParams`），再在 `engine/async_omni_engine.py`
里用 `if pool.stage_type == "diffusion"` 分流到对应的 client 池。

## 6. Inline vs IPC

两个客户端实现同一 `StagePoolDiffusionClient` Protocol，按场景二选一：

| 客户端 | 行数 | 适用场景 | 实现要点 |
|---|---|---|---|
| **`InlineStageDiffusionClient`** | 329 | **单 GPU / 单 stage / 单进程** | `ThreadPoolExecutor(max_workers=1)` 在 Orchestrator 同进程跑 `DiffusionEngine`；无 ZMQ |
| `StageDiffusionClient` + `StageDiffusionProc` | 510 + 788 | 多副本 / 跨主机 / GPU 隔离 | ZMQ IPC + `StageDiffusionProc` 子进程 + 进程 liveness monitor |

`InlineStageDiffusionClient` 的核心 50 行
（`inline_stage_diffusion_client.py:127-180`）：

```python
async def _dispatch_request(self, request_id, prompt, sampling_params,
                            kv_sender_info=None):
    request = OmniDiffusionRequest(prompt=prompt,
                                   sampling_params=sampling_params,
                                   request_id=request_id,
                                   kv_sender_info=kv_sender_info)
    if self.od_config.streaming_output:
        async for results in self._engine.step_streaming(request):
            self._output_queue.put_nowait(results[0])
    else:
        # drain generator, publish only the final output
        result = None
        async for results in self._engine.step_streaming(request):
            result = results[0]
        if result is None:
            raise RuntimeError("Diffusion execution finished without output.")
        self._output_queue.put_nowait(result)
```

## 7. 关键 takeaway（落到 nanovllm-omni 之前的速读）

1. **engine 形状**：request → scheduler → executor → worker → runner → pipeline.{denoise_step, step_scheduler, post_decode} → DiffusionOutput → OmniRequestOutput。这跟 nanovllm 的 request → scheduler → runner 边界**完全对齐**，镜像时只是把 nanovllm 的 `LLMEngine` 拆成两半。
2. **AR 和 diffusion 完全分离**：不要尝试让 diffusion 走 AR scheduler / vLLM EngineCore。`vllm_omni/patch.py` 注入的 hook 跟 diffusion 一点关系都没有。
3. **两种 stage 共享 client protocol，不共享实现**：AR 和 diffusion 各自的 client 都实现 `add_request_async + get_output*_nowait + abort + collective_rpc + check_health + shutdown`，所以 `Omni.generate` 不需要分支。
4. **`SupportsStepExecution` 是模型族的硬契约**：4 个方法，`prepare_encode` / `denoise_step` / `step_scheduler` / `post_decode`。任何 diffusion 模型族都必须实现，缺一不可。
5. **单进程选 Inline**：多副本再考虑 IPC。

镜像到本仓库的具体方案见
[`nanovllm-omni-diffusion-mirror.md`](./nanovllm-omni-diffusion-mirror.md)。
