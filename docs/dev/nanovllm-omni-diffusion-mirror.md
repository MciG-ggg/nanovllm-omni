# nanovllm-omni diffusion 阶段镜像方案

**状态：** 设计计划。Owner: engine team。受众：要把 `sd_turbo` /
Phase 3 的 `sd3.5-medium + qwen-image-edit` / 未来 audio diffusion 模型
接入 `Omni.generate` 的工程师。

本仓库是教学项目，目标是在 4 GB RTX 3050 上跑通 omni 推理链路；一切
取舍的判据是"代码是否更容易读懂"（AGENTS.md）。vllm-omni 的 diffusion
栈是参考，**不** import、**不**复制粘贴。

本文档定义：在 `nanovllm-omni` 里镜像 vllm-omni 的 diffusion engine 需要
哪些新文件、改哪些接缝、跳过哪些东西。

参考架构详见 [`vllm-omni-diffusion-reference.md`](./vllm-omni-diffusion-reference.md)。
两者互为索引。

## 1. 现状：哪些钩子已经接好

下面这些已经在仓里就位，**不要再改**：

| 钩子 | 位置 | 状态 |
|---|---|---|
| `StageExecutionType.DIFFUSION` 枚举值 | `nanovllm_omni/config/registry.py:45` | ✓ |
| `StageConfig.kind` 字段 + 构造时类型校验 | `nanovllm_omni/config/registry.py:81-122` | ✓ |
| `StageConfig.stage_factory` tuple 解析 | `nanovllm_omni/config/registry.py` | ✓ |
| `sd_turbo` 模型族（DIFFUSION stage） | `nanovllm_omni/models/sd_turbo/pipeline.py:27` | ✓ |
| `_load_builtin_pipelines` 自动注册 | `nanovllm_omni/config/registry.py:402-411` | ✓ |
| `OmniRequestOutput.from_diffusion` 工厂 | `nanovllm_omni/outputs.py:249-261` | ✓ |
| `OmniRequestOutput.is_diffusion_output` | `nanovllm_omni/outputs.py:308` | ✓ |
| `Omni.generate` 通过 `PipelineExecutor → PipelineRunner.run` 串联 stages | `nanovllm_omni/entrypoints/omni.py` + `nanovllm_omni/engine/runner.py:132` | ✓ |

也就是说：**数据侧的契约已经完整**，缺的只是 engine 这一层 ——
当前 `models/sd_turbo/stage.py:_sd_turbo_stage` 返回的是 sync
closure `forward(payload, sampling)`，直接把 `pipe(...)` 调一遍
diffusers，绕过任何 scheduler / runner。

## 2. 镜像方案

### 2.1 新增 6 个文件（`nanovllm_omni/diffusion/`）

```
nanovllm_omni/diffusion/
  __init__.py        # re-export                          ~10 LOC
  interface.py       # DiffusionPipeline Protocol（4 方法）~30 LOC
  request.py         # OmniDiffusionRequest dataclass     ~30 LOC
  scheduler.py       # RequestScheduler（跳过 StepScheduler）~80 LOC
  runner.py          # DiffusionRunner（包 pipeline）     ~100 LOC
  engine.py          # DiffusionEngine（编排 + 线程）     ~150 LOC
  client.py          # InlineDiffusionClient              ~120 LOC
                              总计 ~520 LOC
```

每个文件的最小骨架见 §2.3。

### 2.2 修改 2 处接缝

#### 接缝 A：`engine/runner.py:151`

当前：

```python
# nanovllm_omni/engine/runner.py:151
payload = instance(payload, stage_sampling)
```

改为：

```python
if stage_cfg.kind == StageExecutionType.DIFFUSION:
    payload = self._diffusion_clients[stage_cfg.name].run(payload, stage_sampling)
else:
    payload = instance(payload, stage_sampling)
```

`_diffusion_clients` 是 `PipelineExecutor.__init__`（`engine/executor.py:35`
附近）按 `stage.kind == DIFFUSION` 预构建的 dict，键为 stage name。

**契约不变**：`payload` 仍是上一步的产物（str / dict / `AudioPayload` /
PIL.Image），进 `OmniRequestOutput.from_diffusion(...)` 之前形状不
强约束。

#### 接缝 B：`models/sd_turbo/stage.py` 返回值

当前 `_sd_turbo_stage` 返回 sync closure `forward(payload, sampling)`，
里头 `pipe(prompt, num_inference_steps=1, ...).images[0]` 一次性跑完。

改为：返回一个 `SdTurboPipeline` 对象，实现
`nanovllm_omni/diffusion/interface.py:DiffusionPipeline` 的 4 方法：

```python
class SdTurboPipeline:                                # implements DiffusionPipeline
    supports_step_execution: ClassVar[bool] = True
    def prepare_encode(self, state) -> StepState: ... # text → text_embeds
    def denoise_step(self, input_batch, *, states) -> Tensor | None: ...  # UNet
    def step_scheduler(self, state, noise_pred) -> None: ...            # Euler
    def post_decode(self, state) -> DiffusionOutput: ...                # VAE → PIL
```

`_sd_turbo_stage` 工厂**保留**：只把返回类型从 closure 改成
`SdTurboPipeline` 实例（用 `diffusers.StableDiffusionPipeline` 拆解）。
`DiffusionEngine` 通过 `DiffusionRunner` 调这 4 个方法，不再直接调
`pipe(...)`。

### 2.3 最小骨架（参考）

`DiffusionPipeline` Protocol（`diffusion/interface.py`）：

```python
from typing import ClassVar, Protocol
from .request import DiffusionRequest

class StepState(Protocol):
    request_id: str
    latents: object | None  # 当前去噪 latent
    encoder_hidden_states: object | None  # 文本条件

class DiffusionOutput(Protocol):
    images: list | None
    error: str | None
    finished: bool

class DiffusionPipeline(Protocol):
    supports_step_execution: ClassVar[bool] = True
    def prepare_encode(self, request: DiffusionRequest) -> StepState: ...
    def denoise_step(self, state: StepState, *, step: int,
                     num_steps: int) -> object: ...
    def step_scheduler(self, state: StepState, noise_pred: object) -> None: ...
    def post_decode(self, state: StepState) -> DiffusionOutput: ...
```

`DiffusionEngine.step_streaming`（`diffusion/engine.py`）核心 30 行：

```python
async def step_streaming(self, request):
    state = self.pipeline.prepare_encode(request)
    self.scheduler.add_request(state)
    while self.scheduler.has_requests():
        sched = self.scheduler.schedule()
        noise_pred = self.pipeline.denoise_step(state, step=sched.step_id,
                                                 num_steps=request.num_inference_steps)
        self.pipeline.step_scheduler(state, noise_pred)
        if sched.step_id + 1 >= request.num_inference_steps:
            output = self.pipeline.post_decode(state)
            self.scheduler.finish(state.request_id)
            yield [OmniRequestOutput.from_diffusion(output.images[0], state.request_id)]
            return
        self.scheduler.update_from_output(sched, None)
```

`InlineDiffusionClient`（`diffusion/client.py`）核心 20 行：

```python
class InlineDiffusionClient:
    stage_type: str = "diffusion"
    def __init__(self, pipeline_factory, stage_cfg):
        self._engine = DiffusionEngine(pipeline_factory(stage_cfg))
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="diffusion")
    async def add_request_async(self, request_id, prompt, sampling_params, **kw):
        loop = asyncio.get_running_loop()
        async for outputs in self._engine.step_streaming(
            OmniDiffusionRequest(request_id=request_id, prompt=prompt,
                                 sampling_params=sampling_params)):
            self._output_queue.put_nowait(outputs[0])
    def run(self, payload, sampling):
        # sync path for PipelineRunner.run (REQ_BATCH 模式)
        request = OmniDiffusionRequest(request_id="sync", prompt=payload,
                                       sampling_params=sampling)
        return self._engine.run_sync(request)
```

## 3. 跳过清单

下表来自
[`vllm-omni-diffusion-reference.md`](./vllm-omni-diffusion-reference.md)
§1，标注哪些 vllm-omni 模块**不要**镜像到本仓库：

| 模块 | vllm-omni 行数 | 跳过理由 |
|---|---|---|
| `worker/diffusion_worker.py` | 1597 | 多 GPU / 分布式 / 设备初始化 — 3050 单卡 |
| `stage_diffusion_client.py` + `stage_diffusion_proc.py` | 510 + 788 | ZMQ 子进程 IPC — 单进程不需要 |
| `sched/step_scheduler.py` | 140 | streaming / interaction — 等真需要再加 |
| `diffusion_kv/` 整包 | — | paged KV on diffusion latents — 4 GB 没空间 |
| `cache/{teacache,cachedit,magcache,stepcache}/` | — | cache 后端 — 等 profile 显示慢再加 |
| `attention/parallel/`, `attention/backends/ring/` | — | Ulysses / Ring attention — 单卡 |
| `offloader/`, `distributed/` | — | layerwise offload / 多卡分片 — 单卡 |
| `quantization/` | — | FP8 / INT8 量化 — 3050 fp16 已够 |
| 50 个 `models/<family>/` 全部 | — | 本仓库已有 5 个，够教 |

**结果**：~520 LOC 新代码镜像 vllm-omni 的 ~7500 LOC（≈7%）。保留完整
engine 边界（request→scheduler→runner→pipeline）和 AR↔diffusion 分发
契约，砍掉一切 3050 跑不到的部分。

## 4. 实施顺序

按依赖顺序，分 4 个 commit：

1. **commit 1 — Protocol + dataclass**
   新建 `nanovllm_omni/diffusion/{__init__.py, interface.py, request.py}`。
   加 `tests/test_diffusion_protocols.py`：`DiffusionPipeline` 是
   `Protocol`、不强制运行时检查，至少锁住 4 方法名 + `supports_step_execution`。

2. **commit 2 — Scheduler + Runner**
   新建 `nanovllm_omni/diffusion/{scheduler.py, runner.py}`。
   `RequestScheduler` 复用
   `BaseScheduler` 风格（waiting/running/finish deque）但**不**复制
   vllm-omni 的 `BaseScheduler`（后者依赖 paged KV，太重）。
   加 `tests/test_diffusion_scheduler.py`：FIFO 入队 / capacity 限制 /
   finish 状态机。

3. **commit 3 — Engine + Client**
   新建 `nanovllm_omni/diffusion/{engine.py, client.py}`。
   `DiffusionEngine.step_streaming` 走单线程 + `asyncio.Queue`，
   不开 worker thread（参考 vllm-omni 的 inline 路径但简化）。
   `InlineDiffusionClient` 实现 §2.3 的 20 行骨架。
   加 `tests/test_diffusion_engine.py`：跑通 REQUEST_BATCH 端到端。

4. **commit 4 — 接缝 A + 接缝 B**
   改 `engine/runner.py:151` 加 `if stage_cfg.kind == StageExecutionType.DIFFUSION` 分支。
   改 `models/sd_turbo/stage.py:_sd_turbo_stage` 返回
   `SdTurboPipeline`（实现 `DiffusionPipeline` Protocol 的 4 方法）。
   加 `tests/test_sd_turbo_pipeline.py`：跑一次 `Omni.generate(prompt)`
   端到端，验证生成的 PIL.Image 与改动前位级一致（同一个 sd-turbo
   checkpoint + seed 应该是 deterministic）。

每一步都跑 `scripts/pre-commit`（lint + 单测）；commit 4 跑
`scripts/pre-push`（完整 CI 等价集）。

## 5. 验证

### 5.1 引擎层（commit 3）

```python
from nanovllm_omni.diffusion import DiffusionEngine, OmniDiffusionRequest
from nanovllm_omni.diffusion.interface import DiffusionPipeline

class IdentityPipeline:
    supports_step_execution = True
    def prepare_encode(self, r): return {"latents": 0.0, "request_id": r.request_id}
    def denoise_step(self, state, *, step, num_steps): return 0.1
    def step_scheduler(self, state, noise_pred): state["latents"] += noise_pred
    def post_decode(self, state): return type("O", (), {"images": [state["latents"]],
                                                         "error": None, "finished": True})()

engine = DiffusionEngine(IdentityPipeline())
outputs = engine.run_sync(OmniDiffusionRequest(request_id="t", prompt="hi",
                                               sampling_params=type("S",(),{"num_inference_steps":3,"extra":{}})()))
assert outputs[0].images[0] == pytest.approx(0.3)  # 3 × 0.1
```

### 5.2 接入层（commit 4）

```python
from nanovllm_omni import Omni, SamplingParams

# sd-turbo 不走 ModelRunner；graph-vs-eager 由 stage 默认决定（见 ADR-0005）。
# OmniEngineArgs 不再接受 enforce_eager  kwarg（config/params.py:69-71）。
omni = Omni(model="stabilityai/sd-turbo")
out = omni.generate("a photo of a cat",
                    sampling_params=SamplingParams(extra={"num_inference_steps": 1}))
assert out[0].final_output_type == "image"
assert out[0].multimodal_output["image"] is not None  # PIL.Image
# 与改动前位级一致（同 checkpoint + 同 seed）
```

### 5.3 CI 门禁

`scripts/pre-push` 已经覆盖 ruff + black + pytest + compileall。
新增两个测试文件（`test_diffusion_engine.py`、`test_sd_turbo_pipeline.py`）
要随 commit 一起进入；`@pytest.mark.smoke` 标记 sd-turbo 端到端测试，
避免 3050 上无显卡环境失败。

## 6. 风险与回退

- **风险 1**：`PipelineRunner.run` 改成同步阻塞 diffusers 调用 → 接缝 A
  改完后变成异步（`InlineDiffusionClient.run` 走 `ThreadPoolExecutor`）。
  现有 sd-turbo smoke test（`tests/test_serving_image.py`）要同步改，
  否则会卡死 event loop。
- **风险 2**：`SdTurboPipeline.post_decode` 返回的 `DiffusionOutput`
  必须能被 `OmniRequestOutput.from_diffusion` 直接吃。`from_diffusion`
  当前期望 `output` 是 `list | tuple | 单值`（`outputs.py:249-261`），
  `SdTurboPipeline.post_decode` 返回 `[PIL.Image]` 列表，兼容。
- **回退**：commit 4 是最后一步，回退到 commit 3 时把
  `engine/runner.py:151` 改回 `payload = instance(payload, stage_sampling)`，
  删 `_diffusion_clients` 引用即可，commit 1-3 仍可保留（未接入）。
