---
name: add-new-model
description: 给 nanovllm-omni 新增模型族(走 pipeline-registry 契约)。当用户说 "add a new model"、"new model family"、"加新模型"、"新增模型"、或想接线一个新模型族进 registry / deploy yaml / Omni(...) 生成链路时使用。含 3050(4GB) 选型、接线、测试、对齐文档、提交前闸门。
---

# 给 nanovllm-omni 新增模型族

nanovllm-omni 是 registry 驱动:`模型`= 一个 **PipelineConfig**(注册在 `pipeline.name` + 每个 `registration_handles` 别名下),每个 stage 的实现通过点分字符串 factory 可达。加新族 = 加这个 config + 它指向的 stage 模块 + 一个 deploy yaml + 测试 + 一条对齐记录。没有别的注册路径。

范围:该族必须能在用户本机(3050 4GB 显存)跑起来。参考树内两个例子:`nanovllm_omni/models/minimind_omni/`(深,3-stage 音频)与 `nanovllm_omni/models/smolvla/`(薄,1-stage)。

## 文件地图

- `nanovllm_omni/models/<family>/` — 每族一个目录:
  - `pipeline.py` — 声明式 `PipelineConfig` + `register_pipeline(...)`,必须定义 `PIPELINE` 符号。
  - `__init__.py` — 只 re-export(`PIPELINE`、bundle/loader)。
  - 各 stage 模块 — `factory` / `process_input` 指向的 callable(`module:attr`)。
- `nanovllm_omni/config/registry.py` — `register_pipeline`、`resolve_pipeline_config`、`OMNI_PIPELINES`、`StageExecutionType`(closed set)、`_load_builtin_pipelines()`(每族加一行 import)。
- `deploy/<family>.yaml` — 运行时旋钮:顶层 `max_batch`、每 stage `default_sampling_params`。
- `nanovllm_omni/models/__init__.py` — re-export 该族 loader/bundle。
- `docs/aligned_interfaces.md` — 公共符号/行为有任何变化时记录于此。

## Step 0 — 塞得进 4GB 吗?

接线前先估模型体积;用户已选好模型就跳过。

- 粗估:`checkpoint 字节数 × 1.2` 必须 ≤ 4GB(权重 + 运行时开销)。
- 在线估内存(一行):
  `hf mem <repo_id>`(Hugging Face CLI 内存估算)。
- 装不下:量化(GGUF/AWQ/GPTQ)或换更小兄弟模型。registry 只看族名,不管格式。

把尺寸/量化结论报给用户选型——不要擅自替换模型。

## Step 1 — 建族目录

`nanovllm_omni/models/<family>/`:

1. **`__init__.py`** — 只 re-export 公共符号(仓库规则:`__init__` 只 re-export,实现在独立模块)。
2. **`pipeline.py`** — 契约本体。对着 `minimind_omni/pipeline.py` 抄形状:

```python
from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_FAMILY = "nanovllm_omni.models.<family>"

PIPELINE = PipelineConfig(
    name="<family>",
    stages=(
        StageConfig(
            stage_id=0,
            name="<stage_name>",
            kind=StageExecutionType.LLM_AR,   # closed set,见下
            factory=f"{_FAMILY}.<module>:_<stage>_factory",
            process_input=None,               # 或 "module:attr" 桥接
            input_sources=(),
            is_terminal=True,                 # 出产物的 stage
            final_output_type="audio",        # "audio" | "text" | "actions" | ...
        ),
    ),
    default_deploy_config_name="<family>.yaml",
    registration_handles=("<family>", "<hf repo id>"),
    hf_architectures=("<HFModelClass>",),     # 可选,layer-6 消歧
)
register_pipeline(PIPELINE)
```

3. **各 stage 模块** — 每个 `factory` / `process_input` 点分路径必须能解析(`StageConfig.__post_init__` 立刻解析,坏路径在注册时就炸,不在请求时才炸)。`process_input` 把上一 stage 输出转成这一 stage 输入;`None` = 直通。

**`StageExecutionType` 是 closed set**:`LLM_AR`("ar")、`LLM_GENERATION`("generation")、`DIFFUSION`("diffusion")、`CODEC`("codec")。把新族映射进现有成员(smolvla 就把 VLA 映射成 `LLM_GENERATION`)。加成员 = 契约变更:ticket + 改 locked closed-set 测试。

## Step 2 — 注册

`nanovllm_omni/config/registry.py` 的 `_load_builtin_pipelines()` 里加一行,让 side-effect 的 `register_pipeline` 跑起来:

```python
from nanovllm_omni.models.<family> import pipeline as _<family>_pipeline  # noqa: F401
```

## Step 3 — deploy 旋钮

`deploy/<family>.yaml`(文件名必须等于 `default_deploy_config_name`):

```yaml
max_batch: 2
stages:
  - name: <stage_name>
    default_sampling_params:
      temperature: 0.7
      max_tokens: 512
```

采样/资源默认值只放这里,不放 `pipeline.py`(deploy/topology 分离规则)。`merge_pipeline_deploy` 按 `name` 匹配,所以 yaml 的 `stages[].name` 必须等于 `StageConfig.name`。

## Step 4 — 公共 re-export(有 loader 才做)

有 loader/bundle 就把 `load_*_bundle` 挂进 `nanovllm_omni/models/__init__.py` 的 `__all__`(对着 `load_minimind_omni_bundle`)。纯注册无 loader(smolvla 那样)就跳过。

## Step 5 — 测试

加 `tests/test_<family>.py`,至少断言注册 + 形状:

- `resolve_pipeline_config("<family>")` 返回 config,每个 `registration_handles` 别名也能解析。
- stage kind 是 closed set 成员;terminal stage 的 `final_output_type` 与 `OmniRequestOutput.from_pipeline` 期望一致。
- 薄模板:`tests/test_smolvla.py`(注册 + `OmniBase` 路由)。深模板:minimind 的测试(bundle 装载、生成、WAV 字节)。

## Step 6 — 对齐记录

族改了任何公共符号/行为,就按仓库工作流更新 `docs/aligned_interfaces.md`(Pipeline 拓扑节)。

## Step 7 — 本地验证

1. import 冒烟:
   `python -c "import nanovllm_omni; from nanovllm_omni.config.registry import resolve_pipeline_config; assert resolve_pipeline_config('<family>')"`
2. 生成冒烟(真实产物,不是 mock)。音频族要能解码出 WAV:
   `python -c "from nanovllm_omni import Omni; r = Omni('<family>').generate(['hi'])[0]; print(r.multimodal_output); print(len(r.multimodal_output['audio'].wav_bytes()))"`
   `OmniRequestOutput` 字段:`request_id` / `outputs` / `multimodal_output` / `error`;按终态产物(`audio`/`text`/`actions`)调整断言。`Omni(...)` 的 model 参数直接用注册句柄(如族名或 HF repo id)。
3. 重模型下载/运行测试标 `smoke`,默认 `pytest` 不跑(`-m "not smoke"`)。

## Step 8 — 提交闸门

提交前跑 `ci-safe-delivery` skill 清单(ruff、black、非 smoke pytest、公共 import、compileall)。每个连贯改动单独提交、写清消息;检查过了才 push。拿去 WSL 跑实验是 `nanovllm-omni-wsl-loop` skill 的事,不归本 skill。

## 禁止清单(从 AGENTS.md 移到这里)

- 只用 stdlib:`@dataclass` 定义契约、`http.server` 做服务。**不加新依赖**。
- 禁 FastAPI、uvicorn、Pydantic、WebSockets、分布式执行、全双工流式。
- 永不修改 / import 只读参考仓库 `/Users/mcig/Projects/vllm-omni`;它只用来核对名字/签名/形状。
- 不抄 vllm-omni 代码;适配到本仓库的更小范围。
- 不加 `StageExecutionType` 成员(closed-set 测试锁 4 个名字),除非走 ticket。
- 任何故意破坏公共 API 的改动必须写进 ticket,并反映在测试/文档里。