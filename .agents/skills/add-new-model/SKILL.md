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
- **`hf mem` 估的是权重,不是推理峰值**。实测峰值常是估值的 1.5–6 倍(激活、attention、VAE decode 都加进去,Sana 实测 8.81 GB vs 估 ~1.2 GB)。**真测一遍**再下结论:
  ```python
  import torch
  from diffusers import <PipelineClass>
  pipe = <PipelineClass>.from_pretrained(
      "<repo_id>", torch_dtype=torch.float16, variant="fp16", local_files_only=True,
  ).to("cuda")
  torch.cuda.reset_peak_memory_stats()
  pipe("warmup", num_inference_steps=1)
  print(f"peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
  ```
  峰值 ≤ 3.5 GB 才能放心说"4 GB 卡可跑"。

把尺寸/量化结论报给用户选型——不要擅自替换模型。

## Step 0.5 — 模型下载路径(网络现实)

`hf download` / `huggingface.co` 直连**经常不可达**(GFW、企业代理、断流)。下载前先 HEAD 测候选端点,选一个能稳定的:

1. **HEAD 测试**(2 秒一个,带超时):
   ```bash
   for url in \
     "https://huggingface.co/<repo>/resolve/main/model_index.json" \
     "https://hf-mirror.com/<repo>/resolve/main/model_index.json" \
     "https://www.modelscope.cn/<owner>/<repo>/resolve/main/model_index.json"; do
     timeout 8 curl -fsI --connect-timeout 5 "$url" 2>&1 | head -1 | sed "s|^|$url |"
   done
   ```
2. **稳定路径优先级**:
   - HF 直连:偶发可用,但大文件常 TLS reset;**最快但不稳**。
   - hf-mirror.com:经常 302→HF,然后再卡;**不推荐**。
   - **ModelScope CDN**(`www.modelscope.cn/<owner>/<repo>/resolve/main/...`):中国网络稳定,~5–10 MB/s。Stability-AI / NVIDIA / Efficient-Large-Model 都有镜像。**首选**。
3. **下载工具**:`wget -c`(单流,断点续传)在该 CDN 上比 `aria2c -x16` 稳——多连接在该 CDN 上会触发 SSL 握手失败。
4. **大模型拆文件下**:列出 `repo` 所有子目录(`text_encoder/`、`unet/`、`vae/`...)分别 HEAD 拿 Content-Length,只下需要的;fp32/fp16 双份常占 ~2 倍磁盘,**只下 fp16 variant** 即可。
5. **下载完成后同步到 jingrui/WSL 跑**:本机 `tar -cf - -C <local> . | ssh <host> 'tar -xf - -C ~/models/<target>/'` 绕开 macOS `openrsync` 与 Linux rsync 的版本不兼容。

## Step 1 — 建族目录

命名规则先看 `AGENTS.md` 的「命名约定」节(`num_*` 优先、全词优先;远端模型属性名不动)。

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

**diffusers 族(`StageExecutionType.DIFFUSION`)的 stage 模板**:

- `from_pretrained(..., torch_dtype=torch.float16, variant="fp16")` —— **不要省略 `variant`**。diffusers 仓库的 `diffusion_pytorch_model.safetensors` 是 fp32 权重;某些模型(Sana DiT 等)fp32 文件本身有数值问题,cast 到 bf16 产出噪声而非图。**`variant="fp16"` 加载 fp16 变体文件**,且支持 `local_files_only` 模式下的 per-component 缺失 fallback。
- 默认 `local_files_only=True`;走 `OmniEngineArgs.extra["allow_hf_download"]` opt-in 出网。
- `forward(payload, sampling)` 里所有私有旋钮(`num_inference_steps` / `guidance_scale` / `height` / `width` / `num_images_per_prompt` ...)都从 `sampling.extra` 读,**不要扩 `SamplingParams` dataclass**(那是顶层契约)。
- 蒸馏模型(SD-Turbo / LCM / Sana-Sprint)**必须 `guidance_scale=0.0`**:CFG > 0 立刻毁图。stage 默认锁 0.0,deploy yaml 里也写死 0.0。
- 检查 `args.dtype`:只接受 `float16`/`bfloat16`/`float32`;非 float dtype(例如 SmolVLA 的 `int8` 量化路径)直接 `ValueError`,免得错传到扩散 stage 里。

4. **`hf_architectures` 怎么填**:在 `<repo>` 的 `model_index.json` 里查 `_class_name` 字段(SD-Turbo → `"StableDiffusionPipeline"`,Sana → `"SanaPipeline"`)。这是 diffusers 索引管线的 key,用于本地 snapshot 路径不含族名时的层-6 路由消歧。

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
- **import-order-agnostic 守卫**:`monkeypatch` 掉 `builtins.__import__` 同时拦截 `diffusers` 和 `torch`,验证 stage 在重依赖缺失时抛 `ImportError`(不然 CI 在无 torch 机器上跑会因 import 顺序 flake)。

## Step 6 — 对齐记录

族改了任何公共符号/行为,就按仓库工作流更新 `docs/aligned_interfaces.md`(Pipeline 拓扑节)。

**如果是替换既有族(rev1 → rev2)**:GitHub issue 的 title + body 也要同步(`gh issue edit <n>` 不可逆,改之前草稿准备好)。正文里写明 model pivot 原因、新 commit 引用、acceptance criteria 勾选状态。本地命令行是 `rtk gh issue edit <n> --title "..." --body-file /tmp/issue.md`(项目里 `rtk` 是 gh wrapper,不带 `--quiet`)。

## Step 7 — 本地验证

1. **import 冒烟**:
   ```bash
   python -c "import nanovllm_omni; \
     from nanovllm_omni.config.registry import resolve_pipeline_config; \
     assert resolve_pipeline_config('<family>')"
   ```
2. **生成冒烟(真实产物,不是 mock)**:
   - 音频族要能解码出 WAV;图像族要能写出 PNG;动作族要拿到非空数组。
   - 命令模板(图像族,real-weight smoke):
     ```bash
     PYTHONPATH=. python examples/offline_inference/<family>/run.py \
         --model <repo_or_local_snapshot> --prompt "<prompt>" --output /tmp/<family>_smoke.png
     ```
   - `OmniRequestOutput` 字段:`request_id` / `outputs` / `multimodal_output` / `error`;按终态产物(`audio`/`text`/`actions`)调整断言。`Omni(...)` 的 model 参数直接用注册句柄(如族名或 HF repo id)。
3. **VRAM 实测 + 释放模式**:
   - **CLI demo**(`run.py`)一次性进程退出,VRAM 由 `cudaFree()` 自动归零;**不需要手动清理**。
   - **长进程测试**(REPL / notebook / 跨 stage pipeline 里反复 `generate(...)`):每次后显式释放:
     ```python
     out = omni.generate(...)[0]
     del omni
     import gc; gc.collect()
     torch.cuda.empty_cache()
     ```
   - **三时刻验证**(防止"以为释放了其实还占着"):
     ```bash
     nvidia-smi --query-gpu=memory.used --format=csv,noheader -i <gpu_idx>  # T0 跑前
     # ... run script ...
     nvidia-smi --query-gpu=memory.used --format=csv,noheader -i <gpu_idx>  # T5 跑后
     ```
     T5 应回到 T0 的基线(几十 MiB,系统 CUDA context 占的)。Sana 实测:T0 37 MiB → T3 8.81 GB → T5 37 MiB。
4. **`run.py` 模板必须显式传 `deploy_config_path`**(否则 demo 受 CWD 影响,在 jingrui/WSL 复制即挂):
   ```python
   from pathlib import Path
   _REPO_ROOT = Path(__file__).resolve().parents[3]  # examples/offline_inference/<family>/run.py -> repo root
   _DEPLOY_YAML = _REPO_ROOT / "deploy" / "<family>.yaml"
   omni = Omni(args.model, device=args.device,
               extra={"deploy_config_path": str(_DEPLOY_YAML), "allow_hf_download": False})
   ```
5. **重模型下载/运行测试标 `smoke`,默认 `pytest` 不跑(`-m "not smoke"`)**。如果本机没权重,smoke 跳过——靠 `run.py` 手动验证,不强行 mock 假图。
6. **demo 跑完清 artifact**:测试用 PNG / 临时下载缓存(`*.incomplete` 之类的)/ 大模型复本(fp32 副本若只下 fp16 variant 即可),跑完确认盘上没多余文件。Sana 时本机留了 8.3 GB 复本,删掉才舒服。

## Step 8 — 提交闸门

提交前跑 `ci-safe-delivery` skill 清单(ruff、black、非 smoke pytest、公共 import、compileall)。每个连贯改动单独提交、写清消息;检查过了才 push。拿去 WSL 跑实验是 `nanovllm-omni-wsl-loop` skill 的事,不归本 skill。

## Step 9 — 移除族(rev 替换场景)

如果做 rev 替换(rev1 不可用,改成 rev2),新族落地后**旧族**的产物一并清理:

- `git rm -r nanovllm_omni/models/<old_family>/ deploy/<old_family>.yaml examples/offline_inference/<old_family>/ tests/test_<old_family>.py`
- `nanovllm_omni/config/registry.py` `_load_builtin_pipelines()` 里删对应 import 行
- `nanovllm_omni/models/__init__.py` `__all__` 删对应 loader(如有)
- `docs/aligned_interfaces.md` 改 `<old_family>` 行 → `<new_family>` 行
- **不要 `git rebase -i` squash 旧 commit**:历史里保留 rev1 探索对下一轮 pivot 是有用的;一次普通 `feat(...): add new family; remove old family` commit 即可,git 会自动识别 `tests/test_<old>.py -> tests/test_<new>.py` 重命名。
- 物理删除模型快照:本机 `rm -rf ~/models/<old_snapshot>` + 远端(WSL/jingrui)同样清理(常 5–10 GB)。
- 关联的 demo artifact(PNG / JSON / 临时下载)也清掉。

## 禁止清单(从 AGENTS.md 移到这里)

- 只用 stdlib:`@dataclass` 定义契约、`http.server` 做服务。**不加新依赖**。
- 禁 FastAPI、uvicorn、Pydantic、WebSockets、分布式执行、全双工流式。
- 永不修改 / import 只读参考仓库 `/Users/mcig/Projects/vllm-omni`;它只用来核对名字/签名/形状。
- 不抄 vllm-omni 代码;适配到本仓库的更小范围。
- 不加 `StageExecutionType` 成员(closed-set 测试锁 4 个名字),除非走 ticket。
- 任何故意破坏公共 API 的改动必须写进 ticket,并反映在测试/文档里。