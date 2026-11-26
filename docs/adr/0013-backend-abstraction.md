# ADR 0013: stage 层的 backend 接缝（torch / MLX）

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0004（模型与引擎边界）、ADR 0005（图策略在 stage）、0010（并发不是杠杆）、
  0012（请求生命周期与批处理）

## 问题

stage 层隐式绑在 torch + CUDA 上，而且这个绑定比看上去深：

- `third_party/nano-vllm/nanovllm/layers/attention.py` 在**模块顶层** `import triton`。
  triton 3.9.0 在 PyPI 上的 wheel 只有 `manylinux x86_64` 与 `manylinux aarch64`，
  **没有 macOS / arm64**。所以 fork 的 attention 在 Mac 上连 import 都不成立。
- `models/minimind_omni/stage_runner.py::decode_minimind` 与
  `models/smolvlm/stage.py` 直接摸 `stage_runner.model_runner`，
  调 fork 的 `prepare_prefill` / `prepare_decode` / `prepare_sample` / `run_model`，
  并就地替换 `model_runner.sampler`。

目标机器是 Mac（M2 Air 16 GB 统一内存）。要让它跑起来，得先把"框架相关"的部分
圈出来。

## 关键事实（实测）

| 事实 | 证据 |
|---|---|
| **调度层与框架无关** | fork 的 `engine/scheduler.py`、`engine/sequence.py`、`engine/block_manager.py`、`sampling_params.py` **torch 命中全部为 0**。批的形成、slot 复用、抢占、chunked prefill 都在纯 Python 里 |
| **本仓库的调度层也与框架无关** | `engine/stage_engine.py` / `engine/runner.py` / `engine/executor.py` **torch 命中全部为 0** |
| **模型层深度绑定 fork** | `models/minimind_omni/thinker.py` 的 `MiniMindThinker` 用 fork 的 `VocabParallelEmbedding`、带 `packed_modules_mapping` 的 `ThinkerAttention`、fork 的 `RMSNorm` |
| **另两个族是第三方库** | `sd_turbo` 用 diffusers，`smolvla` 用 lerobot；两者都没有 MLX 等价物 |
| **MLX 有 graph 等价物** | `mx.compile` 存在，实测 4.91×（512×512、20 轮算子链：eager 0.108 ms/iter → 0.022 ms/iter） |
| **MLX 没有 varlen** | `mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask=None\|str\|array, sinks, force_fused, stream)` 有 mask、没有 `cu_seqlens`。上游 `vllm-metal` 为此自己写了 Metal kernel，其 benchmark 显示 padded 路径在 ragged 批上 +143% |
| **MPS 与 MLX 看到同一个内存池** | `torch.mps.recommended_max_memory()` = `mx.device_info()["max_recommended_working_set_size"]` = 12713115648（11.84 GiB） |

## 决定：接缝画在「一步执行」，不画在 tensor API

抽象层有两条常见的画法：

1. **tensor API 层**——把 `torch.xxx` 换成 `backend.xxx`，一套 stage 代码跑两个后端。
2. **模型实现层**——框架无关的调度 + 框架相关的模型执行。

选 **2**。理由：MLX 不是 torch 的方言。`torch.nn.functional.scaled_dot_product_attention`
与 `mx.fast.scaled_dot_product_attention` 的 mask 语义、RoPE 的写法、KV cache 的生命周期
都不一致；真按第 1 种画，最后每个 stage 里都会长出 `if backend is mlx:` 的分支，
比第 2 种更贵。

这也正是上游 `vllm-metal` 的做法：upstream vLLM 供调度器 / paged KV / 前端，
`mlx_lm` 供 token-wise 的模型层，插件只补 request-aware 的那一层。

### 接口

`nanovllm_omni/backend/base.py` 定义两个协议：

- `StageRunner`——一步 AR 执行：`prepare_prefill` / `prepare_decode` / `prepare_sample` /
  `forward` / `sample` / `bridge_hidden` / `model` / `config`，加 `set_context` /
  `reset_context`。进出边界上的 `Sequence` 是 fork 的纯 Python 对象，
  跨边界的只有 batch 张量与 logits。
- `Backend`——`name` / `device` / `dtype` / `capabilities`，加 `make_config`、
  `make_stage_runner`。

`set_context` / `reset_context` 在协议里，是因为 stage 会调：fork 把每步的状态放在
模块级全局里，MLX 放在 cache 对象里。stage 不该知道是哪一种。

`config` 也在协议里：stage 在建 runner **之前**往 config 上写 graph 相关的旋钮
（`enforce_eager` / `max_capture_bs` / `graph_max_seq_len`），所以每个 backend 的
config 都得是可写的普通对象。

`BackendCapabilities` 三项：`varlen`（能否一个 forward 承载 ragged 序列）、
`graph`（有没有 graph capture 等价物）、`max_batch`。

### 实现

| backend | 状态 | 说明 |
|---|---|---|
| `torch` | 可用，行为不变 | 转发 fork 的 `ModelRunner` / `Sampler`；`varlen=True`、`graph=True` |
| `mlx` | 可用（thinker-only）；单序列 + dense batch | 用 `mlx_lm.utils.load` 加载；`varlen=False`、`graph=True`（`mx.compile` 未用）、`max_batch=None`（多序列左补齐）。数字见 ADR 0023 |
| `mlx-vlm` | 可用；单序列 | SmolVLM 的 MLX 轨，代码在 `models/smolvlm/vlm_mlx.py`，注册表只记名字 |

`nanovllm_omni/backend/__init__.py` 是注册表（`get_backend` / `register_backend` /
`available_backends`）。默认是 `torch`，且选中它必须让 pipeline 与接缝之前逐位相同。

**修订（2026-11-11）：`backend/` 只放接缝，不放模型族代码。**
talker 契约（`TalkerRequest` / `TalkerRunner` / `make_talker_runner`）是 MiniMind
的形状——bridge 行进、八个码本出，只有 MiniMind 有 talker stage——所以它连同
`TorchTalkerRunner` / `MlxTalkerRunner` 一起搬进 `models/minimind_omni/`，并从
`Backend` 协议里删掉；`backend/base.py` 与 `backend/torch_backend.py` 不再 import
任何模型族。SmolVLM 的 MLX 轨同理搬到 `models/smolvlm/vlm_mlx.py`，注册表按名字
懒加载它。

## 工作量评估

| 工作项 | 行数 | 难度 | 说明 |
|---|---:|---|---|
| `backend/base.py`（协议 + 能力 + 注册表） | ~130 | 低 | 已落地 |
| `backend/torch_backend.py`（转发） | ~60 | 低 | 已落地，零行为变化 |
| `backend/mlx_backend.py` + `backend/mlx_runner.py`（骨架） | ~150 | 低 | 已落地，一次一序列 |
| `engine/stage_runner.py` 补 4 个转发方法 + `sample(sampler=)` | ~40 | 低 | 已落地 |
| `decode_minimind` 去 fork 耦合（7 处直接访问） | ~30 改 | 低 | 已完成：全部改走协议，vendor sampler 按调用传入 |
| `smolvlm/stage.py` 去 fork 耦合 | ~20 改 | 低 | 已完成 |
| `ThinkerStage` / `SmolVLMStage` 走 backend（默认 torch） | ~40 改 | 低 | 已完成；`extra["backend"]` 选后端，默认路径与接线前一致 |
| **MLX backend：paged KV + 批 decode** | ~400 | 中 | 未做；`mlx_lm.generate.BatchGenerator` 自带增量批处理，可作参照 |
| **MLX varlen attention（Metal kernel）** | — | **高** | 未做；`vllm-metal` 的整个插件就为了这个 |
| 测试 + 文档同步 | ~250 | 低 | 已落地：`tests/test_backend_wiring.py` 锁注册表、懒加载、协议一致性与 stage 不再摸 `model_runner` |

**结论：接缝本身很便宜（约 300 行），成本 90% 在 varlen attention kernel 上。**
而那个 kernel `vllm-metal` 已经写了（license 待查）——能复用就从"写 kernel"降到"接 kernel"。

## 边界（明确没做到的）

- **MLX backend 没有 varlen**：多序列走左补齐 + `BatchKVCache`，补齐按最长成员
  付费，整批跑到最慢成员结束。这是 ADR 0015 拒绝的过渡形态；真 CB 要等
  `docs/dev/mlx-varlen-kernel-interface.md` 的 kernel 落地。
- **MLX backend 没有 bridge hidden**：MiniMind-O 的 thinker → talker 桥在 MLX 侧不存在。
  > 已被 ADR 0023「talker 的 MLX 版」及「thinker stage 在 MLX 上跑通」取代（2026-11-10）：
  > `MlxStageRunner.bridge_hidden()` 现在返回 bridge hidden 行。
- **MLX backend 的首轮实测是 thinker-only**：后续已增加 MiniMind talker 的 MLX runner
  与 SmolVLM 的 `mlx-vlm` 路径，详见 ADR 0023。首轮 thinker 数字与限制仍见
  `docs/perf/minimind-thinker-mlx-2026-11-09.md`；MLX 与 CUDA 的 token 级一致仍未验证。
- **talker 不再是 backend 的臂**（2026-11-11 修订）：talker 契约归 MiniMind 族，
  `Backend` 协议不再有 `make_talker_runner`；SmolVLM 没有 talker stage，
  `mlx-vlm` 自然也没有。
- **模型族与 backend 的组合有边界**：MiniMind thinker / talker 可走 `mlx`，SmolVLM
  有独立的 `mlx-vlm` 路径；Code2Wav 仍由 torch 实现，Mac 上走 MPS（ADR 0024）。
  MLX paged kernel 尚未接入 `MlxStageRunner`，不能据此声称所有 stage 都由 MLX 执行。
- **fork 的 torch AR backend 在 Mac 上跑不了**：triton 无 macOS wheel（见上）；
  Mac 上的 MiniMind thinker / talker 与 SmolVLM 分别走 MLX backend，Code2Wav 走 torch/MPS。
- `mlx_lm.generate.BatchGenerator` 自带一套增量批处理（`PromptProcessingBatch`
  支持 extend / filter / split）。本 backend 不用它——nanovllm-omni 自己的调度器才是
  被讲的东西。这是另一个值得比的实现，不是现成答案。

## 上游对照（2026-11-11）

vllm 与 vllm-omni 处理「不同 backend」的方式，逐条读过源码：

| 上游做法 | 证据 | 对我们的含义 |
|---|---|---|
| 平台按硬件分文件/分目录，一个平台一个对象 | `vllm/platforms/{cuda,rocm,cpu,xpu,tpu}.py`；`vllm_omni/platforms/{cuda,rocm,npu,xpu,musa}/platform.py` | 我们的 `backend/` 就是这个位置 |
| 平台对象负责探测与选择 | `vllm_omni/platforms/__init__.py`：探测函数 + 懒初始化 `current_omni_platform` + 外部 plugin group | 我们的 `get_backend(name)` 是它的简化版：只按名字选，不探测 |
| stage 不 import 硬件代码，只向平台要**类名字符串** | `OmniPlatform.get_omni_ar_worker_cls()` / `get_omni_generation_worker_cls()` / `get_default_stage_config_path()` | 我们的 `Backend.make_config` / `make_stage_runner` 是同一形状的简化版 |
| 设备查询与计时在平台对象上，上层不直接摸 `torch.cuda` | 平台对象提供 `get_torch_device` / `synchronize` / `record_device_event` / `get_free_memory` / `get_profiler_cls` / `get_graph_wrapper_cls` | **我们还没做到**：`models/minimind_omni/stage_runner.py` 的 decode 循环直接 `torch.cuda.synchronize()`，并按 backend 名字分支 |
| 平台可以主动放弃一个能力 | NPU 换掉 `requires_contiguous_pa_kv_cache`（`vllm_omni/platforms/npu/platform.py`），注释写明 padded pages 会性能回退 | 我们的 `BackendCapabilities.varlen=False` 是同一形状 |
| 平台特有的模型差异放在平台目录，且是**对同一份模型的 patch** | `vllm_omni/platforms/npu/models/voxcpm2_talker.py` 等 patch 掉 `vllm_omni/model_executor/models/` 里的家族模型 | 见下面的偏差 |
| backend 名字 → 类路径字符串，注册表按需覆盖 | `vllm/v1/attention/backends/registry.py` 的 `AttentionBackendEnum`；`vllm_omni/diffusion/attention/backends/registry.py` | 我们的 `get_backend` 是同一个形状：名字 → 类，平台可覆盖 |

**结论：上游的 seam 画在「平台 / worker 类」这一层，不是 tensor API**，和本 ADR
「决定」一节的选择一致。`Backend` 协议对应上游的平台对象，只是我们把
`make_stage_runner` 做成方法而不是类名字符串（我们没有独立 worker 进程，
没有跨进程惰性加载的必要）。

### 偏差：上游的平台都是 torch 方言

上游每个平台跑的都是 torch：NPU 的 worker 直接继承厂商栈 `vllm_ascend` 的
`NPUWorker`，只覆盖 profiler 和 model runner（`platforms/npu/worker/npu_ar_worker.py`）。
所以「同一份模型 + 换边界层」在它们那儿成立。

我们不成立：MLX 不是 torch 的方言（本 ADR「关键事实」一节：triton 没有 macOS
wheel、MLX 没有 varlen、mask 与 RoPE 的写法都不同），MiniMind talker 在 MLX 上是
**整份移植**（`models/minimind_omni/talker_mlx.py`），不是 patch。因此我们照抄上游
的目录形状时，MLX 轨的文件不能只「换边界层」，只能并行放一份。这条偏差记在这里，
免得以后拿上游的目录形状当标准来评判我们的布局。

同一条偏差还有一个具体后果：上游把平台特有的模型差异放在**平台目录**
（`platforms/npu/models/`），我们把 MLX 移植放在**家族目录**
（`models/minimind_omni/`）。选家族目录的理由是 MLX 轨是完整移植而不是 patch，
家族目录能让「这个族在这条轨上的全部实现」读在一起。代价是 `backend/` 里没有
按轨分的子目录；将来加第三条轨（例如 NPU）时这条规则要重新审。

### 未做：family 里的 backend 分支

`models/minimind_omni/stage_runner.py` 的 `ThinkerDecodeSession` 目前按
`self._backend` 分支（同步、采样、MLX cache 行过滤、bridge 收集）。上游的做法是
这些都归平台/worker：`npu_ar_worker.py` 覆盖 model runner，stage 不分支。我们
还没下沉，是下一步的活。

## 未验证

- **torch 路径的输出是否逐位不变**：本机没有 GPU，跑不了真实解码。
  依据是代码级等价：`_sampler_with_rp` 的调用参数、次序、次数不变；
  `stage_runner.forward` 内部仍是 `model_runner.run_model`；
  `bridge_hidden()` 内部仍是 `model.get_bridge_hidden()`；
  `torch.cuda.synchronize()` 的位置没动，stage_ms 口径不变。
  **需要在 WSL 上跑一次真实 MiniMind 解码确认。**
- **MLX 与 CUDA 的 token 级一致**（ADR 0021）：本机没有 CUDA 基线，且两边
  RNG 流不同。MLX 侧自己的单序列 / dense batch 数字已实测，见 ADR 0023。
- `vllm-metal` 的 Metal kernel 是否可复用、license 是否兼容（未查）。
- `mx.compile` 在真实模型上的收益（本机只测了 20 轮小算子链，4.91× 是这个量级的数，
  不是模型级的数）。
