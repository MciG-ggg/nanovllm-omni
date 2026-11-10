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
  `make_stage_runner` 与 `make_talker_runner`。

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

`nanovllm_omni/backend/__init__.py` 是注册表（`get_backend` / `register_backend` /
`available_backends`）。默认是 `torch`，且选中它必须让 pipeline 与接缝之前逐位相同。

## 工作量评估

| 工作项 | 行数 | 难度 | 说明 |
|---|---:|---|---|
| `backend/base.py`（协议 + 能力 + 注册表） | ~130 | 低 | 已落地 |
| `backend/torch_backend.py`（转发） | ~60 | 低 | 已落地，零行为变化 |
| `backend/mlx_backend.py`（骨架） | ~150 | 低 | 已落地，一次一序列 |
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
- **并非每个 backend 都有 talker 臂**：`MlxVlmBackend.make_talker_runner()` 在
  `backend/mlx_vlm.py` 中抛出 `NotImplementedError`；该 backend 用于没有 talker
  stage 的 SmolVLM，`max_batch=1`。
- **模型族与 backend 的组合有边界**：MiniMind thinker / talker 可走 `mlx`，SmolVLM
  有独立的 `mlx-vlm` 路径；Code2Wav 仍由 torch 实现，Mac 上走 MPS（ADR 0024）。
  MLX paged kernel 尚未接入 `MlxStageRunner`，不能据此声称所有 stage 都由 MLX 执行。
- **fork 的 torch AR backend 在 Mac 上跑不了**：triton 无 macOS wheel（见上）；
  Mac 上的 MiniMind thinker / talker 与 SmolVLM 分别走 MLX backend，Code2Wav 走 torch/MPS。
- `mlx_lm.generate.BatchGenerator` 自带一套增量批处理（`PromptProcessingBatch`
  支持 extend / filter / split）。本 backend 不用它——nanovllm-omni 自己的调度器才是
  被讲的东西。这是另一个值得比的实现，不是现成答案。

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
