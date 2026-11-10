# TASKS

还没做的事。每条带证据（`file:line` 或 ADR 章节），按「影响双轨可比性 > 教学价值 > 清理」排序。

最后更新：2026-11-10（20:15）。

`[x]` 已完成，`[~]` 部分完成，`[ ]` 未做。

## 一、Continuous batching

**现在的批是快照式的，步内加不了人。** 这是 ADR 0014 承诺「真 continuous batching」但
我们给不了的东西。

| 位置 | 现状 |
|---|---|
| `engine/stage_engine.py` | 基类 `step()` 一次服务一个请求 |
| `nanovllm_omni/models/minimind_omni/stage.py:321` | `ThinkerStage.step()` 覆写为 `list(self._pending)` 快照，decode 跑完才 `clear()` |
| `nanovllm_omni/models/minimind_omni/stage_runner.py:99` | `decode_minimind` 把整个生成循环包在一个函数里（`while not scheduler.is_finished()`） |
| `nanovllm_omni/engine/executor.py:79` | `run_many` 先把全部请求入队再 drain——**这是批能形成的唯一入口**，晚到的请求不与在跑的拼批 |
| `docs/adr/0012-batch-lifecycle.md` §4 | 明确记录：「晚到的请求不会和已经在跑的请求拼批（与参考实现默认 `request_batch_max_wait_ms == 0` 同义）」 |

参考实现 `vllm_omni` 的形状是「每个 stage 自己持有请求队列，批由那个 stage 的 scheduler
从队列里选」（ADR 0012 引）。我们做到了「stage 持有队列」，但 `step()` 是快照式，
不是每步重选。

### 待办

- [ ] **CB-1 把 decode 循环从 `decode_minimind` 里提出来，变成增量式 step**
  `decode_minimind` 现在是一个函数跑完整个生成。要做 CB，生成循环得活到
  `StageEngine.step()` 的粒度上：每一步返回增量，而不是最终输出。
  依赖：先有 CB-2。
  证据：`nanovllm_omni/models/minimind_omni/stage_runner.py:99`、`nanovllm_omni/models/minimind_omni/stage.py:321`

- [ ] **CB-2 `StageEngine.step()` 的语义从「快照式批」改成「增量式」**
  基类和所有子类都要改：`ThinkerStage`、`TalkerStage`、`SmolVLMStage`、
  `SmolVLAVlmStage`、`DiffusionRunner`、`MiniMindOmniCode2Wav`
  （6 个子类，`rg "StageEngine\)"` 可列全）。
  证据：`engine/stage_engine.py`、`models/*/stage.py`

- [ ] **CB-3 接上 `BatchKVCache.extend`——中途进来的新请求**
  MLX 侧 `extend` 会自动右对齐补齐新行（实测：`_idx` 取 max、每行自己的
  `offset`、`left_padding` 补差），但**目前没有调用方**。它就是 CB 最后一块拼图：
  新请求单独 prefill，再 extend 进正在跑的 cache。
  证据：`backend/mlx_backend.py` 的 `forward`（`is_prefill` 且 `self._cache` 已存在时走 extend）、
  ADR 0025「还没做到：真正的中途 join」一节

- [ ] **CB-4 退出的序列要能提前离场，不再让整批等最慢的**
  `filter_cache_rows` 已经能回收行（实测 `(4 → 3)`、`(3 → 0)`），但
  `ThinkerStage.step()` 的批是一次性定死的，退出的序列仍占着批。
  证据：`backend/mlx_backend.py::filter_cache_rows`、ADR 0025

- [ ] **CB-5 修 ADR 0014 的措辞**
  它现在写「它能给的是**真 continuous batching + 图捕获**」，但两边都给不了真 CB。
  在 CB-1..4 做完之前，这句话应该改成「有 CB 的零件，还没接线」。
  证据：`docs/adr/0014-dual-track-targets.md:13`

## 二、Offload

ADR 0018 定了三步，顺序是「权重 mmap → 量化 → KV cache SSD」。**三步都还没开始。**

### 待办

- [ ] **OF-1 权重 mmap + 按需分页**
  `safetensors` 本来就是 mmap 的，要做的是不要把整个权重 copy 进 RAM，让文件页留在
  page cache 里由 OS 换出。归属 serving 层，不碰 `engine/`。
  收益：模型可以比 11.84 GiB 工作集更大；代价：首触是一次磁盘读（TTFT 变差，吞吐不变）。
  证据：`docs/adr/0018-offload-three-steps.md`「第一步」

- [ ] **OF-2 量化**
  4× 权重压缩，换来「能装下更大的模型」。**必然破坏逐位 parity**，只能降到
  ADR 0011 的 Compatible 层。Mac 上 bitsandbytes 不可用，候选是 `torchao` /
  MLX 量化路径 / GGUF。**量化前后的性能结论不能混在一张表里。**
  依赖：OF-1 做完才知道内存到底差多少。
  证据：`docs/adr/0018-offload-three-steps.md`「第二步」

- [ ] **OF-3 KV cache offload 到 SSD**
  serving 场景真正值钱的那一个。每步换页是一次随机读 → 伤 TTFT 不伤吞吐。
  **上游 `vllm-metal` 也还没做完**（它的 What's next 里写着 "mature SSD-backed
  KV cache offloading"），所以要么自己写要么等上游。归属 serving 层，与 `engine/`
  的 paged KV 是两层。ADR 0018 说它不承诺进 12 月的交付范围。
  证据：`docs/adr/0018-offload-three-steps.md`「第三步」

- [ ] **OF-4 code2wav 的 MLX 版本**
  ADR 0024 决定 code2wav 保持 torch 实现、跑在 MPS 上，MLX 轨的吞吐结论里
  code2wav 单独按设备记账。**但 MLX 轨因此缺一族模型**，talker 的 MLX 版已经做完了
  （ADR 0023「talker 的 MLX 版」），code2wav 是最后一个缺口。
  依赖：OF-2（量化）。
  证据：`docs/adr/0024-code2wav-stays-on-torch.md`

## 三、更大模型的支持

现在的硬约束是「所有模型必须能在 RTX 3050 4GB 跑起来」（AGENTS.md）。ADR 0016 说
这个约束已经失效，ADR 0014 定了双轨。但代码里还有小模型假设没清。

### 待办

- [ ] **BM-1 清理写死的小模型假设**
  已知的三处：
  - `nanovllm_omni/backend/mlx_kernel/build.py:117`：`DecodeShape(num_heads=8, num_kv_heads=4, head_dim=96)`
    是 MiniMind-O 的形状
  - `deploy/minimind_omni.yaml`、`deploy/smolvlm.yaml`：所有 stage 的 `max_num_seqs: 1`
  - `nanovllm_omni/models/smolvlm/smolvlm.py:275`：SigLIP hidden (768) → SmolLM2 hidden (576) 的
    connector 是写死的
  证据：以上三处 `rg` 可复查

- [ ] **BM-2 重测四族显存，把 4GB 约束从代码里解掉**
  ADR 0006 记了四族在 3050 上的占用（minimind_omni 峰值 1086.7 MiB，smolvlm 1922.1 MiB，
  sd_turbo 3097.6 MiB，smolvla 1217.6 MiB）。`smolvla` 的 803 MiB 差异**至今未解释**。
  证据：`docs/adr/0006-four-family-vram-2026-10-30.md:77`（明说「尚未解释」）

- [ ] **BM-3 重新定图捕获策略**
  `max_capture_bs` 默认 1，因为和更大尺寸一起捕获会换归约顺序，bridge 行差最后几位，
  graph 与 eager 不再产生逐字节相同的音频。更大的模型需要不同的策略（分段捕获？）。
  依赖：BM-2。
  证据：`docs/adr/0012-batch-lifecycle.md` §6、`third_party/nano-vllm/nanovllm/config.py`
  的 `max_capture_bs` 注释

## 四、MLX 轨还没做完的

### 待办

- [ ] **MX-1 把 Metal kernel 接进 `MlxStageRunner`**
  `backend/mlx_kernel/` 有 352 行（`kernels.py` 168 + `build.py` 158），但
  **kernel 尚未接进 `MlxStageRunner`**。ADR 0023 明确记了这条「未做」。
  证据：`docs/adr/0023-mlx-track-order.md:101`、`docs/adr/0023-mlx-track-order.md`「未做」

- [ ] **MX-2 写 prefill 的 varlen kernel**
  ADR 0021 的自研最小集里，decode 已做、prefill 没写。ADR 0023「未做」明列。
  证据：`docs/adr/0023-mlx-track-order.md`「未做」

- [ ] **MX-3 `MlxVlmBackend` 只能单序列，要能批**
  `max_batch=1`（`nanovllm_omni/backend/mlx_vlm.py:220`），单序列没有调度可言。
  ADR 0025 边界里明确「不改 `MlxVlmBackend`」。
  依赖：CB-3（要有 join 机制）。
  证据：`nanovllm_omni/backend/mlx_vlm.py:220`

- [ ] **MX-4 抽 `MlxRunnerBase`，消掉两个 MLX runner 的重复**
  `mlx_vlm.py` 与 `mlx_backend.py` 有 14 个同名方法，`prepare_prefill` /
  `prepare_decode` / `prepare_sample` / `sample` / `set_context` / `reset_context`
  基本逐行一样。`mlx_vlm.py` 245 行里约一半是拷贝，能省 ~100 行。
  根因是按库切开而没按能力切——正确切法是按能力抽基类，子类只覆写 `__init__`
  （加载）和 `forward`（cache 类型不同）。
  证据：`backend/mlx_vlm.py:76-134` 对 `backend/mlx_backend.py:142-209`

- [ ] **MX-5 合并两处采样配方**
  `_apply_repetition_penalty` + top-p 在 `nanovllm_omni/models/minimind_omni/stage_runner.py:83`（`_apply_repetition_penalty`）
  与 `nanovllm_omni/backend/mlx_backend.py:327`（mlx 侧）各写了一遍。
  ADR 0021 说 token 级一致靠「采样参数与停止条件的精确对齐」，但现在对齐全靠人眼比对两份代码。
  这条直接关系到项目承诺的正确性。
  证据：以上两处

- [ ] **MX-6 消掉 4 处 `if backend == "mlx"` 分支**
  `nanovllm_omni/models/minimind_omni/stage.py:248`（model_class 与 model_path）、
  `nanovllm_omni/models/minimind_omni/stage.py:353`（调哪个 decode）、`nanovllm_omni/models/minimind_omni/stage.py:498`（talker 的 model_path）、
  `nanovllm_omni/models/smolvlm/stage.py:217`（`MlxVlmSupport` 还是 transformers processor）。
  ADR 0013 选方案 2 的理由原文是「真按第 1 种画，最后每个 stage 里都会长出
  `if backend is mlx:` 的分支，比第 2 种更贵」——但分支还是长出来了。
  解法：把判断收进 `Backend` 协议（`Backend.resolve_model_path(args)`、
  `StageRunner.decode(...)`）。
  证据：以上四处

- [~] **MX-7 补 ADR 0023 里没做的测量** —— 部分完成（`0ce7896`）
  - [x] 「8 个独立池各自 rebind」的测量（ADR 0023 说这是「本轮最大的测量缺口」）
  - [ ] trace 次数没测

- [ ] **MX-8 MLX 轨缺的模型族**
  ADR 0014 边界：talker、code2wav（Mimi）、SD-Turbo（diffusers）、SmolVLA（lerobot）
  没有 MLX 等价物。talker 已补（ADR 0023），code2wav 见 OF-4。
  SD-Turbo 与 SmolVLA 还没有 MLX 方案。
  证据：`docs/adr/0014-dual-track-targets.md`「边界」

## 五、文档漂移

已经做了但文档没跟上。改代码时顺手改，别单开一轮。

- [x] **DR-1 ADR 0013 的 backend 表没提 `mlx-vlm`** —— 已修（`0ce7896`）
  补了 `mlx-vlm` backend、`make_talker_runner`、模型族与 backend 的组合边界。

- [x] **DR-2 ADR 0013 的工作量表已过期** —— 已修（`0ce7896`）
  标了部分完成并指向 ADR 0025。

- [ ] **DR-3 ADR 0012 的 bs=8 结论要在 MLX 轨重测**
  ADR 0014 明说「ADR 0012 的『bs=8 是净亏』是 CUDA 轨的结论；MLX 轨上 talker 仍然
  没有实现，所以 MLX 轨的 CB 数字要重新测，不能引用」。
  talker 的 MLX 版已经补上（ADR 0023），这条现在可做了。
  证据：`docs/adr/0014-dual-track-targets.md`「后果」

- [ ] **DR-4 ADR 0006 的 `smolvla` 803 MiB 差异未解释**
  ADR 0006 自己标了「尚未解释」。
  证据：`docs/adr/0006-four-family-vram-2026-10-30.md:77`

## 六、环境阻塞（不是代码问题，但挡住了好几件事）

- [ ] **EN-1 air-1 上还没有 nanovllm-omni 项目和 minimind-3o 权重**
  air-1（A800×8）是 CUDA 验证的唯一可用机器，有 rxk env（torch 2.8.0 + flash_attn
  2.8.3 + triton 3.4.0 + vllm 0.11.0），但没有项目目录也没有权重。
  需要 rsync 项目过去 + 从 air-2 中转权重（air-1 到 HF 直连不可达）。
  **挡住了**：talker P1/P2 parity、窗口化 KV 的 stage 级验证、bs=8 重测、
  MLX vs CUDA 的 greedy 对比。
  证据：2026-11-09 的环境查证记录

- [ ] **EN-2 WSL 上跑不了 talker parity（没有 flash_attn）**
  fork 的 attention 顶层 `import triton` + `flash_attn_varlen_func`，WSL 上没有
  flash_attn，fork 的 attention import 不了。装它要编译（CUDA 12.8 + py3.12），
  4GB 卡上不现实。ADR 0014 又规定 parity 基线不能跨机器。
  证据：2026-11-09 的环境查证记录

- [ ] **EN-3 文档里引用了 `/home/mcig/...` 这种 Linux 路径**
  `tests/test_optim_bench.py` 等测试在 Mac 上因为这个失败。全量 pytest 的失败集合里
  有这一类，是环境问题不是代码问题，但会让 CI 判红。
  证据：`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`
