# ADR 0028: 第三轨——昇腾 NPU

- Status: Accepted
- Date: 2026-11-11
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0015（MLX 要 varlen）、
  0023（MLX 轨顺序）、0026（增量 step）、0027（MLX attention 三臂实测）

## 问题

ADR 0014 把轨道数定在两条：3050（CUDA）与 Mac（MLX）。两条都不覆盖昇腾 NPU。
现在要在租用的 910B 上再跑一条轨。教学点不是「又支持一个后端」，而是
**backend 接缝在 torch 方言内的第三个使用者**：调度层框架无关，模型执行层
每换一个加速器都要重换一部分。

要回答三个问题：

1. NPU 轨是第三条长期轨，还是接缝的一次性验证？——决定范围与文档承诺。
2. `torch_npu` 是 torch 的设备扩展，fork 的 CUDA 假设有多深？
3. 没有真卡时，代码与结论的边界在哪？

## 关键事实

| 事实 | 证据 |
|---|---|
| backend 接缝已在，画在「一步执行」 | ADR 0013；`nanovllm_omni/backend/base.py:56-193` |
| NPU 属于 torch 方言，不是另一套框架 | `torch_npu` 在 `torch` 上提供 `torch.npu` 设备；与 MLX 的处境不同 |
| fork 的 device 硬编码只有三处 | `third_party/nano-vllm/nanovllm/engine/model_runner.py:109-113`（`nccl`、`torch.cuda.set_device`、`set_default_device("cuda")`） |
| attention 模块顶层 `import triton`，KV 写入无回退 | `third_party/nano-vllm/nanovllm/layers/attention.py:3-4, 196-198`；SDPA 只回退了计算 |
| 构造后遍历模块并改写属性是 fork 既有做法 | `third_party/nano-vllm/nanovllm/engine/model_runner.py:86-90` |
| 上游 vllm-omni 用 platform 层按硬件选择类 | `vllm_omni/platforms/__init__.py:65-78, 132-168`；`vllm_omni/engine/stage_init_utils.py:371-389` |
| 上游 NPU 复用厂商栈，只换 worker / model runner | `vllm_omni/platforms/npu/worker/npu_ar_worker.py:16-25` |
| 上游 NPU 用 unpadded / contiguous KV | `vllm_omni/platforms/npu/platform.py:49-58`；`tests/platforms/npu/test_contiguous_kv_cache.py:86-89` |
| bench 的设备信息只有 CUDA 分支 | `nanovllm_omni/bench/env.py:28-50`；`nanovllm_omni/bench/runner.py:97-137` |
| 现有测试用 mock 锁注册表与懒加载 | `tests/test_backend_wiring.py:51-86` |

## 决定

第三轨**是长期轨**，但只承诺 MiniMind thinker 一个族。其余三族是第三方库
（`transformers` / `diffusers` / `lerobot`）的路径，移植不归本项目。

1. **基线硬件**：910B。910C 只记录「顺带能跑」，不进基线。CANN 与 `torch_npu`
   版本在 runbook 里钉死，首次实验后回填。
2. **backend 命名与选择**：`ascend`。`torch.npu.is_available()` 为真时自动选它；
   其他环境默认仍是 `torch`，3050 / Mac 的行为不变。
3. **代码归属**：fork 的 `ModelRunner` 做最小 device 参数化（`nccl`→`hccl`、
   device 跟随、显式 `num_blocks`）；attention 通过 `model_class` 注入
   `attention_cls`，新类放 `nanovllm_omni/backend/ascend_attention.py`。
   不做构造后替换，不在 fork 里写 `if is_npu`。
4. **KV 布局**：NPU 用连续 KV，kernel 不读 block table；调度器的 block manager
   照常跑（教学骨架不变）。先例是上游的 `apply_contiguous_kv_patch`。
5. **计算算子**：`torch_npu.npu_fusion_attention` 默认，varlen 用
   `actual_seq_lengths`；SDPA 作回退。精度 bf16。不做显存自动估算：
   `num_blocks` 由 backend 按 `max_model_len × max_num_seqs` 推导。
6. **里程碑**：(1) bs=1 prefill+decode 跑通，出 TTFT / 吞吐 / 显存；
   (2) varlen + continuous batching。graph 不进计划。
7. **测试**：`pytest.importorskip` 守门 + 注册表 / 懒加载锁；不写假
   `torch_npu` stub——假契约比不测更糟。
8. **profiling**：按 ADR 0001 加 NPU 变体（`torch.npu.Event` /
   `synchronize` / `max_memory_allocated`、`npu-smi info`）。该变体已写入
   ADR 0001。
9. **结果文档**：`docs/perf/minimind-thinker-ascend-<date>.md`；原始 CSV 与
   trace 按 ADR 0001 的口径留在本地。

## 后果

- 三条轨共享同一份调度器与 paged KV 教学骨架；NPU 的差异被压到 backend 与
  attention 两层。这正是 ADR 0013 接缝的第三个真实使用者。
- fork 需要一次 device 参数化，3050 基线必须复测逐位不变（WSL）。
- `bench/env.py` 目前把非 CUDA 主机判成 `"cpu"`；`bench_minimind.py` 见到
  `"cpu"` 会短路写零行（`nanovllm_omni/bench/bench_minimind.py:144-152`）。
  第三轨出数前必须补 NPU 的设备信息 / 计时 / 显存三条路径；在此之前
  用 `python -m nanovllm_omni.bench time` 的 thinker 路径，不要走
  `bench_minimind` 入口。
- `pyproject.toml` 现有 `cuda` / `mlx` 两个 extra，没有 `npu`。首次实验按
  runbook 手动安装 `torch_npu`；是否新增 `npu` extra 等首次实验后再定。
- 文档承诺多一条轨；没有真卡时不得写任何性能数字。

## 边界

- **不做 CUDA parity**：跨厂商浮点归约顺序不同；MLX 轨已把判据从逐位改成
  token 级三档（ADR 0021 / 0027），NPU 轨连 token 级一致都不承诺。
- **没有真卡不出数字**：代码可以先写、本地测试可以绿，但 TTFT / 吞吐 / 显存
  必须来自 910B 实测。
- **只承诺 MiniMind thinker**；talker、code2wav、SmolVLM、SD-Turbo、SmolVLA
  不在范围。
- **graph 不进计划**：NPU 侧对应 ACL graph，是否值得要实测后再开 ADR。
- **不跨机相减**（ADR 0014）：3050 与 910B 的数字只并列；精度也不同
  （fp16 vs bf16），结果文档要写明。

## 未验证

- `torch_npu.npu_fusion_attention` 在 910B 上的 varlen 参数与 decode 单 query
  行为；GQA 头数是否原生支持（否则要 `repeat_interleave`）；`sparse_mode=3`
  在 TND prefill 的因果语义是否与 fork 的 SDPA 一致。
- fork 的 `allocate_paged_kv_cache` 仍会给 `AscendAttention` 赋 paged 张量；
  本类忽略并自建连续缓存，但显存被占了一份，上卡确认可接受或改为跳过。
- CANN / `torch_npu` 的具体可用版本与镜像。
- `torch.npu.Event` / `synchronize` / `max_memory_allocated` 是否与 CUDA API
  一一对应。
- 连续 KV 相对 paged 在 NPU 上的实测差异：上游只给了「padded page 回退」的
  方向，没有我们的尺寸。
- `import triton` 在 NPU 环境是否可用（triton-ascend）；不可用时 fork 的
  模块级 import 是否需要 try/except。
- NPU 轨是否需要 token 级判据；MLX 的三档判据是否适用。
