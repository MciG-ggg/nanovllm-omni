# 昇腾 910B 第三轨 runbook

本文件是 ADR 0028 的操作步骤。设计理由见 ADR 0013（backend 接缝）、
ADR 0014（双轨目标）与 ADR 0028（第三轨）。

## 1. 租机

- 单卡 910B（32 GB 或 64 GB），按小时计费。
- 镜像：CANN + PyTorch + `torch_npu` 已装好。版本 TBD，首次实验后回填。
- 开工前先记录，结尾原样写进结果文档：

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.version.cuda)"
python -c "import torch; print(torch.npu.is_available(), torch.npu.device_count())"
```

最后一行必须是 `True 1`。不是就停，先修环境。

## 2. 环境

```bash
git clone <repo> nanovllm-omni && cd nanovllm-omni
git submodule update --init --recursive
uv sync
```

- **不要装 `.[cuda]`**：它拉的是 triton / flash-attn 的 CUDA 轮子。
- `torch_npu` 按镜像已有版本手动安装，不要用 pip 覆盖镜像里的 torch。
- 仓库目前没有 `npu` extra；是否新增等首次实验后再定（ADR 0028）。

## 3. 权重

```bash
hf download jingyaogong/minimind-3o --local-dir "$HOME/minimind-3o"
```

- 第三轨只跑 thinker，不需要 Mimi / code2wav 权重。
- 记录权重路径与 `du -sh`，结果文档要写。

## 4. 跑 bench

里程碑 (1) 用 thinker bundle 路径（默认 `--pipeline thinker`）：

```bash
python -m nanovllm_omni.bench time \
  --pipeline thinker \
  --model "$HOME/minimind-3o" \
  --device npu \
  --prompts short_01,medium_01,system_01 \
  --max-tokens 16 \
  --runs 20 --warmup 3 \
  --out docs/perf/minimind-thinker-ascend-$(date +%F).csv
```

- `--device npu` 要求 `torch.npu` 可用；`load_minimind_omni_bundle` 会把它透传。
- **不要用 `bench_minimind` 入口**：它把非 CUDA 主机判成 `"cpu"` 并短路写零行
  （`nanovllm_omni/bench/bench_minimind.py:144-152`）。
- **不要用 `--pipeline full`**：它按 `device.startswith("cuda")` 选 dtype，
  `npu` 会落到 `float32`（`nanovllm_omni/bench/__main__.py:102, 201-207`）。
  里程碑 (1) 只测 thinker；full 路径的 dtype 与 talker 支持另算。
- 采样配置固定 `temperature=0.7`、`top_p=0.9`、`seed=42`；改了就写进结果文档。

## 5. 产物回收

每次实验必须留下：

- 结果文档 `docs/perf/minimind-thinker-ascend-<date>.md`，写清：硬件、驱动 /
  CANN / torch / torch_npu 版本、代码 commit、输入、采样配置、runs、warmup、
  wall、显存、阶段分解、限制。
- 原始 CSV、stdout 日志、`npu-smi info` 输出（本地产物，不进版本控制）。
- 若接了 profiler：trace 文件路径与体积。
- 与 3050 的数字只并列，不做跨机相减（ADR 0014）；精度不同也要写明。

## 6. 常见故障

TBD：首次实验后回填。至少覆盖：

- `torch.npu.is_available()` 为 False（驱动 / CANN 版本不匹配）。
- `hccl` 初始化失败。
- 显存不足（`num_blocks` 显式值调小）。
- `import triton` 失败或 triton-ascend 与 fork kernel 不兼容。
