# nanovllm-omni

[English](README.md) | **简体中文**

[![CI](https://github.com/MciG-ggg/nanovllm-omni/actions/workflows/ci.yml/badge.svg)](https://github.com/MciG-ggg/nanovllm-omni/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](.)
[![Models](https://img.shields.io/badge/models-4-green.svg)](.)

- 🎯 **MiniMind-O 流水线** — 最小的 Thinker → Talker → Code2Wav 全链路运行时,加载真实的 `jingyaogong/minimind-3o` 权重
- ⚡ **RTX 3050 (4 GB) 上 ~320 ms p50 / ~345 ms p95 / ~350 ms p99** — `d2ebe56 perf(stack)` 合并后单条 MiniMind-O audio 请求(融合 QKV/gate-up 投影、融合 RMSNorm、融合 RoPE 在 `nanovllm_omni/models/minimind_omni/attention.py`;预分配 KV buffer;SDPA decode `is_causal=True`)。p50/p95/p99 来自 `docs/perf/minimind-omni-under-500ms.md`(20 次跑,torch 2.13):p50 320 ms、mean 323 ms、stdev 11.6 ms、min 305 ms、max 343 ms;p95/p99 用 mean + z·stdev 近似。WSL 上复现命令 `python -m nanovllm_omni.optim.bench time`。
- 🔁 **StagePool 模式演示** — `num_replicas ≥ 2`,RoundRobin 负载均衡,每个输出带 `(stage_id, replica_id)`
- 🌐 **统一的 omni I/O 契约** — MiniMind-O(音频)+ SmolVLM(文本)+ SD-Turbo(图像)+ SmolVLA(动作)共用同一个 `OmniRequestOutput` 信封

一个小型、本地的参考实现,在单卡上跑 vllm-omni 的阶段化(stage-based)服务架构。本项目**不**声称为 vllm-omni 全功能集的实现。

![RTX 3050 上的 SD-Turbo 单步图像生成(512×512)](docs/images/sd_turbo_sample.png)

![MiniMind-O 音频输出:来自 Thinker → Talker → Code2Wav 流水线的 8.88s @ 24 kHz mono](docs/images/minimind_o_waveform.png)

## 与 `nano-vllm` 和 `nanovllm` 的区别

还有两个 Python 项目名字里带 `nano-vllm` 风格,**它们和本项目不是一回事**:

| 项目 | 模态 | 目标 |
|---|---|---|
| `nanovllm-omni`(本仓库) | 音频 / 图像 / 动作 / 文本 | 与 vllm-omni 消费侧 API 对齐的 omni-模态运行时,跑在单张 4 GB 卡上 |
| [`GeeeekExplorer/nano-vllm`](https://github.com/GeeeekExplorer/nano-vllm) | 仅文本 | 最小化教学用 vLLM 文本路径重写 |
| [`zhx-llm/nanovllm`](https://github.com/zhx-llm/nanovllm) | 仅文本 | 另一个教学用 vLLM 文本路径重写 |

如果你搜的是*纯文本*的 mini-vLLM,看上面两个仓库才对。

## 支持的模型

| 模型 | 阶段数 | 输出 | 权重 |
|---|---:|---|---|
| MiniMind-O(`minimind-3o`) | 3 (Thinker → Talker → Code2Wav) | 音频(24 kHz mono WAV) | `jingyaogong/minimind-3o` + `kyutai/mimi` |
| SmolVLM-500M-Instruct | 1 (VLM) | 文本 | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| SD-Turbo | 1 (DIFFUSION, 1-step) | 图像(512×512 PNG) | `stabilityai/sd-turbo` |
| SmolVLA | 1 (LLM_GENERATION) | 动作 chunk | `HuggingFaceTB/SmolVLA-256M` + LIBERO 数据集 |

四个模型全部跑在单张 4 GB 消费级卡(RTX 3050)上,一个 Python 进程。视频生成、超出所列四类的多模型流水线、完整 VLA 栈、全双工 S2S 都是当前规划、暂未实现。

## 快速开始

安装包及其已有依赖;音频路径加 `[minimind]` extra(torch / transformers),SmolVLA examples 加 `.[smolvla]`:

```bash
pip install -e ".[dev,minimind]"
```

跑 MiniMind-O(音频),先把权重一次性拉到本地(bundle loader 是 offline-first 的,不会自动 fetch):

```bash
hf download jingyaogong/minimind-3o --local-dir "$HOME/minimind-3o"
hf download kyutai/mimi             --local-dir "$HOME/mimi"
```

然后用本地权重跑单 prompt 烟雾测试:

```bash
cd examples/offline_inference/minimind_o
HF_HUB_OFFLINE=1 bash run_end2end.sh \
    --model "$HOME/minimind-3o" --mimi "$HOME/mimi" --out audio.wav
```

`audio.wav` 落到 example 文件夹里。建议用至少 4 GB VRAM 的 CUDA 机器;CPU 推理也能跑但慢。

SD-Turbo(图像)、SmolVLM(文本)、SmolVLA(动作)看 `examples/offline_inference/<family>/` 下的 per-family examples。

四个家族共用同一个对齐 API:

```python
from nanovllm_omni import Omni, SamplingParams

# 音频(MiniMind-O)
engine = Omni("jingyaogong/minimind-3o")
outputs = engine.generate(["hi"], SamplingParams(max_tokens=8))
with open("audio.wav", "wb") as f:
    f.write(outputs[0].multimodal_output["audio"].wav_bytes())

# 图像(SD-Turbo)
engine = Omni("stabilityai/sd-turbo")
outputs = engine.generate(["a red apple"], SamplingParams(max_tokens=1))
outputs[0].multimodal_output["image"].save("apple.png")

# 文本(SmolVLM)
engine = Omni("HuggingFaceTB/SmolVLM-500M-Instruct")
outputs = engine.generate(["What is in this image? <image>"], SamplingParams(max_tokens=64))

# 动作(SmolVLA)
engine = Omni("HuggingFaceTB/SmolVLA-256M")
outputs = engine.generate([{"prompt": "do the task", "image": rgb_obs}],
                          SamplingParams(max_tokens=50))
outputs[0].multimodal_output["actions"].array  # np.ndarray [chunk, action_dim]
```

### 复现延迟数字

上面的 320 ms p50 需要 CUDA-Graph 快路径(`--use-cuda-graph` opt-in;CUDA-only)。不传这个 flag,同一硬件上默认跑 ~715 ms p50。

```bash
# 在 WSL 里(~mcig@mcigs-wsl)—需要 torch + 本地权重 snapshot
python -m nanovllm_omni.optim.bench time \
    --model ~/minimind-3o --mimi ~/mimi \
    --max-tokens 16 --runs 5 --warmup 2 \
    --use-cuda-graph
```

完整分布(CUDA-Graph path,25 次实验 trace,kernel 级别拆分):`docs/perf/minimind-omni-under-500ms.md`。Eager path 基线(`d2ebe56` 合并前):`docs/perf/session-1.md`。

## 配置

流水线拓扑在代码里(`nanovllm_omni/config/registry.py`);per-stage 的采样和资源默认在 `nanovllm_omni/deploy/*.yaml`,打进 wheel,运行时从包目录解析。

## 仓库结构

```
nanovllm_omni/  # 包实现(config 层在 nanovllm_omni/config/)
    deploy/     # per-family 采样/资源默认(打进 wheel)
examples/
    offline_inference/
        minimind_o/   # Python seam 音频烟雾测试(单条 + 批量)
        sd_turbo/     # SD-Turbo 图像生成
        smolvla/      # SmolVLA 策略(合成 L1 + LIBERO 评测)
        smolvlm/      # SmolVLM 文本/VLM
    online_serving/
        minimind_o/   # /v1/chat/completions 的 curl/stdlib 客户端
tests/          # 自动化测试
docs/           # 项目笔记与性能归档
```

## 范围

本仓库明确**不**包含:视频生成、超出所列四类的多模型流水线、分布式执行、WebSocket、FastAPI/uvicorn 服务。对齐的 HTTP seam 在可用时使用 Python 标准库 HTTP server。

### 单卡、单进程运行时

一个 `Omni(...)` 构造出一个 `MinimindBundle`(`nanovllm_omni/models/minimind_omni/bundle.py`),把完整的 MiniMind-O checkpoint、Mimi codec 和 tokenizer 都装在**一个** Python 进程里。没有 per-stage 子进程池、没有 `StageRuntime`、没有多卡 dispatch、也没有 tensor-parallel worker — 这是设计上有意的。

这是与 vllm-omni 的有意分歧:vllm-omni 里每个 stage 跑在独立的 `StageEngineCoreProc` 子进程里,stage 可以绑定到不同的 GPU。vllm-omni 能这么做是因为每个 stage 都是独立的 HF checkpoint(thinker 30B、talker 2B、code2wav 1B 等)。MiniMind-O 的 `thinker` 和 `talker` 是同一个 `AutoModelForCausalLM` 的子模块,从同一个 safetensors 文件加载,所以 per-stage 子进程隔离会强制每个 stage 重新加载完整的 ~3 GB checkpoint — 在我们目标的 4 GB 卡上跑不动,在更大的卡上也是浪费。`bundle.py` 改成所有东西在一个进程里;stage 之间的通信(thinker hidden states → talker → code2wav)用 Python tensor 引用,零序列化、零 IPC。

同卡上的请求级并行用 in-process 批量 runner(`examples/offline_inference/minimind_o/batched.py`,由 `tests/test_batched_generation.py` 锁定 — Q10a)。多卡、per-stage 子进程隔离、tensor-parallel 和 pipeline-parallel 调度器都明确不在范围内;以后要加任意一项,改动必须从改造 `bundle.py`(每个 replica 一个 bundle)和 `engine/runtime.py`(per-replica 推理路径)开始,而不是把 vllm-omni 的 `StageRuntime` 硬塞进一个今天用不上的运行时。

四个支持的模型族共用同一个单进程运行时:per-stage 持续批处理(TK-004)和 per-stage replica + RoundRobin LB(TK-007)都是 in-process 数据结构(`engine/runtime_scheduler.py`、`engine/load_balancer.py`),不是子进程池。

## License

Apache-2.0。详见 [LICENSE](LICENSE)。
