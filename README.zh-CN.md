# nanovllm-omni

[English](README.md) | **简体中文**

[![CI](https://github.com/MciG-ggg/nanovllm-omni/actions/workflows/ci.yml/badge.svg)](https://github.com/MciG-ggg/nanovllm-omni/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](.)
[![Models](https://img.shields.io/badge/models-4-green.svg)](.)

- 🎯 **MiniMind-O 流水线** — 最小的 Thinker → Talker → Code2Wav 全链路运行时,加载真实的 `jingyaogong/minimind-3o` 权重
- ⚡ **full 三阶段 E2E 在 RTX 3050 (4 GB) 上 total 中位约 0.65–1.04 s** — 一条 prompt → thinker → talker → MTP → Mimi → WAV,真 `jingyaogong/minimind-3o` + `kyutai/mimi` 权重,torch 2.14.0+cu130,2026-09 在仅 full 分支上重测。六道 bench 题各 20 次:**total 中位 628–1038 ms**(short ~0.63 s、medium ~0.75 s、system ~1.04 s),p95 ≤ 1.11 s,显存峰值 ~1881 MiB。数据与原始 CSV 见 `docs/perf/tk005-rtx3050.md` / `docs/perf/full-e2e-rtx3050.csv`。
- 🔧 **thinker 解码 bench 原语(回归用,不代表端到端)** — 默认 `bench time` 只测单段 thinker 解码:同一硬件上 `--use-thinker-cuda-graph` 约 177 ms、eager 约 483 ms total 中位。该数字**不含** talker/MTP/Mimi 阶段,不可当作端到端延迟引用。
- 🔁 **StagePool 模式(单 replica in-process)** — per-stage 持续批处理通过 `RuntimeScheduler` 完成;多 replica + RoundRobin LB 已删除(测得 `num_replicas=1 == num_replicas=2`,见 `tests/test_batched_runner_contract.py`)
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

## 同名兄弟项目

[`Rising0321/nano-vllm-omni`](https://github.com/Rising0321/nano-vllm-omni) 是另一个独立项目,碰巧和我们的仓库名 *以及* Python 包名(`nanovllm_omni`)都一样。它**不是**本仓库的 fork,**也不是**重写 —— 是被别人起的、scope 不同的同名项目。

| 维度 | `MciG-ggg/nanovllm-omni`(本仓库) | `Rising0321/nano-vllm-omni` |
|---|---|---|
| 模态 | 音频、图像、VLM-文本、动作 | 视频(I2V / TI2V) |
| 模型族 | MiniMind-O、SD-Turbo、SmolVLM、SmolVLA | 只 Wan2.2-TI2V-5B |
| 硬件目标 | RTX 3050 4 GB,单进程 | RTX 3090 24 GB,单进程 + CPU offload |
| 流水线形态 | 一个 `Omni(...)` 内 per-family 阶段工厂 | 显式 step-wise 调度器(`request → scheduler → runner → pipeline`) |
| 对齐目标 | `vllm-omni` 消费侧 API | `vllm-omni` 扩散阶段契约(`prepare_encode → denoise_step → step_scheduler → post_decode`) |

两个项目都是对 `vllm-omni` 的独立教学式解读,谁也不是谁的 fork。如果你要找**视频 I2V / Wan2.2** 的实现,Rising0321 那个仓库才是。

> **提醒:** 两个项目都发布同名 Python 包 `nanovllm_omni`。同一个 venv 里装会冲突,得用两个独立环境。

## 支持的模型

| 模型 | 阶段数 | 输出 | 权重 |
|---|---:|---|---|
| MiniMind-O(`minimind-3o`) | 3 (Thinker → Talker → Code2Wav) | 音频(24 kHz mono WAV) | `jingyaogong/minimind-3o` + `kyutai/mimi` |
| SmolVLM-500M-Instruct | 1 (VLM) | 文本 | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| SD-Turbo | 1 (DIFFUSION, 1-step) | 图像(512×512 PNG) | `stabilityai/sd-turbo` |
| SmolVLA | 1 (LLM_GENERATION) | 动作 chunk | `HuggingFaceVLA/smolvla_libero` + LIBERO 数据集 |

四个模型全部跑在单张 4 GB 消费级卡(RTX 3050)上,一个 Python 进程。视频生成、超出所列四类的多模型流水线、完整 VLA 栈、全双工 S2S 都是当前规划、暂未实现。

## Gallery

四个对齐模型族在单张 RTX 3050 (4 GB) 卡上的真实输出。音频和图像用 `~/minimind-3o`、`~/mimi` 以及 HF 缓存里的本地权重 snapshot;SD-Turbo + SmolVLM 权重需预下载(`hf download …`),loader 是 offline-first 的。

### MiniMind-O — 音频(24 kHz mono WAV)

<audio controls src="docs/gallery/minimind_o_1.wav"></audio>

本地最长的样本(~8.88 s,~386 KB)—— 由 Thinker → Talker → Code2Wav 流水线生成。4 GB 卡上请保持 `max_tokens ≤ 16`,避免 Mimi codec 前向时 OOM。代码:`examples/offline_inference/minimind_o/`。

### SD-Turbo — 图像(512×512 PNG)

两张样例,均 1 步扩散,`guidance_scale=0.0`:

![雾中晨曦山谷,油画,8k](docs/gallery/sd_turbo_1.png)
> `a misty mountain valley at sunrise, oil painting, 8k`

![木桌上冒着蒸汽的玻璃茶壶,棚拍](docs/gallery/sd_turbo_2.png)
> `a glass teapot with steam rising, on a wooden table, studio photo`

SD-Turbo 的对抗蒸馏锁死 `guidance_scale=0.0` 与 `num_inference_steps=1`;传其他值画质会下降。代码:`examples/offline_inference/sd_turbo/`。

### SmolVLM-500M-Instruct — 文本(图像 + 文本 → 文本)

> **图像**:`docs/images/sd_turbo_sample.png`(512×512 PNG,由 SD-Turbo 生成 —— 就是上面那张猫图)
>
> **Prompt**:`What is in this image and what art style is it rendered in?`
>
> **回答**:
>
> > There is a cat in the image. The image is a photograph.

500 M 参数的模型回答很短,风格问题被归成一个标签(`photograph`)而不是描述;更大的 VLM 会展开。代码:`examples/offline_inference/smolvlm/`。完整转写在 [`docs/gallery/smolvlm_0.md`](docs/gallery/smolvlm_0.md)。

### SmolVLA — 动作 chunk(LIBERO 评测回放)

<video controls src="docs/gallery/smolvla_0.mp4" width="480"></video>

LIBERO 评测片段;LLM_GENERATION stage 输出 action chunk(`numpy.ndarray`,形状 `[chunk_size, action_dim]`)。视频是把 chunk 喂给 LIBERO 模拟器跑出来的。代码:`examples/offline_inference/smolvla/libero_eval.py`。

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
engine = Omni("HuggingFaceVLA/smolvla_libero")
outputs = engine.generate([{"prompt": "do the task", "image": rgb_obs}],
                          SamplingParams(max_tokens=50))
outputs[0].multimodal_output["actions"].array  # np.ndarray [chunk, action_dim]
```

### 在线 Demo

打包好的 Gradio UI 用四个 tab 拼四个模型族,每个 tab 首次点击时才 lazy-load `Omni(...)` 引擎:

```bash
pip install -e ".[dev,minimind,smolvla,gradio]"
python -m nanovllm_omni.serving.app
```

4 GB 卡上,同一时刻只加载一个 tab 的模型。launcher 只依赖 `[gradio]` —— 想跑哪个 tab 就先装对应的 `[minimind]` / `[smolvla]`。

### 复现延迟数字

顶部的 full-E2E 数字来自 `--pipeline full` bench,它让一条 prompt 走完 `Omni.generate`(真权重端到端):

```bash
# 在 WSL 里(~mcig@mcigs-wsl)—需要 torch + 本地权重 snapshot
python -m nanovllm_omni.engine.bench time \
    --pipeline full \
    --max-tokens 16 --runs 20 --warmup 1
```

thinker 单段原语(`bench time` 不带 `--pipeline full`)保留用于 stage 回归,不作为 E2E 数字。

Full-E2E 数字与验证:`docs/perf/tk005-rtx3050.md`,原始 CSV 见 `docs/perf/full-e2e-rtx3050.csv`。thinker CUDA-Graph kernel 拆分(历史):`docs/perf/minimind-omni-under-500ms.md`。Eager path 基线(`d2ebe56` 合并前):`docs/perf/session-1.md`。

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

同卡上的请求级并行用 in-process 批量 runner(`examples/offline_inference/minimind_o/batched.py`,由 `tests/test_batched_generation.py` 锁定 — Q10a)。多卡、per-stage 子进程隔离、tensor-parallel 和 pipeline-parallel 调度器都明确不在范围内;以后要加任意一项,改动必须从改造 `models/minimind_omni/bundle.py`(每个 replica 一个 bundle)和 `engine/runner.py` / `engine/executor.py`(per-replica 推理路径)开始,而不是把 vllm-omni 的 `StageRuntime` 硬塞进一个今天用不上的运行时。

四个支持的模型族共用同一个单进程 pipeline runner。per-stage 持续批处理(TK-004)目前只用于 MiniMind-O,通过 `models/minimind_omni/runtime_scheduler.py` in-process 实现。per-stage replica + RoundRobin LB 层(TK-007)已删除(测得 `num_replicas=1 == num_replicas=2`,见 `tests/test_batched_runner_contract.py`)。

## 致谢

- [`vllm-omni`](https://github.com/vllm-project/vllm-omni) — 我们对其 omni 模态服务架构与消费侧 API 契约进行教学式解读并与之对齐。本仓库不导入它的代码,具体偏离见 README「Scope / 范围」一节。
- [`GeeeekExplorer/nano-vllm`](https://github.com/GeeeekExplorer/nano-vllm)
  和 [`zhx-llm/nanovllm`](https://github.com/zhx-llm/nanovllm) — 把 vLLM 文本路径压缩到 ~1k 行可读 Python 的教学重写,本仓库的 per-family 阶段工厂受到了它们的结构启发。
- 模型作者(我们直接使用其权重):
  - MiniMind-O — [`jingyaogong/minimind-3o`](https://huggingface.co/jingyaogong/minimind-3o)
  - Mimi 编码器 — [`kyutai/mimi`](https://huggingface.co/kyutai/mimi)
  - SD-Turbo — [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo)
  - SmolVLM-500M — [`HuggingFaceTB/SmolVLM-500M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
  - SmolVLA — [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) 跑在 [LIBERO](https://libero-project.github.io) 基准上
- [Hugging Face `diffusers`](https://github.com/huggingface/diffusers) 和
  [`transformers`](https://github.com/huggingface/transformers) — SD-Turbo / SmolVLM / MiniMind-O 背后的推理代码路径。

## License

Apache-2.0。详见 [LICENSE](LICENSE)。
