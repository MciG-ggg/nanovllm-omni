# MiniMind-O 性能优化与修复全流程

> 记录从问题发现、错误尝试、架构纠偏，到当前优化实现和验证结果的完整过程。
>
> 当前代码基线：`66c5d2e fix: preserve sampling order in optimized generation`

## 1. 背景与目标

项目需要在 MiniMind-O 音频生成链路上降低 `generate` 延迟，同时保持：

- `Omni` 公开 API 不变；
- 文本 token 生成结果不变；
- Mimi audio code 不变；
- 最终 WAV 字节内容不变；
- 不引入新的运行时依赖；
- `vendor/` 不作为运行时模型代码来源。

整个链路主要是：

```text
Omni.generate
  -> engine/runtime.py
  -> models/minimind_omni/thinker.py
  -> model.generate(..., stream=True) 或本地优化循环
  -> Mimi decode
  -> WAV 编码
```

## 2. 初始性能基线

早期 TK-011/TK-015 的实测基线如下：

- GPU：RTX 3050 4GB；
- CUDA：13.1；
- PyTorch：2.13.0+cu130；
- 输入：6 个固定 prompt；
- `max_tokens=16`。

`session-1` 记录：

- `generate` 中位数约 **540 ms**；
- `total` 中位数约 **559 ms**；
- `generate` 占端到端延迟约 **96.6%**。

随后从 session-7 到 session-13，在同类环境下出现了约 **1.7 倍**的延迟漂移：

- `generate` 常见为 **770–960 ms**；
- `total` 常见为 **800–1000 ms**。

后来把旧版 TK-015 代码 `b364b84` 重新检出，在同一 WSL 环境跑出的结果仍是 740–940 ms，说明这不是 vendor 改造引入的回归，而是代码版本之外的环境/运行时漂移。

## 3. 第一次尝试：`torch.compile` 与 int8

TK-015 尝试过：

- `torch.compile`；
- bitsandbytes int8 MLP 量化；
- `reduce-overhead` 等编译路径。

结果：实现可以运行，但没有稳定的端到端收益。原因是主要开销并不只是矩阵乘法，而是逐 token 生成过程中的 Python 调度、模块调用和大量小 kernel launch。

因此不能把“模型已经 compile/int8”直接等同于“生成链路已经加速”。

## 4. 第二次尝试：CUDA Graph

随后尝试了 CUDA Graph：

1. 将模型拆成更容易捕获的 forward；
2. 增加通用 `graph_wrap`；
3. 在真实模型上测试 graph replay。

实测问题：

```text
graph capture failed for mimi.decode shape=(1, 8, 8):
CUDA error: operation failed due to a previous error during capture
```

`session-10` 到 `session-12` 的共同结论：

- `mimi.decode` 在当前 codec 上无法稳定捕获；
- graph 模式随后只能回退 eager；
- 主 generate 路径仍然是 eager；
- graph 模式没有稳定优于 eager，甚至可能更慢；
- 主要瓶颈仍然在 generate，而不是 decode。

### 4.1 Kineto 归因

`session-12` 的一次 16-step profile：

| 来源 | 每次调用耗时 | 占比 |
|---|---:|---:|
| Python/interpreter gap | 约 1034 ms | 约 68% |
| `cudaLaunchKernel` 主机调度 | 约 323 ms | 约 21% |
| 实际 GPU kernel | 约 138 ms | 约 9% |
| memcpy 与同步 | 约 31 ms | 约 2% |

每个 generate 调用大约有：

- 17,000 次 `cudaLaunchKernel`；
- 大量小于 10 微秒的 kernel；
- 500 次 KV 相关 memcpy；
- 344 次强制同步。

结论是：GPU 经常在等待 CPU 调度，CUDA Graph 只覆盖 decode 或固定形状子路径，无法解决完整的逐 token 主循环。

## 5. 关键代码定位

真正的热循环位于 MiniMind-O 的 `stream_generate`：

```python
while input_ids.shape[1] < start_pos + max_new_tokens:
    out = self.forward(...)
    ...
    text_token = torch.multinomial(...).item()
    ...
    for i, audio_logit in enumerate(out.audio_logits):
        ...
        code = torch.multinomial(...).item()
    input_ids = torch.cat(...)
    audio_buffer = torch.cat(...)
```

主要问题：

1. 每一步都重新增长 `input_ids`；
2. 每一步都重新增长 `audio_buffer`；
3. 8 路 audio code 分别采样并分别 `.item()`；
4. repetition penalty 通过 `input_ids[0].tolist()` 把序列拉回 CPU；
5. Attention 内部的 KV cache 仍然逐层 `torch.cat`；
6. 每一步都要执行完整的 thinker/talker forward。

cProfile 进一步确认：

- `stream_generate`：约 1.49 秒；
- `forward`：16 次；
- `nn.Module._call_impl`：约 4580 次；
- block forward：192 次；
- `torch.cat`：792 次；
- `.item()`：108 次；
- `torch.multinomial`：108 次。

## 6. 架构纠偏：`vendor/` 只是参考，不是运行时依赖

中间曾经把上游 MiniMind 建模代码复制到：

```text
nanovllm_omni/vendor/minimind/
```

并让 `bundle.py` 直接导入：

```python
from nanovllm_omni.vendor.minimind.model_omni import MiniMindOmni
```

这与项目意图不一致：`vendor/` 只是参考代码，不应成为运行时依赖。

随后完成纠偏：

- 删除 `nanovllm_omni/vendor/`；
- 从 `pyproject.toml` 删除 vendor package 配置；
- `.gitignore` 忽略 `nanovllm_omni/vendor/`；
- `bundle.py` 恢复：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

model = AutoModelForCausalLM.from_pretrained(
    snapshot_dir,
    trust_remote_code=True,
).eval()
```

优化逻辑不再复制模型实现，而是放在项目自己的：

```text
nanovllm_omni/models/minimind_omni/generation.py
```

这保持了模型加载职责和项目优化职责的分离。

## 7. 当前修复：项目侧优化生成循环

### 7.1 实现位置

新增：

```text
nanovllm_omni/models/minimind_omni/generation.py
```

`thinker.py` 的真实 MiniMind 模型路径改为调用：

```python
stream_generate_optimized(...)
```

对于测试用的轻量 fake model，仍保留原始 `model.generate(...)` 回退，以维持现有单元测试接口。

### 7.2 已实现的优化

#### A. 预分配输入 buffer

按最大生成长度预分配：

```python
text_buffer = torch.empty((1, capacity), ...)
audio_buffer = torch.full((1, 8, capacity), ...)
```

之后按位置写入，避免本地生成循环中的反复 `torch.cat`。

#### B. GPU 上完成 repetition penalty 索引

原逻辑：

```python
set(input_ids[0].tolist())
```

新逻辑：

```python
torch.unique(text_buffer[0, :current_len])
```

避免把完整 token 序列转换成 Python list。

#### C. audio logits 统一处理

8 路 audio logits 先在 GPU 上堆叠，再统一执行：

- temperature；
- 最近 code repetition penalty；
- `topk(50)`。

最后将结果批量转回 Python，减少设备到主机的数据往返。

#### D. 保留必要的随机顺序

第一次把 8 路 `multinomial` 完全合并后，虽然帧数正确，但固定 seed 下文本、audio code 和 WAV 都发生变化。

原因是 PyTorch Philox 随机数消耗顺序改变，进而影响后续文本 token。

因此最终方案保留 8 路独立的 `multinomial` 调用，只合并 logits/top-k 处理，并在最后一次性收集结果：

```python
sampled = torch.cat(
    [torch.multinomial(probabilities[row], 1) for row in range(len(active))]
)
codes = top_indices.gather(1, sampled[:, None]).flatten().tolist()
```

这是正确性优先的折中：不追求理论上最少的 kernel，而是保持原始随机序列和生成结果。

## 8. Correctness 验证

验证版本：`66c5d2e`。

固定条件：

- seed：`42`；
- 同一个已加载模型；
- 同一个 prompt；
- `max_tokens=16`；
- 原始 `model.generate(..., stream=True)` 对比 `stream_generate_optimized(...)`。

结果：

```text
text_steps:        16 / 16
audio_frames:       8 / 8
text_exact:         true
audio_codes_exact:  true
wav_exact:          true
```

两份 WAV 的 MD5 完全相同：

```text
8eca3d7a9b09382274573f57d466f3ec
```

因此当前优化没有改变：

- 文本 token；
- audio code；
- 音频帧数量；
- 最终 WAV 字节内容。

## 9. 性能验证

### 9.1 初步验证

`session-14` 使用 `runs=3,warmup=1`：

- 优化后 total 平均约 818 ms；
- 早期 eager 基线约 895 ms；
- 初步看约 8–10% 改善。

### 9.2 稳定复测

`session-15` 使用 `runs=5,warmup=2`：

- 优化后 total 平均中位数约 818 ms；
- 相比早期基线约改善 8.6%；
- VRAM 仍约 486 MB，没有额外显存代价；
- 所有 prompt 都生成 8 帧音频。

### 9.3 再次复测

`session-16` 使用相同的 `runs=5,warmup=2`：

- 优化后 total 平均中位数约 876 ms；
- 相比早期基线约改善 2%；
- 相比 session-15 慢约 7%。

这说明 WSL 当前运行存在明显波动。因此更严谨的结论是：

> 优化代码确实保持 correctness，并且在部分复测中带来约 8–9% 收益；但跨运行环境的稳定收益目前只能保守表述为约 2–9%，不能固定宣称 9%。

## 10. 当前提交历史

相关提交：

| Commit | 内容 |
|---|---|
| `17adbd3` | 临时引入 MiniMind vendor 代码 |
| `0a42002` | 尝试让 forward 支持 CUDA Graph capture |
| `77850a9` | 增加通用 CUDA Graph wrapper |
| `e656664` | 在 `models/minimind_omni/` 增加优化生成循环 |
| `a44a383` | 删除运行时 vendor 依赖，vendor 改为参考目录 |
| `66c5d2e` | 修复批量采样导致的随机顺序变化 |

当前代码组织：

```text
nanovllm_omni/models/minimind_omni/
├── bundle.py       # HF 模型与 Mimi 加载
├── thinker.py      # 生成入口与兼容回退
├── generation.py   # 项目自有优化生成循环
├── code2wav.py     # Mimi decode 与 WAV 编码
├── talker.py
└── pipeline.py
```

## 11. 当前结论

### 已解决

- vendor 不再参与运行时 import；
- 优化逻辑已经落在 `models/minimind_omni/`；
- 原始模型代码保持不改；
- 固定 seed 下文本、audio code、WAV 完全一致；
- 优化后性能在多次实测中有可见收益；
- 显存没有明显增加；
- 非 smoke 测试通过。

### 尚未解决

- Attention 内部 KV cache 仍然逐步 `torch.cat`；
- 主 forward 仍包含大量小 kernel launch；
- 完整 CUDA Graph 仍未接入主 generate 路径；
- WSL 环境存在约 1.7 倍的跨时段性能漂移；
- 当前收益受运行环境影响，仍需更多受控重复实验才能给出稳定百分比。

### 不建议继续做的事情

在没有明确范围授权前，不建议：

- 修改或重新维护完整 vendor 模型副本；
- 直接复制 vllm-omni 的大规模 KV cache/fused kernel 实现；
- 为此引入 CUDA/C++ 扩展或新依赖；
- 把 CUDA Graph 的局部成功误判为端到端加速。

当前最合理的停点是：保留 `generation.py` 的 correctness-safe 优化，把 KV cache 预分配作为独立、风险更高的后续任务。

## 12. 验证命令

本地静态与单元检查：

```bash
uv run ruff check nanovllm_omni/ tests/
uv run black --check nanovllm_omni/ tests/
uv run python -m compileall -q nanovllm_omni
uv run python -m pytest -m "not smoke" -q
uv run python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"
```

WSL 性能基准：

```bash
.venv/bin/python -m nanovllm_omni.optim.bench time \
  --model /home/mcig/minimind-3o \
  --mimi /home/mcig/mimi \
  --max-tokens 16 \
  --runs 5 \
  --warmup 2
```
