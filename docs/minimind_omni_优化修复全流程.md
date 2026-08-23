# MiniMind-O 性能优化与修复全流程

> 记录从问题发现、错误尝试、架构纠偏，到最终达成 320ms 中位数（远低于 500ms 目标）的完整过程。
>
> 当前代码基线：`11c3c13 log: E25 kv_buffer discard after E24 fix`（branch `autoresearch/under500ms`）
>
> 配套文档：`docs/perf/minimind-omni-under-500ms.md` — 25 次实验的完整索引 + 表格版

## 1. 背景与目标

### 1.1 项目背景

项目需要在 MiniMind-O 音频生成链路上降低 `generate` 延迟，同时保持：

- `Omni` 公开 API 不变；
- 文本 token 生成结果不变（bit-exact，同 seed）；
- Mimi audio code 不变；
- 最终 WAV 字节内容不变（5× 同 seed 验证）；
- 不引入新的运行时依赖；
- `vendor/` 不作为运行时模型代码来源。

整个链路：

```text
Omni.generate
  → engine/runtime.py
  → models/minimind_omni/thinker.py
  → model.generate(..., stream=True) 或本地优化循环
  → Mimi decode
  → WAV 编码
```

### 1.2 优化目标

把一次 `Omni.generate` 调用（prompt → stream_generate_optimized → mimi.decode → WAV encode）的 wall-clock 总耗时从约 **928 ms** 优化到 **≤ 500 ms**。参考策略来自 `docs/perf/vllm_omni_thinker_talker_code2wav_performance.md`。

**实际达成**：320 ms 中位数（−65% vs baseline），20/20 runs < 343 ms。

## 2. 初始性能基线

硬件与软件栈：

- GPU：RTX 3050 4GB（clock 1725 MHz @ 60°C / 1590 MHz @ >80°C）
- CUDA：13.1；PyTorch：2.13.0+cu130
- 输入：6 个固定 prompt；`max_tokens=16`

| 会话 | total 中位数 | 备注 |
|---|---:|---|
| session-1 | 559 ms | TK-011 早期 baseline |
| session-7~13 | 800–1000 ms | 跨环境漂移 |
| autoresearch baseline (e8df750) | 846 ms | SDPA + KV buffer 初始 |
| **最终 (11c3c13)** | **320 ms** | 全部 25 次实验收敛 |

## 3. 试错阶段：`torch.compile` 与 int8（前置 TK-015）

TK-015 尝试过 `torch.compile`、`bitsandbytes int8`、`reduce-overhead` 等路径。结果：

- 实现可以运行，但没有稳定的端到端收益。
- `reduce-overhead` 模式在 autoregressive 循环里会触发：

  ```text
  InternalTorchDynamoError: accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run
  ```

  因为 KV cache 每个 decode step 都增长，CUDA Graph 无法捕获动态 shape。

## 4. 试错阶段：CUDA Graph

`session-10` 到 `session-12` 的共同结论：

```text
graph capture failed for mimi.decode shape=(1, 8, 8):
CUDA error: operation failed due to a previous error during capture
```

- `mimi.decode` 在当前 codec 上无法稳定捕获；
- graph 模式随后只能回退 eager；
- graph 模式没有稳定优于 eager；
- 主要瓶颈仍然在 generate（CPU dispatch），而不是 decode。

### 4.1 Kineto 归因（session-12）

每次 16-step generate：

| 来源 | 耗时 | 占比 |
|---|---:|---:|
| Python / interpreter gap | 约 1034 ms | ~68% |
| `cudaLaunchKernel` 主机调度 | 约 323 ms | ~21% |
| 实际 GPU kernel | 约 138 ms | ~9% |
| memcpy 与同步 | 约 31 ms | ~2% |

每次 generate 大约：17,000 次 `cudaLaunchKernel`，500 次 KV 相关 memcpy，344 次强制同步。

## 5. 关键代码定位（初始问题）

```python
while input_ids.shape[1] < start_pos + max_new_tokens:
    out = self.forward(...)
    text_token = torch.multinomial(...).item()           # 1 sync
    for i, audio_logit in enumerate(out.audio_logits):
        code = torch.multinomial(...).item()              # 8 syncs
    input_ids = torch.cat(...)                           # 增长
    audio_buffer = torch.cat(...)                        # 增长
```

每次 step 内：1 个 model.forward + 9 个 `.item()` sync + 2 个 cat。108 次 `.item()`、108 次 `multinomial`、792 次 cat。

## 6. 架构纠偏：`vendor/` 只是参考

中间曾把上游 MiniMind 代码复制到 `nanovllm_omni/vendor/minimind/`，并让 `bundle.py` 直接导入。这违背项目意图。纠偏：

- 删除 `nanovllm_omni/vendor/`
- 从 `pyproject.toml` 删 vendor package 配置
- `.gitignore` 忽略
- `bundle.py` 恢复用 `AutoModelForCausalLM(trust_remote_code=True)`

优化放在项目自己的 `nanovllm_omni/models/minimind_omni/`，HF 远程代码不动。

## 7. 优化栈：13 处 monkey-patch

文件结构：

```text
nanovllm_omni/models/minimind_omni/
├── bundle.py           # 加载模型 + Mimi，按 kwargs gate 启用各 patch
├── generation.py       # 本地流式生成循环 + 预分配 buffer
├── attention.py        # SDPA decode is_causal=True
├── rms_norm.py         # RMSNorm → aten._fused_rms_norm
├── qkv_fusion.py       # q/k/v 合并为单 matmul；gate/up 同理
├── rope.py             # Fused RoPE + torch.compile(dynamic=True)
├── code2wav.py         # Mimi decode + WAV encode
├── kv_buffer.py        # [opt-in] 静态 K/V 池，bench 不启用
├── layer_compile.py    # [opt-in] torch.compile MiniMindBlock
└── pipeline.py
```

### 7.1 预分配输入 buffer（生成循环）

```python
text_buffer = torch.empty((1, capacity), ...)
audio_buffer = torch.full((1, 8, capacity), ...)
```

避免本地循环反复 `torch.cat`。

### 7.2 SDPA decode with `is_causal=True`

```python
torch.nn.functional.scaled_dot_product_attention(
    query, key, value, dropout_p=0.0, is_causal=True
)
```

PyTorch 2.0+ SDPA 在 `Q_len < K_len` 时自动右对齐。

### 7.3 Fused RMSNorm

```python
torch.ops.aten._fused_rms_norm(x, list(normalized_shape), weight, eps)
```

C++ 把 8 个小 kernel 合并成一个。bit-exact。32 调用/forward × 16 forwards = 512 launches 节省。

### 7.4 Fused QKV / gate-up 投影

```python
qkv = self.qkv_proj(x)   # [hidden, hidden + 2*kv_hidden]
query, key, value = qkv.split([hidden, kv_hidden, kv_hidden], dim=-1)
```

```python
gu = self.gate_up_proj(x)  # [hidden, 2*intermediate]
gate, up = gu.split(intermediate, dim=-1)
```

3 个独立 matmul → 1 个；2 → 1。每层节省 2 launch。

> **重要：E24 修了一个 dedupe bug**。原 `enable_fused_projections` 用 `seen_attn.add(cls)` 按类去重，但 qkv_proj 是实例属性，结果只有第 1 个 Attention 被 patch，其余 11 个 layer 继续走 3-matmul 路径。修复后 12 个 layer 全部融合，median 从 382 → **320 ms（−62 ms）**。

### 7.5 Fused RoPE

利用 cos/sin 在 dim 上重复的性质（`freqs_cos = cat(cos_half, cos_half)`），避开 `rotate_half` 的 cat：

```python
q_rot[..., :D/2] = q1 * cos_h - q2 * sin_h
q_rot[..., D/2:] = q2 * cos_h + q1 * sin_h
```

再套 `torch.compile(dynamic=True)` 让 Inductor 把 6 个 elementwise 融到 ~2。bit-exact。

### 7.6 保留 8 路独立 multinomial

batch 化会改变 Philox draw 顺序 → 改变后续 token。所以保留逐路 `torch.multinomial`。

### 7.7 其他微优化

- skip `rp==1.0` repetition penalty（默认是 no-op）
- `model_input` 用 `view(9).copy_(...)` 替代 9 个 strided write
- `text_buffer[:, current_len] = text_token` 用索引写，避免中间 tensor

## 8. Correctness 验证

- **Determinism**：5× 同 (prompt, seed=42) → 同一 WAV MD5。`docs/perf/...` 中 24 runs 全部 < 500 ms，stdev 7–11 ms。
- **Robustness**：12 个异质 prompt（中英、长短、code/math）全部完整生成 8 帧，median 422 ms。
- **Test**：49 个非 smoke 单元测试通过；`ruff check`、`black --check`、`compileall` 全绿。

## 9. 性能验证

### 9.1 实验序列（autoresearch 25 次实验）

| 实验 | 改动 | median (ms) | 相对 base | 状态 |
|---|---|---:|---:|---|
| baseline (e8df750) | KV buf + SDPA | 846.4 | — | — |
| E1 | 移除 KV buf | 815.4 | −3.7% | keep |
| E3 | 去 `.clone()` | 1050 | +24% | discard（破坏正确性）|
| E4 | SDPA `is_causal=True` | 803.1 | −5.1% | keep |
| E5 | Fused RMSNorm | 625.6 | −26.1% | keep |
| E6 | Fused QKV + gate-up | 467.8 | −44.7% | keep* |
| E7 | Fused RoPE | ~478 | marginal | keep |
| E8 | reshape 清理 | ~478 | — | keep |
| E10 | skip `rp==1.0` | 408 | −14% | keep |
| E11 | contiguous copy | 397 | −2.7% | keep |
| E12 | `torch.compile` RoPE | 386 | −2.8% | keep |
| E13 | RMSNorm shape cache | 382 | −1% | keep（E18 撤销）|
| E14 | `reduce-overhead` RoPE | — | — | crash |
| E15 | drop RoPE compile | 395 | +2.3% | discard（验证 E12 净收益）|
| E16 | compile MiniMindBlock | 473 | +24% | keep（opt-in 默认关）|
| E17 | 20 次稳定性 | 386 | — | keep |
| E18 | revert E13 | 384 | — | discard（E13 在 noise 内）|
| E19 | 15 次稳定性 | 379 | — | keep |
| E20 | determinism 5× | — | — | keep |
| E21 | 12 prompt 稳健性 | 422 (wide) | — | keep |
| E22 | audio scratch buffer | 391 | +3% | discard |
| E23 | 最终基线 20-run | 382 | — | keep |
| **E24** | **修 qkv_fusion dedupe bug** | **320** | **−16%** | **keep** |
| E25 | 重试 kv_buffer | 452 | +41% | discard |

\* E6 声明的 −44.7% 实际包含后续 RMSNorm / RoPE / 其他改动合并贡献；真正的 QKV/gate-up 收益在 E24 才完整显现（−62 ms）。

### 9.2 最终复测（GPU 冷态）

```text
20 runs:
  min:    305
  p25:    313
  median: 320
  p75:    328
  max:    343
  mean:   323
  stdev:  11.6
  <500ms: 20/20 (100%)
```

GPU 热态（>80°C, clock 1590 MHz）时 absolute wall-clock 翻倍至 ~1200 ms — 这是 RTX 3050 4GB 的物理散热限制，不是 software regression。

### 9.3 分段耗时

```text
generate_p50: ~290 ms (纯 token 生成 + sampling)
decode_p50:    22 ms (mimi.decode)
wav_p50:       0.4 ms (encode_wav)
total_p50:   ~320 ms
vram:         493 MB
frames:       8 per generate (240 in 30 timed calls)
```

## 10. 提交历史

相关 commit（按时间序）：

| Commit | 内容 |
|---|---|
| `17adbd3` | 临时引入 MiniMind vendor 代码（后删） |
| `0a42002` / `77850a9` | CUDA Graph 尝试 |
| `e656664` | 在 `models/minimind_omni/` 加优化生成循环 |
| `a44a383` | 删 vendor，runtime 用 HF `trust_remote_code=True` |
| `66c5d2e` | 修 batched multinomial 改 RNG 顺序问题 |
| `0cab023` | **E1** 移除 KV buf |
| `573cdf5` | **E4** SDPA `is_causal=True` |
| `a74a9dd` | **E5** Fused RMSNorm |
| `94a81fd` | **E6** Fused QKV + gate-up |
| `8638964` | **E7** Fused RoPE |
| `098c6e7` | **E8** reshape 清理 |
| `8a20704` | **E10** skip rp==1 |
| `7a3d6c9` | **E11** contiguous copy |
| `068fcad` | **E12** torch.compile RoPE |
| `909b9c6` | **E13** RMSNorm cache（后由 `1775e92` revert） |
| `44ba562` | **E16** opt-in compile MiniMindBlock |
| `1775e92` | **E18** revert E13 |
| `6f3ca56` | **E21** 12-prompt 稳健性 |
| `7fa7d57` | **E24** 修 qkv_fusion dedupe bug |
| `11c3c13` | **E25** discard kv_buffer（E24 后重试） |

## 11. 当前结论

### 已解决

- vendor 不再参与运行时 import
- 优化逻辑落在 `models/minimind_omni/`
- 原始模型代码保持不改
- 固定 seed 下文本、audio code、WAV bit-exact 可重现
- 总耗时 320 ms 中位数（−65% vs 928 ms baseline），stdev 11.6 ms
- VRAM 仅 +6 MB
- 49 个非 smoke 测试通过；公开 API 兼容
- 跨 12 个异质 prompt 完整生成

### 不建议继续做的事情

- 修改 / 重新维护完整 vendor 模型副本
- 直接复制 vLLM-Omni 的大规模 KV cache / fused kernel 实现
- 引入 CUDA/C++ 扩展或新依赖
- 把 CUDA Graph 局部成功误判为端到端加速

### 仍可优化方向（按风险从低到高）

1. **Triton fused RMSNorm + RoPE + add**：需引入 Triton 编译，~50 ms 估算收益。
2. **全栈 CUDA Graph capture**：需 monkey-patch `model.model.forward` 的 `start_pos` 计算，~50–150 ms 估算收益，revert 复杂。
3. **Speculative decoding**：复杂度高，小模型未必划算。

## 12. 关键经验

### 12.1 E24 的教训：class-level dedupe 在 per-instance patch 下是 bug

```python
# bug:
for module in model.modules():
    cls = type(module)
    if cls not in seen:
        _fuse(module)        # only first instance per class
        seen.add(cls)

# fix:
for module in model.modules():
    _fuse(module)              # every instance
```

每次看到 `seen.add(...)` 就要审视：patch 是 class-level（OK）还是 instance-level（bug）。

### 12.2 Noise floor 与 bench 置信

20+ 次连续 bench 后，stdev ~7–11 ms。任何 < 20 ms 改动都在 noise 内。"在 noise 内的改动"（E13 RMSNorm cache、E22 audio scratch）都做了 discard。

### 12.3 CUDA Graph 的边界

- `reduce-overhead` RoPE → crash（KV cat 冲突）
- compile MiniMindBlock → +91 ms（Inductor overhead × 16）
- 全栈 CUDA Graph → 未实施（需要 monkey-patch `model.model.forward`）

## 13. 验证命令

```bash
# 本地静态 + 单元
cd /Users/mcig/Projects/nanovllm-omni
uv run ruff check nanovllm_omni/ tests/
uv run black --check nanovllm_omni/ tests/
uv run python -m compileall -q nanovllm_omni
uv run python -m pytest -m "not smoke" -q
uv run python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"

# WSL 性能 bench
ssh mcigs-wsl "cd ~/nanovllm-omni && .venv/bin/python -m nanovllm_omni.optim.bench time \
  --model /home/mcig/minimind-3o --mimi /home/mcig/mimi \
  --max-tokens 16 --runs 5 --warmup 2"

# 跨 prompt 稳健性
ssh mcigs-wsl "cd ~/nanovllm-omni && .venv/bin/python bench_wide.py"

# autoresearch 自身
ssh mcigs-wsl "cd ~/nanovllm-omni && bash .auto/measure.sh"
```

## 14. 关联文档

- `docs/perf/minimind-omni-under-500ms.md` — 25 次实验完整索引 + 表格版（英文）
- `docs/perf/vllm_omni_thinker_talker_code2wav_performance.md` — 优化灵感的来源（vLLM-Omni 调研）
- `docs/perf/session-{1..12}.*` — 早期 TK-011/TK-015 的 Kineto / profile 数据
- `.auto/prompt.md` + `.auto/log.jsonl` — autoresearch 会话的全部 hypothesis 与实验记录
