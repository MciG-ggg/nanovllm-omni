# MiniMind-O thinker 的 MLX 轨首个数字（2026-11-09）

回答一个问题：MiniMind-O 的 thinker 在 M2 Air 的 MLX 上能不能跑、跑多快。
结论是**能跑**：单序列 32 token 的 TTFT 中位 6.06 ms，decode 4.33 ms/步；
4 条序列的 dense batch 每步 6.63 ms，TTFT 20.77 ms。

**这些是 MLX 轨的单机数字，不是与 CUDA 轨的对比**（ADR 0014：不跨机相减）。
本机没有 CUDA 基线，ADR 0021 的「token 级一致」尚未验证。

## 测量条件

| 项 | 值 |
|---|---|
| 硬件 | MacBook Air M2（Apple M2，10 核 GPU，16 GB 统一内存），macOS 27.0.1，Metal 4 |
| 工作集上限 | `mx.device_info()["max_recommended_working_set_size"]` = 11.84 GiB |
| 代码版本 | `698856c`（本文件所在分支） |
| 运行时 | mlx 0.32.4，mlx-lm 0.32.2，Python 3.12 |
| 模型 | `jingyaogong/minimind-3o` 的 thinker，转换为 `~/models/minimind-3o/mlx-thinker`；checkpoint bf16，加载为 fp16 |
| 输入 | `bench/prompts.py::medium_01`，chat template，`open_thinking=False`，45 token |
| 采样 | `temperature=0.0`（按 CUDA thinker 钳到 `1e-5`）、`top_p=1.0`、`repetition_penalty=1.05`、`max_tokens=32`、`seed=42` |
| runs / warmup | 5 / 2 |
| 计时口径 | `time.perf_counter()`；TTFT = prefill + 首 token 采样，decode = 其后每次 forward + 采样 |

命令：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
    uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/bench_mlx_thinker.py --runs 5 --warmup 2
```

## 单序列

| 指标 | 值 |
|---|---:|
| 模型加载 | 5387.3 ms |
| TTFT | p50 6.06 ms（min 5.67，max 9.01） |
| decode | p50 4.33 ms/step（min 4.11，max 4.71） |
| 32 token 的推导 wall | ≈ 6.06 + 31 × 4.33 = 140.3 ms |
| 峰值 Metal 内存 | 145.5 MiB |
| 跑完后 active 内存 | 124.9 MiB |
| 输出 | 32 token（第 32 个是 EOS=2） |

输出原文：

```text
MiniMind-O模型在优势方面较为强大，能够处理大量数据，而传统语言模型的优势则相对较为有限。
```

TTFT 的 run-to-run 漂移比 decode 大：同一命令在不同 session 的 p50 见过
5.75 / 8.16 / 6.06 ms。无风扇 M2 Air 的短时预热与后台负载都会落进这一段，
所以只报区间，不把 1 ms 级差异当结论。

## Dense batch（4 序列）

输入长度 `[19, 45, 22, 53]`，左补齐到 53。命令在单序列命令后加：

```bash
--batch-prompts short_01,medium_01,short_02,medium_02 --max-tokens 32
```

| 指标 | 值 |
|---|---:|
| 模型加载 | 5247.9 ms |
| TTFT | p50 20.77 ms（min 19.31，max 22.94） |
| decode | p50 6.63 ms/step（min 6.15，max 6.98） |
| decode 步数 | 32（跑到最长成员的 `max_tokens`） |
| 每序列输出 token 数 | `[12, 32, 13, 32]` |
| 推导 wall | ≈ 20.77 + 32 × 6.63 = 232.9 ms |
| 峰值 Metal 内存 | 214.5 MiB |
| 跑完后 active 内存 | 133.9 MiB |

两个可以直接读出来的事实：

1. **4 行只让每步贵了 1.53×**（6.63 / 4.33）。固定开销占大头，dense batch
   在 4 行这个规模上是有收益的。
2. **静态批的浪费被量化了**：两条短请求在第 12/13 个 token 就结束，但整批仍跑
   32 步。这是 ADR 0015 拒绝 dense padded 的原因；真 CB 要在成员结束后释放
   slot，而那需要 varlen attention 与 paged KV。

## 验证

### 1. 权重转换可重复

同一份 checkpoint 用 `scripts/convert_minimind_thinker_mlx.py` 转两次
（`mlx-thinker` 与 `mlx-thinker-repro`），`model.safetensors` 的 sha256 相同：

```text
f0525e44619dad670a67a1f97447aab2196404b7b682708a4a0ccd2f30deb760
```

`tie_word_embeddings=True`（`lm_head.weight` 与 `model.embed_tokens.weight`
逐位相同），所以 `lm_head.weight` 未写入文件。

### 2. 模型结构对着 PyTorch 参照

同一台机器、同一组 token `[1, 80, 301, 108, 80, 916]`：

| 实现 | argmax | top-5 |
|---|---:|---|
| vendor `MiniMindOmni`（CPU，fp32，`trust_remote_code=True`） | 869 | 869, 108, 80, 1930, 118 |
| MLX `qwen3`（bf16） | 869 | 869, 108, 80, 1930, 118 |

这是**同机正确性检查**，不是性能对比。它排除的是「权重 key 映射错、q/k_norm
或 RoPE 结构对不上」这类会输出看似合理的垃圾的情况。

### 3. Dense batch 与单序列逐 token 相同

`short_01` + `medium_01`，相同采样配置，8 token：

```text
single1 [849, 658, 294, 4166, 2621, 5983, 705, 4151]
batch 1 [849, 658, 294, 4166, 2621, 5983, 705, 4151] MATCH
single2 [80, 301, 108, 80, 916, 48, 82, 1135]
batch 2 [80, 301, 108, 80, 916, 48, 82, 1135] MATCH
```

这个检查抓到了一个真实 bug：批路径 prefill 后没有把首个采样 token 写回
`Sequence.last_token`，导致首个 decode step 重复喂 prompt 末 token、后续输出整体
错一位。修复后才得到上面的 MATCH。

### 4. top-p 分支被执行

`--top-p 0.9 --temperature 0.7 --max-tokens 8` 正常出 token
（`MiniMind-O模型的`），说明 `argsort` / `cumsum` / `put_along_axis` 那条路径
不是死代码。

## 限制

- **没有 CUDA 基线**：ADR 0021 的 token 级一致未验证。本机的 PyTorch 参照是
  CPU vendor 实现，只能证明结构与权重映射正确。
- **seed 只是参数对齐，不是 RNG 流对齐**：CUDA 侧用 `torch.multinomial` +
  per-sequence `torch.Generator`，MLX 侧用 `mx.random.categorical` +
  per-sequence `mx.random.key`。相同 seed 不会产生相同随机流。要拿到跨后端
  token 级一致，需要两边共用同一个可复现的采样器（或只在 greedy 下比较）。
- **fp16 only**：bf16 在 M2 上可用但未测；fp32 未测。
- **未用 `mx.compile`**：backend 声明 `graph=True`，但这条 bench 是 eager。
  compile 的收益需要单独测量，不能引用 ADR 0015 的 4.91×（那是 512×512
  小算子链的数，不是模型级）。
- **dense batch 是过渡形态**：补齐按最长成员付费，且整批跑到最慢成员结束。
  这不是 ADR 0015 接受的最终形态。
- **无风扇机器的热漂移**：TTFT 的 p50 在不同 session 间可见 5.75–8.16 ms，
  decode 的 p50 在 4.11–4.82 ms/step。1 ms 级差异不应当作结论。

## 原始产物

bench 输出只在本文件里版本化；权重与转换结果在 `~/models/minimind-3o/`，
不进 git。
