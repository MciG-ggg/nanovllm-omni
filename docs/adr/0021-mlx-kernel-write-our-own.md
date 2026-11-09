# ADR 0021: MLX 的 varlen kernel 自研最小集，输出一致定义为 token 级

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0013（backend 接缝）、0014（双轨目标）、0015（MLX 要 varlen）、0019（两轨并行）

## 问题

ADR 0015 定了"MLX 必须有 varlen"，并把路线写成"移植 `vllm-metal` 的 attention 子系统"
（`pagedattention.metal` 125 KB + block table + `cu_seqlens` + KV 写入 + rope，
外加 52 个相关 Python 文件）。

复核后这条路偏重了：`vllm-metal` 的 attention 子包里绝大多数是**我们用不到的东西**
——MLA、GDN、bidi、linear attention、Mamba2、KDA、turboquant、M5 NAX。
我们的模型只有 4 层、768 hidden、单个 head 组。

同时，ADR 0015 当时留了一个未定的验收标准：「要求输出与 CUDA 一致」。

## 决定

### 一、kernel 自研，只写自己需要的

不移植 `vllm-metal`，自己写一个**最小集**。需要的能力：

| 能力 | 说明 |
|---|---|
| paged decode attention | q 是每序列 1 个 query；k/v 按 block table 从 KV 里读；按各序列的 context_len 屏蔽 |
| paged prefill attention | q 是本步所有 scheduled token 拼成的扁平序列，用 `cu_seqlens` 划界 |
| KV 写入 | 把新算出的 k/v 写进 block table 指定的 slot（对应 fork 的 `store_kvcache`） |
| RoPE | 优先用 MLX 自带的（`mx.fast.rope` 一类），不自写 |
| reshape / 布局转换 | 把 `[B, H, T, D]` 与 `[T, H, D]` 之间来回搬 |

**明确不做**：MLA、GDN、bidi、linear attention、Mamba2、KDA、turboquant、NAX。

这个规模比移植小一个量级，也让「教学项目」的名副其实——kernel 的每一行都能解释。

### 二、「输出一致」的定义

**逐位一致做不到**：Metal 与 CUDA 的浮点归约顺序不同，fp16/bf16 的舍入也不同，
kernel 的分块策略还不同。同一份权重、同一 seed，两边生成**不同的 token** 是可能的。

可执行的标准定为（2026-11-09 修订，见「上游怎么做」一节）：

> **三档判据 + greedy 优先**
>
> - **greedy（`temperature=0`）是首选的比较方式**：没有随机流，token 完全可比。
> - 判据分三档，照 vLLM 自己的跨实现标准（ADR 0011 引的 `check_logprobs_close`）：
>   - `EXACT`：token 序列完全相同，且长度都到 `max_tokens`。
>   - `TOP_K_MATCH`：在**第一个分歧位**上，双方的 token 都在对方的 top-k logprobs 里。
>   - `FAIL`：其它。
> - 需要采样随机性时（同 seed 同 token），才需要统一采样器实现。

这落在 ADR 0011 的 **Compatible** 层（任务效用与安全不劣化），不是 Exact 层。
Exact 层的逐位一致只在**同一后端内部**成立（重写版 vs vendor，ADR 0017）。

**这不是妥协，这正是要教的东西**：跨后端 serving 的一致性只能到 token 级，
比同后端低一档。想守住这一档，靠的是采样参数与停止条件的精确对齐，
而不是指望两边浮点一样。

### 上游怎么做（2026-11-09 查证 `vllm-metal`）

`vllm-metal`（Apache-2.0）的"一致"分三层，**跨厂商那层最粗**：

| 层 | 手段 | 判据 |
|---|---|---|
| 同 MLX 生态内的实现对齐 | `tools/parity_utils.py::parity_report` | `EXACT` / `TOP_K_MATCH` / `FAIL` |
| 上游 CUDA 代码的移植保真 | `tests/test_kv_offload_cuda_parity.py` | 比对 `compute_sub_block_ptrs` 的**纯指针算术**，不是输出 |
| 跨厂商输出质量 | SiliconBench（arXiv 2609.19169） | **分类任务** vs NVIDIA 参考 → 任务级 fidelity |

> **这一节里提到的路径都在 `vllm-project/vllm-metal` 仓库里，不是本仓库的。**
> 本仓库要实现同一套判据（尚未开工），落点在 `nanovllm_omni/backend/` 与 `tools/`。

SiliconBench 原文："We use a **classification task to check for quality regressions against
an NVIDIA reference**."——**没有任何一处是拿 Apple Silicon 的输出和 NVIDIA 的 token
逐个对齐的**。

两条可直接抄的东西：

1. **三档判据就是 vLLM 自己的标准**。`TOP_K_MATCH` 让失败有可操作的信息
   （分歧位上双方各自的 top-k 是什么），比"token 不一样"有用得多。
   证据用 OpenAI API 的 `logprobs` 取 top-k 候选。
2. **他们绕开 RNG 流问题的办法是 greedy**：`tools/dspark_parity.py` 与
   `tools/check_parity.py` 都写死 `temperature=0.0` / `top_p=1.0` / `top_k=0`。

**本 ADR 的修订就是从这两条来的**：先前写的"相同 seed、相同采样配置下 token 序列相同"
把 greedy 排除在外，等于把"统一采样器"当成前置条件；实际上 greedy 就能比，
统一采样器只在"同 seed 同 token"这个更强的诉求下才需要。

## 后果

- **ADR 0015 的路线要改**：从"移植 `vllm-metal`"改成"自研最小集"。
  ADR 0015 的**结论不变**（MLX 必须有 varlen），变的是实现路径与成本量级。
- **自研 kernel 的验收要对着 CUDA 轨的 token 序列**，不是对着数值。
  这要求两边的采样配置严格对齐（temperature / top-k / top-p / seed / max_tokens），
  否则连 token 级都保证不了。
- **MLX 轨第一站仍是 SmolVLM**（没有 talker 依赖，能独立成链）——
  但 SmolVLM 的视觉塔是 SigLIP（第三方库），MLX 版要走 mlx-vlm 的移植。
- **两轨的对比口径**：MLX 轨出数字后，与 CUDA 轨比较的是**趋势**
  （ragged 批 vs 齐长批的相对变化），不是绝对值（ADR 0014：不跨机相减）。

## 边界

- 本 ADR **不**承诺 Metal kernel 的性能能对上 `vllm-metal`。对标是"功能等价"，
  不是"速度等价"。
- 本 ADR **不**处理 M5 NAX 路径（M2 Air 没有 tensor 单元）。
- 本 ADR **不**改 ADR 0014 的双轨结论，也不改 ADR 0019 的分工。

## 实施记录（2026-11-09）：三档判据落地，同 MLX 生态内的实测结论

`tools/parity_check.py` 把上面第二节的三档判据做成可跑的工具。**参考臂是 `mlx-lm`
自己的 AR 循环**（`mlx_lm.generate.generate_step` + `greedy_sampler`），被测臂是本仓库的
`MlxStageRunner`。两个臂各加载一份权重，互不共享状态。

### 环境与口径

| 项 | 值 |
|---|---|
| 机器 | Apple Silicon M2 Air / 16 GB |
| 版本 | mlx 0.32.4、mlx-lm 0.32.2、Metal toolchain `32023.921` |
| 权重 | `~/models/minimind-3o/mlx-thinker`（8 层、hidden 768、8 heads、4 kv heads、head_dim 96、vocab 6400、`tie_word_embeddings=true`、`rope_theta` 1e6） |
| dtype | `float16`（`float32` 也跑过，结论相同） |
| 采样 | greedy：`temperature=0.0` / `top_p=1.0` / `top_k=0`，无随机流 |
| 输入 | `nanovllm_omni/bench/prompts.py` 的 6 条 `BENCH_PROMPTS`，`open_thinking=False` |
| `max_tokens` | 64 |
| top-k | 5 |

`logprobs` 两臂都拿得到，**没有退化**：参考臂直接收 `generate_step` 每步 yield 的
`logprobs`（`[vocab]`，已做 log-softmax）；被测臂从 `runner.forward()` 返回的 logits
自己算 `log_softmax`（fp32，避免 fp16 舍入自己决定排序）。工具在任一侧拿不到
`logprobs` 时会退化成只报 `EXACT` / `FAIL` 并把原因写进 `reason`，这次没有触发。

### 结论：`TOP_K_MATCH`，分歧位 `system_01@17`

6 条 prompt 里 **5 条 `EXACT`**（`short_01` / `short_02` / `short_03` / `medium_01` /
`medium_02`），1 条 `TOP_K_MATCH`：

```
prompt system_01 (67-token prompt, 64 generated)
  verdict: TOP_K_MATCH at token 17
    reference picked 2205, top-k 2205:-1.6172 2210:-1.6172 3648:-1.8672 697:-2.5312 607:-2.5391
    candidate picked 2210, top-k 2210:-1.6158 2205:-1.6236 3648:-1.8736 607:-2.5377 697:-2.5377
```

一个 token 之差，且双方都落在对方的 top-2 里（参考臂的 2205/2210 在 fp16 下**连 logprob
都打平**，都是 `-1.6172`）。按判据这是 `TOP_K_MATCH`，不是 `FAIL`。

### 分歧的根因：prefill 形状，不是 decode 数学

不是 kernel 或采样器算错，是**两份代码喂给 `mx.fast.scaled_dot_product_attention` 的
prompt 长度不同**：

- `mlx_lm.generate.generate_step` 分块 prefill：先跑 `prompt[:-1]`，再把最后一个 token
  当一步 decode（`prefill_step_size=2048` 时 67 token 的 prompt 走 `66 + 1`）。
- `MlxStageRunner.prepare_prefill` 一次跑完 67 个 token。

同一份权重、同一个模型对象，只改 prefill 形状，**raw logits 就不再逐位相同**：

| dtype | 67 一次 prefill vs 66+1 | 首步 `max|diff|` |
|---|---|---|
| float32 | 不完全相同 | 1.29e-05 |
| float16 | 不完全相同 | 7.81e-03 |

逐层对比最后一位置（fp16），差异从第 0 层就出现并逐层放大，**embedding 行完全相同**，
所以差异全部来自 attention 与 MLP 在**不同序列长度**下的归约顺序，不是权重或 RoPE：

```
embed        max|diff|=0            identical=True
layer0-out   max|diff|=0.00195312   identical=False
layer2-out   max|diff|=0.0078125    identical=False
layer5-out   max|diff|=0.125        identical=False
layer7-out   max|diff|=0.0625       identical=False
```

反向验证：把被测臂的 prefill 也切成 `66 + 1`（`--split-prefill`），**6 条全是 `EXACT`**。
所以这一档分歧的完整归因是「prefill 分块形状 → fp16 归约顺序 → 一个 token」，
跟 MLX backend 的 kernel 正确性无关。

### 复现

```bash
PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
    uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/parity_check.py --max-tokens 64
# 全 EXACT 的对照：
PYTHONPATH=$PWD:$PWD/third_party/nano-vllm \
    uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/parity_check.py --max-tokens 64 --split-prefill
```

### 实测到的 API 签名（写在报告里，避免下次再猜）

```python
mlx_lm.utils.load(path_or_hf_repo, tokenizer_config=None, model_config=None,
                  adapter_path=None, lazy=False, return_config=False,
                  revision=None, trust_remote_code=False) -> (model, tokenizer)
mlx_lm.generate.generate_step(prompt: mx.array, model, stream=..., *, max_tokens=256,
                              sampler=None, logits_processors=None, max_kv_size=None,
                              prompt_cache=None, prefill_step_size=2048, kv_bits=None,
                              kv_group_size=64, quantized_kv_start=5000,
                              prompt_progress_callback=None, input_embeddings=None
                              ) -> Generator[tuple[mx.array, mx.array]]  # (token, logprobs)
mlx_lm.generate.generate(model, tokenizer, prompt, verbose=False, **kwargs) -> str
mlx_lm.sample_utils.greedy_sampler(logprobs) -> mx.argmax(logprobs, axis=-1)
mlx_lm.sample_utils.make_sampler(temp=0.0, top_p=0.0, min_p=0.0, min_tokens_to_keep=1,
                                 top_k=0, xtc_probability=0.0, xtc_threshold=0.1,
                                 xtc_special_tokens=None) -> Sampler
qwen3.Model.__call__(self, inputs, cache=None, input_embeddings=None)
```

`mlx_lm.generate` 是**模块**不是函数，`from mlx_lm import generate` 拿到的是
`mlx_lm.generate` 函数本身（`mlx_lm/__init__.py` re-export），要拿 `generate_step`
必须 `importlib.import_module("mlx_lm.generate")` 或 `from mlx_lm.generate import ...`。

`load` 默认给 `bfloat16`（checkpoint 存的），两个臂都显式 `set_dtype(mx.float16)`
才能保证比的是同一份数值。

### 这一轮没做的

- **没有 CUDA 轨的对照**：按 ADR 0014 / 任务约束，CUDA 基线要等 WSL。
- **没跑 `float32` 的全套 prompt**：只在 `system_01` 上验证了根因与 dtype 无关，
  没有把 6 条都跑一遍 fp32 拿 `EXACT`。
- **被测臂走的是 `MlxStageRunner` 的公开入口**（`prepare_prefill` / `forward` /
  `prepare_decode`），**不是自研 paged kernel 那条路**。ADR 0023 的
  `mlx_kernel/` 没有接进这条 parity 链，所以这个结论**不能**当作自研 kernel 的验收。

