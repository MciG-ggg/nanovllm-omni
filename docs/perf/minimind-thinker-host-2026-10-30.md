# MiniMind-O thinker 每步主机侧开销（2026-10-30）

回答一个问题：thinker 120 步 decode 里，**每步 19 ms 到底花在哪**，哪些能删。
结论是「只有重复惩罚那 0.33 ms 值得改」——另外三项候选实测为零或不可测，
而 85% 是 fork forward 的派发开销，不在本次改动范围内。

## 测量条件

| 项 | 值 |
|---|---|
| 硬件 | RTX 3050 4 GB Laptop（WSL2），CUDA 13.x，torch 2.11 |
| 代码版本 | `perf/colab-2x` @ `78121b6` + 本次改动 |
| 模型 | `jingyaogong/minimind-3o`（本地 HF 缓存，`HF_HUB_OFFLINE=1`） |
| 输入 | `你好，请用一句话介绍你自己。`（10 token prompt） |
| 采样 | `max_tokens=120, temperature=0.7, top_p=0.9, seed=7` |
| runs / warmup | 5 / 1（warmup 也设 seed，否则 talker 偶发 0 帧） |
| thinker 步数 | 120 步/请求（`ignore_eos` 填充时钟） |
| 峰值显存 | 1007.4 MiB（整条 pipeline，`max_memory_allocated`） |
| 阶段分解 | thinker decode 循环 2.32 s / 单请求 wall 7.0 s；talker + code2wav 占其余 ≈4.7 s |

## 每步开销分解（600 步 = 5 请求 × 120 步，`perf_counter` 逐项计时）

| 项 | ms/步 | 占循环 | 说明 |
|---|---:|---:|---|
| `run_model` | 16.32 | 84.4% | fork eager forward 的 CPU 派发，**非**算力 |
| `sample` | 2.17 | 11.2% | 含 `.tolist()` 的 GPU 排空等待；其中 rp 见下行 |
| ├ 重复惩罚 | 0.33 | 1.7% | 逐 token `logits_i[t] /= rp` |
| `prepare` | 0.30 | 1.5% | fork 每步新建 5 个 pinned 张量 + H2D |
| `bridge` | 0.13 | 0.7% | clone 整块 4096×768 buffer + `cat` |
| `sync` | 0.05 | 0.3% | 循环顶部的 `torch.cuda.synchronize()` |
| 其余 | 0.03 | 0.2% | `schedule` / `postprocess` / `reset_context` / `Scheduler()` |
| **合计** | **19.33** | 100% | |

`run_model` 的 16.3 ms 不是算力：单 token 算力量级 0.05 ms。逐项 profile（thinker
单独跑，`torch.profiler` CPU+CUDA）显示每步 **306 次 kernel launch**
（`cudaLaunchKernel` 223 + `cuLaunchKernel` 82）加 dynamo 每调用开销
（`TorchDynamo Cache Lookup` 49 次/步、`AOTDispatcher`、4 个 `CompiledFxGraph`）。
`enforce_eager=True`（ADR-0005：3050 上 graph capture 失败）把这段全留在 host。

## 改了什么

1. **重复惩罚改成一次张量操作** — `nanovllm_omni/models/minimind_omni/stage_runner.py:24-37`
   新增 `_apply_repetition_penalty`，`:106` 用它替掉原来的
   `for token in set(sequence.token_ids): logits_i[token] /= rp`。
   每个 token 的 `/= ` 是 2 次 kernel launch，改成 1 次 gather + divide + scatter。
   语义不变：每个元素仍被同一个标量除且只除一次，所以逐位相同。
2. **bridge buffer 只 clone 需要的行** — 同文件 `:147-157`。
   原来每步 `bh.detach().clone()` 整块 4096×768（≈6 MB，每请求 ≈720 MB 拷贝），
   现在先切片再 clone（decode 取 1 行，prefill 取 `seq_len` 行）。

未改动 `stage.py` / `bench/runner.py` / `thinker.py`（他人正在改）。

## 改前 / 改后

| 指标 | 改前 | 改后 | Δ |
|---|---:|---:|---:|
| thinker 每步 | 19.333 ms | 19.228 ms | −0.105 ms（−0.5%） |
| ├ 重复惩罚 | 0.330 ms | 0.225 ms | −0.105 ms（−32%） |
| ├ bridge clone | 0.132 ms | 0.131 ms | −0.001 ms（不可测） |
| 单请求 wall（中位） | 6949.3 ms | 6901.1 ms | −48 ms（噪声内） |
| 5 请求总耗时 | 35151.6 ms | 34795.3 ms | −356 ms（−1.0%） |

**每步的两个数字是可信的，wall 的不是。** 每项计时器把噪声隔离在单项之外：
重复惩罚的 −0.105 ms/步 与循环总时长同样 −0.105 ms/步 一一对应，两侧各跑两轮
（0.332/0.330 → 0.219/0.225）都复现。而 wall 的 5 轮中位数在 before/after 上重叠：
before 6949.3 / 7046.1，after 6901.1 / 7069.4，5 请求总和 35151–35712 vs 34795–35297。
也就是说 1% 的改动被 ±1.4% 的 run-to-run 漂移（WSL + 4 GB 卡的 clock 波动）盖住了，
**不能声称吞吐提升**。

## 没做 / 回退的项

- **去掉每步 `torch.cuda.synchronize()`**（0.053 ms/步，6.4 ms/请求 = wall 0.09%）。
  实测：`.tolist()` 每步都会把 stream 排空，所以顶部这个 sync 基本是空转。
  仍然**保留**——收益 0.09%，而它是 fork `prepare_*` 复用 pinned 缓冲时唯一的显式
  同步点，删掉等于让正确性依赖下游那句 `.tolist()`。
- **长生命周期 `Scheduler`**：`Scheduler()` + `BlockManager()` 实测 0.001 ms/步
  （0.3 ms / 5 请求），因为只按请求建一次。不值得为它引入跨请求 KV 残留的证明成本。
- **bridge 只 clone 需要行**：已改，但实测 0.001 ms/步——拷贝是异步的，GPU 本来就空闲，
  这 720 MB 完全藏在 CPU 瓶颈后面。保留是因为它只是把切片移到 clone 之前，
  不增加代码、不改变数值，但**它不是吞吐收益**。
- **dtype 不一致（只报告）**：`torch._dynamo hit config.recompile_limit (8)`，
  `rms_forward`，`expected Half, actual BFloat16`。fp16 权重 + bf16 logits 让 dynamo
  反复重编译后放弃，但每调用仍付 guard + AOT 开销。统一 dtype 会改数值，未动。
- **fork 每步 5 个 pinned 张量**（0.30 ms/步，36 ms/请求）：在 `third_party/`，
  本次未动。
- **`stage.py:322` 每请求 `AutoTokenizer.from_pretrained`**：只报告，未改。
- 另发现：talker 可能返回 0 帧 → `ValueError: audio codes must be non-empty`
  （warmup 未设 seed 时可复现）。既有缺陷，与本次改动无关。

## 限制

- 单进程串行（`PipelineExecutor(max_concurrent=1)`），所以「多请求吞吐」= 1/单请求
  wall；本次改动对它的影响在噪声内。
- 计时器本身有 ~0.01 ms/步 开销；带计时器的 wall 与不带计时器的 wall 不可直接比较。
- 120 步上限下 token set 只有 13 个；长 prompt / 高重复惩罚场景下重复惩罚项的
  收益会更大，未测。

## 复现

```bash
# 逐项分解：.scratch/measure_thinker.py（本地产物，未入库）
rsync -az --exclude .venv --exclude .git --exclude docs . mcigs-wsl:~/Projects/nanovllm-omni/
ssh mcigs-wsl 'cd ~/Projects/nanovllm-omni && HF_HUB_OFFLINE=1 \
  PYTHONPATH=$HOME/Projects/nanovllm-omni:$HOME/Projects/nanovllm-omni/third_party/nano-vllm \
  ~/venvs/nanovllm-omni/bin/python .scratch/measure_thinker.py \
  --model $HOME/models/minimind-3o --runs 5 --warmup 1 --max-tokens 120 --seed 7'
```

数值一致性：同一 `(prompt, seed)` 下 thinker `text_token_ids` 与 bridge hidden
（130×768，md5 `c136b7c85358`）改前改后逐位相同；整条 pipeline 的 WAV md5
在 5 次运行中一致（`621e03498e40`）。CPU 单测
`tests/test_minimind_repetition_penalty.py` 锁住重复惩罚的逐位等价。
