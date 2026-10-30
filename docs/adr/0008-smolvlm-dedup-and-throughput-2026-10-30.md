# ADR 0008: smolvlm 去重、图片路径与吞吐（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

## 问题

`smolvlm` 族此前无法处理图片，且显存与吞吐都没有可用基线。本文记录根因、
修复（`d1406e5`）与修复后的实测。

## 根因（修复前）

1. **图片路径直接崩溃**。processor 返回 `pixel_values` 形状
   `[num_images, num_tiles, 3, H, W]`（一张 512×512 图 → 17 个 tile），
   stage 把它原样交给 SigLIP，后者按 4-D 解包 →
   `ValueError: too many values to unpack (expected 4)`。
   同处还把 `image_embeds[0]` 填进**所有** image-token 位置，17 个 tile 只有第 1 个生效。
2. **模型被构造了两份**。stage 自己建一份
   `SmolVLMForConditionalGeneration`，fork `StageRunner` 内又建一份。
   fork 按剩余显存切 paged KV 池，因此 `num_kvcache_blocks` 只剩 **1**
   （256 token），而一个图片请求需要 1088 个 image token —— 图片永远排不进 KV。
3. **graph 的序列窗口太小**。`graph_max_seq_len` 用 fork 默认 512，
   图片（1088 token）的每一步 decode 都退回 eager。

## 修复（`d1406e5`）

- 复用 runner 的模型（`self.model = self.stage_runner.model_runner.model`），去掉第二份副本。
- `pixel_values.flatten(0, 1)` 把 tile 轴折进视觉塔的 batch 轴；connector 输出
  `flatten(0, 1)` 后按 image-token 位置整段写入。
- `max_capture_bs=1`（每个 stage 调用只调度一个 Sequence）、
  `graph_max_seq_len=2048`（一图 + 512 生成 token）。
- `extra["ignore_eos"]` 让吞吐测量能固定输出长度：默认开启 EOS 时短 prompt 两个
  token 就结束，标着"64 新 token"的行其实测的是两 token 的请求。

## 验证（RTX 3050 4GB / WSL2 / torch 2.11.0+cu130 / 修复后的树）

单进程构建 stage，一次 warmup 后测三次取中位；`ignore_eos=True`、
`num_new_tokens=32`、`temperature=0`、`seed=7`。

| 断言 | 修复前 | 修复后 |
|---|---|---|
| `num_kvcache_blocks` | 1（256 token） | **49（12544 token）** |
| 图片请求 | `ValueError` 崩溃 | **跑通**（1512 ms，峰值 1922 MiB） |
| 加载后显存 | ≈2189 MiB 峰值（含副本） | **1587.6 MiB** |
| 参数 dtype | bf16 | bf16（未受影响） |

吞吐：

| cell | 输入 | wall p50 | tokens/s | 峰值显存 |
|---|---|---:|---:|---:|
| graph | text | 728.2 ms | 43.9 | 1587.9 MiB |
| graph | image（512², 17 tiles） | 1511.9 ms | 21.2 | 1922.1 MiB |
| eager | text | 2543.6 ms | 12.6 | 1578.8 MiB |
| eager | image | 3341.0 ms | 9.6 | 1913.0 MiB |

decode graph 相对 eager：文本 **3.5×**、图片 **2.2×**。

## 这次的教训：同一个决定散在三处

> 2026-10-30 已修：bench 不再固定传 eager，改为跟 stage 默认（`--no-cuda-graph`
> 才测 eager cell）。修复后同一套 bench 命令的两臂实测：
>
> | 输入 | graph | eager |
> |---|---:|---:|
> | short | 112.3 ms | 218.2 ms |
> | medium | 1452.2 ms | 5501.2 ms |
> | long | 1452.8 ms | 5639.4 ms |
>
> 即 medium/long 上 **3.8×**，与 ADR-0003 在 T4 上的 3.16× 同量级，也证实了
> 旧记录里的“241 ms”确实测的是引擎不走的 eager 路径（且那还是早停后的 input）。

原本的分布是这样：

| stage | 默认 | 该族的 bench 传什么 |
|---|---|---|
| minimind thinker | **eager**（graph 需 opt-in） | 与默认一致 |
| smolvlm decode | **graph** | **固定传 eager** |
| smolvla flow | 无开关 | — |

后果：ADR-0003/0006 里 smolvlm 的 2181–2189 MiB / 241 ms 测的是 **eager 路径**，
而引擎默认走 graph。这不是"某个模型慢"，是同一个决定没有单一归属。

下一个改动应当是：把 eager/graph 收进 deploy 一处，并建一个
`graph vs eager` 的逐位 parity 闸门（同 `(输入, seed)` 输出相同才允许开默认），
三个 stage 共用，而不是各配一个 flag。**在闸门存在之前，不改任何默认值。**

## 限制

1. 单请求串行入口，吞吐即 1 / wall；没有批处理，`max_concurrent` 不产生真实并发
   （fork 的 `Context`、sampler、RNG 都是进程级全局）。
2. 图片是合成的 512×512，只保证 tile 数与时间稳定，不代表真实任务分布。
3. 显存口径为 `torch.cuda.max_memory_allocated()`；`total_ms`/CUDA event 的已知问题
   见 ADR-0007。
4. 原始日志与脚本是本地产物（WSL `/tmp`、`.scratch/`），不进 git。
