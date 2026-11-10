# ADR 0024 — code2wav 不写 MLX 版，Mimi 用 torch 跑在 MPS 上

- 状态：Accepted
- 日期：2026-11-10
- 相关：ADR 0013（backend 接缝）、ADR 0023（MLX 轨顺序）

## 决定

code2wav 阶段保持 torch 实现，不移植到 MLX。Mac 上让 Mimi 跑在 MPS
（之前是 CPU，且 MPS 路径直接崩）。MLX 轨的吞吐结论里，code2wav 单独按
设备记账，不和 thinker / talker 的 MLX 数字混在一起。

## 为什么

先量，再决定。M2 Air、80 帧（约 6.4 秒音频）、8 码本：

| 段 | MPS p50 | 占解码 |
|---|---:|---:|
| quantizer.decode | 1.16 ms | 0.6% |
| upsample | 0.47 ms | 0.2% |
| decoder_transformer（8 层） | 16.72 ms | 8.6% |
| decoder 卷积栈（15 层） | 170.81 ms | 88% |
| **整段 decode** | **193.92 ms** | |

再往里拆，卷积栈里 4 个 `ConvTranspose1d` 吃掉 86.2 ms——不是那两层
普通卷积（0.47 + 2.70 ms）。

于是问题变成「MLX 的转置卷积能不能比 torch MPS 快」。同样的形状
（`in=1024, out=512, k=16, stride=8, T=160`）单算子实测：

| 实现 | p50 |
|---|---:|
| torch MPS `ConvTranspose1d` | 20.56 ms |
| MLX `conv_transpose1d` | 21.98 ms |

**同一量级，MLX 还略慢。** 转置卷积在 MLX 上走的也是同一条代价路径，
移植换不来吞吐。

再换个角度：code2wav 在整条 Mac 链里的占比。thinker 走 MLX（prefill
10.53 ms、decode 4.44 ms/步，16 新 token ≈ 82 ms），talker 走 MLX
（22 帧 190.1 ms），code2wav 194 ms。**code2wav 占全链 42%。** 占得不小，
但省下它的唯一办法是让转置卷积更快，而上面那张表说 MLX 给不了。

## 代价

- Mac 上 code2wav 是 fp16 的 MPS 实现，CPU 与 MPS 的输出在同一精度下
  差 2.5e-06（fp32）/ 2.8e-03（fp16，波形峰值 0.88）。这些数字由
  `tests/test_code2wav_device.py` 钉住。
- 不能声称「整条 pipeline 都在 MLX 上」。文档里必须写清 code2wav 是
  torch 侧。ADR 0014 的「一条轨的进展不能默认另一条也成立」同样适用于
  这里的阶段划分。

## 顺带修掉的崩溃

`load_mimi_codec` 之前用 ``MimiModel.from_pretrained`` 的默认
``low_cpu_mem_usage=True``：模型先在 meta device 上搭好再落盘，MPS 上
落完之后 placeholders 没有存储，每次 ``decode`` 都报
``Placeholder storage has not been allocated on MPS device``。也就是
**Mac 上 code2wav 根本跑不了**，只是在 CPU 上跑时没被发现。现在 MPS
路径显式传 ``low_cpu_mem_usage=False``。

## 什么情况下重开这个决定

- 有实测证明某个 MLX 卷积实现（自写 Metal、或用 `mx.conv_general`
  换个分块）能在转置卷积上明显超过 torch MPS——先给单算子数字，再谈移植。
- 或者整条链的其他段快到让 194 ms 变成主要项。目前 thinker 82 + talker
  190 还比它大，先优化那两段更划算。
