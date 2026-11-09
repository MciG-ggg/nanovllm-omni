# ADR 0017: talker 的模型代码自己写，窗口化 KV 一步到位

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0007（talker 成本不在 O(n²)）、0009（smolvla 吞吐）、0011（parity 标准）、
  0013（backend 接缝）、0016（serving 是教学对象）

## 问题

talker 是这个仓库的 84% 瓶颈（ADR 0012：2360 / 2810 ms），但它的代码**不在本仓库**：

```python
# nanovllm_omni/models/minimind_omni/stage.py:432
talker_cls = get_class_from_dynamic_module("model_omni.TalkerModule", model_dir)
```

`transformers` 从 checkpoint 目录动态加载 `model_omni.py`。本仓库只调用它：

```python
# stage.py:477-483
for layer in talker.layers:
    hidden, _ = layer(hidden, position_embeddings, past_key_value=None, use_cache=False)
```

要让 talker 变快（ADR 0012 指出批处理是 CB 的前置，而 ADR 0007 指出窗口化 KV 是定长的前提），
就必须改这段代码。三条路：

| 做法 | 问题 |
|---|---|
| monkeypatch `talker.layers[i].self_attn.forward` | 依赖 vendor 的内部结构与 forward 签名；vendor 一改就哑；patch 逻辑本身在教学项目里不可读 |
| 把 `model_omni.py` 拷进本仓库 | 仍然是 vendor 代码，只是换了个位置；checkpoint 升级会漂移，要自己维护同步 |
| 分桶批处理，绕开改动 | 拿不到定长，拿不到 CUDA Graph；批利用率还受 prompt 长度分布限制 |

## 决定

**talker 的模型代码全部自己写。** 本仓库不再在运行时加载 checkpoint 目录里的模型代码；
除了 fork 的 nano-vllm 之外，所有模型代码都是本仓库的代码。

窗口化 KV 在重写时**一步到位**，不做分桶过渡：

- cache `[0, step-8)`，每步只重算最后 8 列 → query 维度**定长 8**
- 定长 8 带来三件事：可 varlen（每请求 cache 长度不同正好交给 `cu_seqlens_k`）、
  可上 CUDA Graph（形状恒定）、FLOPs 从 O(n²) 降到 O(8n)

ADR 0007 当年否掉窗口化 KV 的理由是"降 FLOPs 换不来时间"（talker 是 dispatch-bound）。
**那个理由在批处理场景下不成立**：窗口化的价值不是省 FLOPs，是让 talker 变成定长，
从而能被 varlen 批处理和图捕获覆盖。

## 后果

- **重写要一次做到位，不要分两期**。窗口化 KV 会改变 attention 的调用方式
  （带 cache、只算最后 8 列），先写个 dense 版本再改窗口等于写两遍。
- **顺带白捡两件事**（都在重写时自然成立）：
  - `lm_head` 只算最后一个位置而不是整条序列——采样只取 `logits[0, -1, :]`
    （ADR 0007 的分解里写的就是"全 lm_head"）。逐位置独立的 Linear 上这是逐位等价的，
    FLOPs 少 T 倍。
  - 每序列一个 RNG generator（像 thinker 那样），批内同 seed 的请求抽到的随机数
    不依赖它和谁同批——这是"同 seed 同输出"的前提。
- **parity 要对着 vendor 输出锁死**（ADR 0011 的 Exact 层）。重写后 talker 的
  `audio_codes` 必须与 vendor 逐位一致。已有的
  `tests/test_minimind_talker_early_exit.py` 这类测试要改成对着重写后的实现跑，
  并保留一份对着 vendor 的一次性比对基线。
- **checkpoint 目录只剩权重**，不再有可执行代码。少一类运行期依赖，
  也少一个"升级 checkpoint 就炸"的来源。
- **这条同时是 MLX 轨的前提**。talker 重写之后，它的实现不再被绑死在
  vendor 的 HF 风格上，才谈得上写一个 MLX 版。ADR 0015 说的「MLX 轨缺 talker」
  因此有了解法方向。

## 边界

- 本 ADR 只定 talker。**其余族的外部代码依赖（`code2wav` 用的 transformers
  `MimiModel`、`smolvlm` 用的 SigLIP、`sd_turbo` 用的 diffusers、`smolvla` 用的 lerobot）
  是否也在"自己写"的范围内，尚未决定**——它们不是 vendor 代码而是第三方库，
  成本量级完全不同。
- 重写不等于逐字拷贝 vendor 源码：是按 vendor 的语义写一份自己的实现，
  保持数值行为一致，但代码结构服从本仓库的风格（AGENTS.md 的可读性要求）。
