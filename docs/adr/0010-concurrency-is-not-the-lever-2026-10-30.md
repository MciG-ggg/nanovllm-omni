# ADR 0010: 并发不是这里的吞吐杠杆（2026-10-30）

- Status: Accepted
- Date: 2026-10-30

## 被检验的前提

ADR-0008 / ADR-0009 都指向同一个诱人的推论：

> 三个族的单步开销里 GPU 只忙 5–10%，所以让多个请求并发就能把空闲的 GPU 用起来。

ADR-0007 还说这条路的代价是"先清掉三处进程级全局"（fork 的 `Context`、
`runner.sampler` 的就地替换、全局 RNG）。本文先测这个推论是否成立，
再决定要不要付那个代价。

## 实验

单进程、同一个 `Omni(pipeline="minimind_o")` 实例、两个 prompt、同 seed：

1. 先对两个 prompt 各跑一次（warmup，把建图与权重加载排除在计时外）；
2. **串行**：依次跑两个 prompt，记 wall；
3. **并发**：`threading.Thread` 同时跑两个 prompt，记 wall、比对音频、抓完整栈。

硬件与代码同 ADR-0007（RTX 3050 4GB / WSL2 / `perf/colab-2x` @ `c449431`）。

## 结果

| 组 | wall | 每请求 | 输出正确性 |
|---|---:|---:|---|
| 串行（2 个请求） | 2653 ms | ≈1327 ms | 基准 |
| 并发（2 个线程） | 2376 ms | ≈2376 ms（只有一个跑完） | 一个崩溃、一个与串行相同 |

**没有任何吞吐收益**：并发那一臂还崩了一个请求，等于用 2376 ms 完成了串行
1327 ms 的工作量，单请求成本约 **1.8× 恶化**。

### 失败点（崩溃）

```
File "nanovllm_omni/models/minimind_omni/thinker.py", line 114, in forward
    o = self.attn(q, k, v)
File "third_party/nano-vllm/nanovllm/layers/attention.py", line 198, in forward
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
File "third_party/nano-vllm/nanovllm/layers/attention.py", line 54, in store_kvcache
    assert slot_mapping.numel() == N
AssertionError
```

另一个线程在 `set_context` 与 attention 调用之间覆盖了模块级 `Context`
（`third_party/nano-vllm/nanovllm/utils/context.py:22`），这次 forward 读到的
是别人的 `slot_mapping`，长度对不上。

## 为什么并发注定无效（根因）

三处共享状态是**可修的表症**，真正否掉这条路的是执行模型：

- 三个族的瓶颈都是**每步上千次 Python/aten 派发**（minimind thinker 306 次/步、
  smolvla flow 2383 次/步），不是 GPU 算力。
- 这些派发在 CPython 里持有 GIL，换线程不会并行，只会交错执行同一份单线程工作。
- 结果就是实测的 1.8× 恶化：没有重叠可以捡，只多了线程切换与争用。

要真正并行 host 侧，只能多进程；但 4 GB 卡装不下两份模型（minimind ≈1.0 GB、
smolvlm ≈1.6 GB、smolvla ≈2.15 GB），smolvlm 与 smolvla 两份立刻越界。

## 决定

**不做并发重构**。因此也**不提交**"Context 改 thread-local"这类前置改动：
没有第二个真实使用者（AGENTS.md：抽象只在出现第二个真实使用者时引入），
它不会带来任何吞吐，只会增加一层无人使用的机制。

`Context` 的全局性、KV 张量的共享、`runner.sampler` 的就地替换、全局 RNG
仍作为已知约束记录在此，供将来若出现"多进程 + 足够显存"的场景时参考。

## 剩下的唯一杠杆

在"单进程、单卡 4 GB、Python 派发主导"这个前提下，吞吐只能靠**减少派发次数**：

1. **CUDA Graph**：minimind thinker 已验证 3.8×（ADR-0007，opt-in）；
   smolvlm decode 3.5×（ADR-0008）。smolvla 的 flow step 被 lerobot 内部
   一次未 pin 的 H2D 拷贝挡住（ADR-0009），需要改依赖才能继续。
2. **统一 graph 决策**：把 eager/graph 收进 deploy 一处 + 建逐位 parity 闸门，
   消除"bench 测 eager、引擎跑 graph"这类分裂（ADR-0008 结尾）。
3. 任何"多发一个请求就更快"的方案（批处理、流水线）都先要过本文这道门：
   批处理在 fork 里是真的（`Scheduler` 能批 prefill/decode），但本仓库的
   `decode_minimind` 每请求新建 `Scheduler` 且 deploy 把 `max_num_seqs` 锁在 1，
   要开它同样属于设计级改动，且必须重测。

## 限制

1. 只测了 `minimind_o` 一个族。但它与 smolvlm / smolvla 的瓶颈类型相同
   （host 派发主导），结论按类型外推；若某族 GPU 占比升高，需重测。
2. 两次请求、两个线程、一个 prompt 对，样本小；但"1.12× wall 且崩一个请求"
   距离任何有价值的收益都很远，扩大样本不会翻转结论。
3. 原始脚本是本地产物（`.scratch/concurrency_probe.py`），不进 git。
