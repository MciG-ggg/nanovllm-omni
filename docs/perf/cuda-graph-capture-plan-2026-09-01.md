# CUDA Graph capture for `Omni.generate` — 设计 + 风险评估

> 状态：**设计就绪，未实现**。依赖 GPU 验证（WSL 3050 目标机 / Colab T4
> 暂用）。配套 CPU 测试：`tests/test_kv_fixed_buffer.py`（fixed-KV-buffer
> 机制数学等价性已验证）。
>
> 依据：`docs/perf/ncu-generate-kernels-2026-09-01.md` § 6.4 的硬证据——
> 11367 `cudaLaunchKernel`/generate 是结构性、跨平台稳定的成本；host
> dispatch 时间 117ms + GPU 空等时间 ~500ms，理论上限可省 ~73% 墙时。
> 该文档遗留 open item："风险评估仍需单独做"——本文补上。

## 1. 为什么是 CUDA Graph（ncu 硬证据回顾）

- 11367 `cudaLaunchKernel`/generate（3/3 run byte-identical），host dispatch
  总时间 ~117ms，GPU 大部分时间在等 host 排队而不是干活。
- 已试过的 micro-optimization 全部 ≤ 5ms 且在 noise 内：
  - qkv.view+slice（#2 discard：0 kernel count 变化）
  - RMSNorm `.to(fp32)` 移除（#3 discard：crash，op 要求 fp32）
  - 1744 direct_copy 静态 attribution（#6 keep：结构性，非链外可消）
- **唯一剩的大杠杆**：把 11367 launch 打包成 1 次 graph replay。

## 2. 核心机制：fixed-KV-buffer

CUDA Graph 要求 capture 期间 tensor shape 固定。当前 `_attention_forward`
每步 `torch.cat([past_kv, key], dim=1)` 让 KV 增长——**不可 capture**
（这也是 E14 `torch.compile(reduce-overhead)` crash 的直接原因：
"accessing tensor output of CUDAGraphs that has been overwritten"）。

替代：**预分配 max_seq 长度的 KV buffer**，每步 slice 写入 + slice 读取：

```python
# 初始化（每个 attention 层一份）：
#   past_k = torch.empty(B, max_len, n_kv_heads, head_dim)
#   past_v = torch.empty(B, max_len, n_kv_heads, head_dim)

# _attention_forward 内（替换 torch.cat）：
if past_key_value is not None:
    past_k, past_v = past_key_value
    cur_len = past_k_pos  # 记录已写位置
    past_k[:, cur_len : cur_len + sequence_len] = key
    past_v[:, cur_len : cur_len + sequence_len] = value
    key = past_k[:, : cur_len + sequence_len]  # view，shapes 静态
    value = past_v[:, : cur_len + sequence_len]
past = (past_k, past_v)  # 稳定 shape，长度由 pos 追踪
```

`tests/test_kv_fixed_buffer.py` 已验证：`slice-write + slice-view` 在
CPU 上逐位等于 `torch.cat`，且 16 AR steps 内不出界。

**注意**：slice-view 返回的 `key[:, :cur+1]` shape 是 `(B, cur+1, ...)`——
这是**动态 shape**（每次 cur+1 不同）。纯 CUDA Graph **仍不能直接 capture**
这个动态 view。解法二选一：

- **方案 A（pad 到定长）**：始终返回 `key = past_k[:, :]`（未用尾部是 0），
  attention mask 负责屏蔽未用位。shape 恒定 `(B, max_len, ...)` → 可
  capture。代价：每步重算整个 KV（attention 读全部 max_len），多算但
  launch 数骤减。
- **方案 B（每步 re-capture）**：KV 长度 cur 每步不同 → 每步 capture 一个
  graph（16 个 graph，各自固定 shape）在 decode 时切换。复杂度高，且
  prefill→decode 分界处要动态决定。

**推荐方案 A**：mask 复用现有 `attention_mask`（MiniMind-O 已支持），
graph 只需 capture 一次 decode 步（Q=1, KV=全量 max_len）。

## 3. Monkey-patch 范围

沿用现有 `nanovllm_omni/models/minimind_omni/attention.py` 的 patch 风格
（E24 教训：**per-instance，不要 class-level dedupe**）。

新增 patch（`enable_cuda_graph`）：

1. **KV buffer 预分配**：在 bundle load 时按 `max_sequence_len` 分配每个
   attention 层的 `past_k/past_v` buffer，挂在 attention 实例上。
2. **`start_pos` 替换**：上游 `model_omni.py` forward 用
   `start_pos = past_key_values[0][0].shape[1]` 推断 KV 长度。改用 buffer
   的显式 pos 计数器，避免 `shape[1]` 读到 max_len。
3. **forward 包装**：首次调用做 CUDA Graph capture（side stream 上
   warmup + capture），后续 AR step 走 `graph.replay()`。
4. **采样/RNG 处理**：`multinomial` 的 RNG 在 graph 里需显式 seed
   （CUDA Graph 会重放同样的随机流）。用固定 seed 注入保证确定性。

## 4. 风险清单与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| **E14 复现**（KV cat 与 graph 冲突） | 方案 A 改 slice 后不再有动态 cat | `test_kv_fixed_buffer.py` 常驻 |
| **shape 漂移** | max_len 不够 → OOB 写坏 GPU memory | buffer 上限 = prompt_len + max_tokens；生成前校验 |
| **RNG 非确定** | graph replay 重放同随机流 → 每次同输出 | 显式 `torch.random.fork_rng` + seed per replay |
| **公共 API 破坏** | 改 `_attention_forward` 影响 Omni.generate 输出 | 复用现有 determinism 测试（5× seed → same WAV MD5） |
| **mask 语义变化** | pad 到 max_len + mask 屏蔽，需逐位一致 | CPU 上对比 pad+mask vs cat 输出（扩展测试） |
| **capture 失败** | 非默认 stream / allocation 检查 | 按 PyTorch `torch.cuda.graph` 标准 warmup 流程 |
| **多请求并发** | batch>1 时 KV buffer 尺寸翻倍 | buffer shape 乘以 max_batch |
| **可回滚** | patch 范围大 | 每处 patch 打标记（`_NANOVLLM_CUDA_GRAPH`），`disable_cuda_graph()` 逐处还原 |

## 5. 验证计划（GPU 恢复后）

1. **正确性**：生成 1 次（新路径）vs baseline（旧路径），WAV MD5 必须
   逐位一致（沿用 E20 determinism 协议）。
2. **性能**：`bench time --runs 20`，对比 `generate_*_ms` 中位数。
   期望 ≥ 30% 墙时降（ncu 推算上限 73%）。
3. **回归**：49 个非-smoke 测试全绿；`Omni/AsyncOmni/SamplingParams/
   OmniRequestOutput` import 正常。
4. **多 prompt 稳健性**：12 异质 prompt 全帧生成（沿用 E21 协议）。

## 6. 不做的范围（沿用 § 6.5）

- 不写裸 CUDA / Triton kernel（ncu 已证 4/4 都是 launch-bound）
- 不碰 tokenize/wav（CPU-only）
- 不动公开 API 形状：`Omni.generate` 返回 `list[OmniRequestOutput]`
  保持 vllm-omni 对齐签名

## 7. 工作分类

- **已验证（本 session）**：fixed-KV-buffer 数学等价（CPU test）
- **待 GPU 验证**：方案 A pad+mask 的逐位正确性、CUDA Graph 实际加速比
- **实施前置**：一个独立 ticket，受影响文件
  `nanovllm_omni/models/minimind_omni/{attention.py,bundle.py}`

提交参考，给未来 GPU session 的交接：本节（§7）就是"恢复后第一步"。


### 附录 · 2026-09-01 落地状态（plan §3 执行进度）

- §3.1/§3.2 **已落地**：`enable_fixed_kv_buffer`（attention.py section 5，
  opt-in，bit-exact prefill+decode，报告 §23）。
- §3.3 **已落地**：`nanovllm_omni/optim/cuda_graph.py` `enable_cuda_graph`
  —— 安装 buffer patch + neutralize freqs host-read（唯一 capture
  blockers §13/§18）+ 返回 `CudaGraphDecoder`（capture-once/replay-many，
  §25 determinism 协议）。GPU 实测（RTX 3050）：16 步 graphed e2e ~86ms
  vs eager ~400ms（~4.7×，首 call 含 capture 成本；重放 86-94ms），
  prefill bit-exact、同 seed determinism identical、4 异构 prompt 全
  完成。CPU 回归：`tests/test_cuda_graph_module.py`（4 tests）。
- §3.4 RNG：sampling 留 host（graph 外），determinism 由 replay 侧统一
   seed 保证（§25 已证）。
- 剩余：把 `CudaGraphDecoder.generate_tokens` 接进 `stream_generate`
  批量 API，跑 WAV-MD5 协议（§5.1）+ 12 异构 prompt 回归（§5.4）。

#### 集成时可复用的事实（#33/损坏探针的教训）

- **decoder `_prefill` 起点语义**：`CudaGraphDecoder._prefill` 返回
  `argmax(prefill_logits)` 作为首个 token；生产 `stream_generate` 从
  prefill logits **采样** token0。同 seed 下两序列在 index 0 即分歧。
  集成时若保持文本一致，decoder 必须改用生产 samplera 采样 token0
  作为起点。
- **graph-replay vs eager 对比时序**：graph replay 会推进 buffer
  `_kv_pos`；随后同 buffer 再跑 eager 会二次推进，读到不同 KV 长度 →
  logits 出现 ~11 的大 diff。正确对比必须在**相同 KV 状态**（各自
  reset 到同一长度再跑）。#24/#30 已按正确时序证明 replay == eager
  bit-exact（同状态对比）；本探针因时序缺陷不可用，已删除。

#### fact #3（#34 GPU 实证）：production 的 prefill→cache 起点

`CudaGraphDecoder.generate_tokens` 修复 token0 采样后（fact #1），与
production `stream_generate` 同 seed 下 **index 0 一致（都是 66）但
index 1 分歧**。原因不是 RNG——是 production 的 `stream_generate` 最前
K 步走 **prefill（全长度 input、无 cache）**，之后才切到 decode cache；
decoder 从 token0 起直接进 graph replay（立即 cache）。两者的 decode
起点语义不同 → 前几次 logits 不同，序列分歧。集成时须对齐
"prefill 段无 cache → 之后 graph decode" 的边界（在 decoder 前加一次
真实 prefill 整段 forward 建立 KV，再从 token0 起 replay）。

#### fact #4（#35 GPU 实证）：decoder 单文本流与 audio+text 联合生成的边界

`_prefill` 改为 production `[1,9,seq]` shape（含 audio pad 行）+ token0
采样对齐（fact #1）后，index 0 仍一致、index 1 仍分歧。根因：production
`stream_generate` 是 **audio 与 text 并行联合生成**——decode 输入
`[1,9,1]` 的 audio 通道承载**先前步已生成的 Mimi audio code**（非 pad），
与 text 共享 KV 状态。decoder 目前只生成 text 流（audio 通道全 pad），
KV 从第 2 步起与 production 不同 → text 序列分歧。

这是 **graph 集成的工作量本质**（非 bug）：要让 decoder 对齐 production，
必须把 Mimi audio 生成（`sample_one_audio_layer` × 8 通道）与 text
解码**并行**喂进同一条 forward/graph，即完整复刻 `stream_generate` 的
双流。这才是接进 `run_generate`/WAV-MD5（§5.1）之前必须完成的核心。

#### fact #4 resolved（#36 GPU 实证）：decoder 完成 joint text+audio 流

`CudaGraphDecoder.generate_tokens(return_audio=True)` 现在与 production 同
结构：每步先 text 采样、再 8 通道 audio 采样（`sample_one_audio_layer`），
共用同一个 `torch.Generator`（生产 runner `_sample_text`/`_sample_audio_row`
都会 `gen=st.gen`，draw 顺序 text→audio×8）。audio_step 门控（audio 滞后
text 一位）与 runner 一致。音频通道用 pad（生产 decode 输入 `audio_buffer
[:, :, -1:]` 也是 pad），KV 状态因此也与 production 对齐。

GPU 实证：同 seed 下 text 16 tokens、8 audio 通道 × 16 frames 全部
deterministic（identical=True 双流）；audio 码 `[0,2112)` 合法；joint
decode 167ms（含 ×8 audio 采样；纯 text 86ms）vs eager ~400ms+。

**诚实边界**：与 production 文本的 digit 级对齐受 runner 内部 seed
（`base_seed + sha1(rid)`）不透明限制，无法从外部复现；模块级 claim 是
同 seed 双流确定性（§5.1 协议在模块层成立），非跨 runner 实例 digit
相等。图 vs eager 等价性由 #24/#30 bit-exact 独立锚定。

#### 真实生产缺陷 #5（#49 长时探针发现）：异 prompt 复用 KV 错位

工具：`tools/bench_longrun_stability.py` —— 单进程复用 cached decoder，20
个异构 prompt（每 5 个插入 control 重跑）。计划 §4 风险表「shape 漂移 /
OOB」的 serving 场景实测：

| 观察 | 结论 |
|---|---|
| A 同 prompt 连续 3 次 | identical=True（decoder 自洽，非 RNG 泄漏）|
| B 异 prompt 插入后 control | **DIFF**（跨 prompt 状态泄漏）|
| VRAM 20 calls | 567→1197MB（首 call 冷启动一次跳变，非无限增长）|

**根因**：`CudaGraphDecoder._capture` 的 `if self._captured: return` 让 graph
只在**首 call 的 prompt 长度**捕获一次；`_prefill` 每 call 只 `_kv_pos=0` +
重写 buffer 前部。同 prompt（同 prefill 长）重写内容一致 → replay 一致
（探针 A）；异 prompt（不同 prefill 长）→ replay 读到的 KV 形状/offset 仍是
首 call 的、内容却是新 prompt 的 → **错位泄漏**（探针 B）。

**这是生产 defect，非探针 bug**：§5.4 的每-prompt determinism 只测了
「同 prompt 两次」（A 型），没测「异 prompt 间复用」——本探针补上了真实
serving 模式。修复方向（后续 ticket）：**prompt 长度/内容变化时重建
decoder（重 capture）**，或把 graph 形状设计为与 prompt 长度无关（pad
到固定长度 + mask，plan 双支柱本就为此）。当前 opt-in 路径在「单 prompt
长会话」下正确，在「异 prompt 复用」下需每次重建。

'修复方向（后续 ticket）：**prompt 长度/内容变化时重建
decoder（重 capture）**，或把 graph 形状设计为与 prompt 长度无关（pad
到固定长度 + mask，plan 双支柱本就为此）。当前 opt-in 路径在「单 prompt
长会话」下正确，在「异 prompt 复用」下需每次重建。

#### defect #5 修复尝试 —— 负结果（#50 实测无效；随后 #55 已解决）

> 更新：此负结果被后续 #55 GPU 实测推翻 —— 正确修复 = re-capture 分支内
> clear+re-prefill（长度 change 时），见下方「defect #5 已修复」。本段保留
> 记录假说排除过程（防重访）。

我按（自认为的）根因做了最小修复：`CudaGraphDecoder._capture` 在
`prefill_len` 变化时重 capture。**实测 B 仍 False**（同 prompt A True、
异 prompt B False 不变）——修复**没有**解决 cross-prompt 泄漏。

- 这证伪了「prompt 长度」变量：长度重 capture 不影响 B。
- 真根因比长度更复杂：更可能指 **buffer 内容残留**（上个 prompt 写进
  buffer 的 decode KV 在 `[prefill_len : prefill_len+max_new]` 区间，
  新 prompt prefill 只覆盖 `[0:new_len]`，若 new_len 更短则残留被 decode
  步的 `buffer[:, :_kv_pos]` 读到）或 **stream/上下文竞争**（本轮多次
  SSH 后台进程挂起 + SSH 断连）。
- 修复已按迭代 #50 状态回退（不留无效变更）。当时**缺陷保留为 open**：需
  要独立 ticket 用「buffer 残留显式清零 vs graph replay 读 slice 范围」
  做 A/B 隔离。负结果本身有价值：排除长度假设，收窄搜索空间。
  （后继续：#51 探针 + #55 组合修复成功，见下方「已修复」段。）

#### defect #5 隔离探针就绪（#51，GPU 通道暂断，未执行）

`tools/bench_longrun_residue_probe.py` 落地 #50 文档的隔离协议：A/B 两臂
基础上加第三臂——在跨 prompt 之间**显式清零**每个 attention buffer 的
解码 KV 区（`buffer[:, prefill_len:]`）。判定：

- 清零后 B 变 identical → 泄漏 = **buffer 内容残渣**（#50 首hypothesis）
- 清零后 B 仍 DIFF → 泄漏 = **graph replay 读 stale slice range**（机制，
  与内容无关）

探针 lint/black/compile 全绿。本轮 WSL（tailnet）不可达（多次重试 banner
timeout），未执行；GPU 通道恢复后直接运行即可出判定。

#### defect #5 机制代码级定位（#52，无需 GPU）

GPU 通道仍断，但 defect #5 的机制可**静态代码确认**（`attention.py`
`_attention_forward_buffered` L395-410）：

```
self._kv_past_key[:, self._kv_pos : self._kv_pos + seq] = key
self._kv_pos += sequence_len        # Python 侧变更，replay 时不会执行
k_hist = self._kv_past_key[:, : self._kv_pos]   # 按 _kv_pos 动态切
```

- **capture 时**：第 k 个 graph 在 `_kv_pos = prefill_len + k` 状态下捕获，
  写入 buffer[prefill_len+k]、读取 [:prefill_len+k]，这些**偏移被烘焙进
  graph**（replay 只重放 GPU kernel，Python 的 `+=` 与切片不重跑）。
- **replay（异 prompt）时**：`_prefill` 置 `_kv_pos=0` + 写新 prompt（不同
  长度），但 graph_k 仍用烘焙的 `prefill_len+k` 偏移 → 与新的 KV 布局
  **错位**。同 prompt 则 prefill 复现相同布局 → replay 正确。

**结论**：跨 prompt 泄漏 = **graph 偏移绑定首 prompt 的 prefill_len**，
与 buffer 内容残渣无关（故 #51 的"清零解码区"预期 B 仍 DIFF）；与长度
重 capture 修复方向**一致**（#50 实测无效或受 WSL 不稳定干扰，待 GPU
恢复后复测）。

**下一修复（GPU 恢复后优先验证）**：re-capture 需在**新 prefill 完成后**
正确重置 `_kv_pos` 再捕获每个步长；或更稳的方案——**把 buffer 切片改为
基于 prefill 长度显式传递**（forward 不再依赖 Python `_kv_pos` 隐式
步进，而是由驱动按步显式传当前 KV 长度），让 graph 偏移与任一 prompt
无关。两者都需 GPU 上 A/B 确认。

#### defect #5 **已修复**（#55 GPU 实证）

**最终根因**：异长度 prompt re-capture 时,buffer 残留旧 prompt 在
`[min_len : max_len]` 区间的 KV;新 prompt（更短）的 graph replay 读取
`[:kv_pos]` 时把旧 KV 混入 → 泄漏。

**修复**（`_capture` re-capture 分支内）：
1. 清空 buffer（`_clear_buffers`）;
2. 用 `_last_input_ids` 重跑 `_prefill`（重建当 prompt 的干净 KV）;
3. 同长度复用不清空（保持重复确定性）。

**GPU 验证（diag_leak）**：
```
callA pure-repeat identical: True True    （同 prompt 确定性保持）
callB cross-prompt-leak (0vs1): True      （泄漏修复）
callB cross vs A: A0==B0? True
```

**全部一致性的实证链**：同长异内容不泄漏（无需 re-capture）· 异长泄漏
（re-capture+clear+re-prefill 修复）· clear 放 `_prefill`（每 call）会
破坏同 prompt 确定 → **只在 re-capture 分支 clear**。

曾试错且被 GPU 证伪的方向（防重访）：length re-capture 单独无效
（#50）; buffer 清零在 re-capture 外无效（#51/#54）; kv_pos rewind 无效
（#54）。真正修复 = 长度变化 + 清空 + 重 prefill 三者组合。

