# ADR 0018: offload 三步走——mmap → 量化 → KV SSD

- Status: Accepted
- Date: 2026-11-09
- 关联: ADR 0014（双轨目标）、0015（MLX 要 varlen）、0016（serving 是教学对象）

## 问题

"offload" 在这个仓库有三个不同的东西，成本、收益、代价都不同：

1. **权重 offload**——模型装不下时把权重放到别处
2. **KV cache offload**——并发一高 KV 就撑爆内存
3. **量化**——把权重压小，是 Mac 上唯一真正省内存的手段

Apple Silicon 没有独立显存（ADR 0014 附注：MPS 与 MLX 看到的是同一个 11.84 GiB 工作集），
所以「layer-wise offload 到 CPU」在 Mac 上不省任何内存——它只是给已经 dispatch-bound
的路径再插一次 host 往返。**Mac 上真正有意义的是磁盘这一层。**

## 决定

**三个都做，顺序是：权重 mmap → 量化 → KV cache SSD。**

### 第一步：权重 mmap + 按需分页

`safetensors` 本来就是 mmap 的。真正要做的是**不要把整个权重 copy 进 RAM**——
让文件页留在 page cache 里，由 OS 在内存压力下换出到磁盘。

- 收益：模型可以比 11.84 GiB 工作集更大（受磁盘与首触延迟限制，不受内存限制）
- 代价：首次触碰某页是一次磁盘读（NVMe 顺序 3–7 GB/s，随机 ~100 µs）→ 表现为 TTFT 变差，吞吐不变
- 归属：**serving 层**，不碰 `engine/`

### 第二步：量化

- 收益：4× 权重压缩（int8/int4），直接换来"能装下更大的模型"
- 代价：**必然破坏逐位 parity**。只能降到 ADR 0011 的 Compatible 层——
  任务效用与安全不劣化，但 token 不再逐位一致
- 依赖：Mac 上 bitsandbytes 不可用；候选是 `torchao` / MLX 的量化路径 / GGUF
- 注意：这会**改变数字**，所以量化前后的性能结论不能混在一张表里

### 第三步：KV cache offload 到 SSD

- 收益：serving 场景真正值钱的那一个——并发一高 KV 就撑爆
- 代价：每次换页是一次随机读 → **伤 TTFT，不伤吞吐**
- 现状：**上游 `vllm-metal` 也还没做完**（它的 What's next 里写着
  "mature SSD-backed KV cache offloading"）。所以这一步要么自己写，要么等上游
- 归属：**serving 层**，与 `engine/` 的 paged KV 是两层

## 后果

- **三步都不在 `engine/` 层**。前两步在模型加载与权重路径上，第三步在 `serving/` 的
  请求生命周期上。符合 AGENTS.md 的范围纪律。
- **每步单独一个 ADR**，各自带复现配置与实测数字。量化那步尤其要单独记，
  因为它改变了所有后续测量的口径。
- **性能结论必须区分"是否量化"**。ADR 0014 已定"不跨机相减"，
  这里再加一条：不跨量化档位相减。
- **KV SSD 与 varlen 是两件独立的事**。ADR 0015 的 varlen 解决"批内长度不齐"，
  KV SSD 解决"总容量不够"。两者都要，但不要互相引用对方的数字。
- 优先做 mmap 的理由是它最便宜、且不改变任何数字；量化放第二是因为它会改变
  数字，先把不改变数字的事情做完，测量基线才干净。

## 边界

- 本 ADR **不**决定具体用哪个量化库（torchao / MLX / GGUF），那要看第一步做完之后
  内存到底差多少。
- 本 ADR **不**涉及 layer-wise offload 到 host RAM——在统一内存上它没有意义。
- 本 ADR **不**承诺 SSD offload 能进 12 月的交付范围；它是三步里最重的一步。
