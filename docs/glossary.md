# 术语表

按英文关键字排序。每条给出：定义、在本仓库哪儿、以及为什么值得单独记一条。

| 术语 | 定义 | 在本仓库哪儿 |
|---|---|---|
| **backend 接缝** | 框架无关的调度层与框架相关的模型执行层之间的分界线。本仓库把它画在「一步 AR 执行」（`prepare_*` / `forward` / `sample` / `bridge_hidden`），不画在 tensor API 层 | `nanovllm_omni/backend/base.py`，ADR 0013 |
| **block table** | 每请求一张表，记录它的 KV 落在哪些 KV block 里。各请求的 KV 因此不必连续排布 | fork `engine/block_manager.py`；`model_runner.prepare_decode` 构造 |
| **chunked prefill** | 把一个长 prompt 切成多段、每步只算一段，让 decode 请求有机会插进同一步 | fork `engine/scheduler.py::schedule`（prefill 分支的 `max_num_batched_tokens` 夹取） |
| **cu_seqlens** | 扁平 token 序列里各请求的起止偏移。varlen attention kernel 靠它切出每段的 q/k | fork `layers/attention.py`（`flash_attn_varlen_func` 的 `cu_seqlens_q/k`） |
| **dispatch-bound** | 瓶颈在 Python/aten 的派发次数而不在 GPU 算力。表现是单步耗时与序列长度几乎无关 | ADR 0007：talker T=1 12.20 ms vs T≈54 12.4 ms/步 |
| **early exit / 早退门** | MiniMind-O talker 的发射门：8 个码本都到齐才发一帧，某个码本提前 stop 就关门不再重开 | `models/minimind_omni/stage.py::_can_emit_frame` |
| **发射门** | 见 early exit | 同上 |
| **head-of-line blocking / 队头阻塞** | 一个批里最先结束的请求要等最慢的请求跑完。ADR 0012 把它列为当前批处理的边界 | ADR 0012 |
| **paged KV** | KV cache 按固定大小的 block 分配，用 block table 索引。请求可以在不重排 KV 的前提下增长 | fork `engine/block_manager.py`；vllm-metal 也用同一套 |
| **padding / 补齐** | 把批内不同长度的序列补到最长。代价由长短差决定，ragged 批上可能很贵（上游实测 +143%） | ADR 0015 引的 vllm-metal benchmark |
| **parity** | 输出与参照实现的一致性。本仓库分层：Exact 逐位一致，Compatible 只保证任务效用与安全 | ADR 0011 |
| **slot 复用** | 一个请求跑完后立刻把它的 slot 给队列里的下一个请求，不必等整批结束。这是真 continuous batching 与 static batch 的分界 | fork `Scheduler` 已支持（`waiting`/`running` 双 deque + `postprocess` 移除完成序列），nanovllm-omni 侧未接 |
| **slot_mapping** | 一步里每个 token 的 KV 该写到哪个 KV slot 的扁平映射 | fork `model_runner.prepare_prefill/prepare_decode` |
| **static batch / 静态批** | 批在开始时固定，跑完才结束。本仓库当前的批就是静态批 | ADR 0012 |
| **unified memory / 统一内存** | CPU 与 GPU 共享同一块物理 DRAM。Apple Silicon 全系如此，没有独立显存 | ADR 0014 附注；`mx.device_info()["max_recommended_working_set_size"]` |
| **vendor code / vendor layer** | 模型作者自己写的代码，**不在本仓库**，躺在 checkpoint 目录里，由 `transformers` 的 `get_class_from_dynamic_module` 动态加载 | talker：`stage.py:432` 加载 `model_omni.TalkerModule`；thinker 反过来明写 "No vendor code"（`thinker.py:234`） |
| **varlen / variable-length attention** | 一次 attention 调用处理长度不一的若干段，靠 `cu_seqlens` 划界，不补齐 | fork 已用（flash_attn）；MLX 没有；MPS 没有（`aten::_flash_attention_forward` 未实现） |
| **窗口化 KV（windowed KV）** | talker 侧只缓存 `[0, step-8)`，每步只重算最后 8 列。FLOPs 从 O(n²) 降到 O(8n)；更重要的作用是让每步的 query 维度定长，从而可 varlen、可上 graph | ADR 0007 当年因「不省时间」否掉，ADR 0017 重新成立并落到 talker 重写里 |
| **mmap 按需分页** | 权重文件保持 mmap，不整体 copy 进 RAM；文件页留在 page cache，内存压力下由 OS 换出到磁盘。Mac 上权重 offload 的实际形态 | ADR 0018 第一步 |
| **量化档位** | 权重被压到 int8/int4 的程度。**量化前后的数字不能放在同一张表里** | ADR 0018 第二步 |
| **vendor 重写** | 按 vendor 的语义写一份自己的实现，而不是把 vendor 源码拷进来。数值行为要与 vendor 一致（parity），代码结构服从本仓库风格 | ADR 0017 |
| **CUDA Graph** | 把一串 kernel 捕获成一张图，replay 时省掉每步的 host 派发。本仓库最大的单一杠杆 | ADR 0007（thinker 3.8×）、ADR 0008（smolvlm 3.5×） |
| **mx.compile** | MLX 的图捕获等价物。实测小算子链 4.91×，是 MPS 上唯一可用的同类手段 | ADR 0015；MPS 上 `torch.mps` 无 graph API |
| **NAX** | M5 的 tensor 单元。`vllm-metal` 在 M5 上用它加速 MHA/GQA/MQA 的 prefill；M2 Air 上没有 | ADR 0015 边界 |
