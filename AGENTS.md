# AGENTS.md

你AGENT是这个项目的主人

## 定位

`nanovllm-omni` 是 vLLM-Omni 的教学版复刻。教学对象是**整条 omni 推理链路**——
从 stage 队列、调度、paged KV 到 batch 形状与吞吐。**serving 不是附属功能，是教材本身**
（ADR 0016）。

目标机器有两条轨：一张 4 GB 的 RTX 3050（CUDA）与一台 Mac（MLX），两边都要能跑通、
都要出性能结论（ADR 0014）。

取舍的判据只有两条：**代码是否更容易读懂**，以及**每条性能结论是否有实测支撑**。
不追求没有实测支撑的吞吐；有 profile 支撑的优化照做。

这些准则偏向谨慎而不是速度；琐碎任务自行判断。默认的 code agent 指引就是本文件：不要新增 `CLAUDE.md`，否则 Claude Code 会读它而不是本文件，规则就分叉了。

决策记录在 `docs/adr/`（ADR 0001 起），术语在 `docs/glossary.md`。动手前先查有没有相关的 ADR——里面的结论已经定过的事，不要重新讨论一遍。

## 先想再写

**不假设，不藏困惑，把取舍摊开。**

动手之前：

- 把假设明确写出来。不确定就问。
- 有多个解读时全部列出，不要默默挑一个。
- 有更简单的做法就说出来，该反对时反对。
- 有不清楚的地方就停下，指出哪里不清楚，然后问。

## 代码风格（最重要的一节）

- 先写最直白的版本。能用一个 `for` 说清楚的，不要用生成器 + 闭包。
- 不处理假想的输入。没有测试或调用方触发的 corner case，不加分支；真出现再加。
- 不做未测量的性能优化。没有 profile 数据支撑的 cache / 批处理 / 并发，一律不写；有支撑的照做，但要把优化前后的对比写进同一份结果文档（见「文档同步」）。
- 抽象只在出现**第二个真实使用者**时引入。一个实现不要接口、工厂、注册表或配置项。
- 一个函数一件事，名字说清楚就少写注释；需要解释「为什么」时写注释，「做什么」交给代码。
- 宁可多几行显式代码，不要「聪明」写法。凌晨三点能一眼看懂的才算合格。
- 可读性和边界/性能冲突时，选可读的那个，加一行注释说明放弃了什么。
- 删代码优于加代码。改完发现能少一个文件、少一层，就少。
- 只做被要求的事。没被要求的功能、没被要求的开关，都不加。
- 写完自问：资深工程师会不会说这写复杂了？200 行能压到 50 行就重写。

## 改动的边界

**只碰必须碰的地方。收拾也只收拾自己的。**

改已有代码时：

- 不「顺手改善」相邻代码、注释或格式。
- 注释只写必要的、一行能说清的东西。只有运行时才知道的事实——张量形状、dtype、取值范围、单位——值得写下来：放进一行 docstring 写成 `[B, C, H, W]`，或写成它所描述的字段 / 语句的行尾注释。签名已经说清的事不复述。
- docstring（模块 / 类 / 函数）一行写完，不超过 `line-length`。不写 `Args:` / `Returns:` 块，不写设计叙述，不写示例。已有代码里的多行 docstring 不动。
- 真正的约束——上游 stock 实现的坑、版本一升就坏的地方、数值或 bit-identity 要求、实现者必须遵守的契约——写进最近的 `README.md`（`## Gotchas` 加一条），docstring 一行指过去。不要把这段话挪进 `#` 注释块，那只是换了语法的腐烂。这一条只管「多句论述」；形状 / dtype / 单位 / 范围的行尾标注留在原地。
- 不重构没坏的东西。
- 跟着现有风格走，哪怕你会写成另一个样。
- 看到无关的死代码就提一句，不要删。
- 你的改动造成的孤儿 import / 变量 / 函数，自己删干净；改动之前就有的死代码，不问不删。

检验标准：每一行改动都能直接追溯到用户的请求。

## 目标驱动执行

**先定成功标准，再循环到验证通过。**

把任务改写成可验证的目标：

- 「加校验」→「先为非法输入写测试，再让它们通过」
- 「修 bug」→「先写一个能复现的测试，再让它通过」
- 「重构 X」→「重构前后测试都要绿」

多步任务先给一份简短计划：

1. [步骤] → 验证：[检查]
2. [步骤] → 验证：[检查]

强的成功标准能让你自己循环；弱标准（「让它跑起来」）只能靠反复找人确认。

## 文档同步

以下变化发生时，必须在同一变更中更新对应文档，并在验证记录中注明结果：

- 修改模型 pipeline、stage contract、deploy 默认值或公开推理行为时，更新对应 `docs/dev/` 迁移文档或 `docs/adr/` 决策记录。
- 修改 benchmark 参数、输入、计时方式、profiler marker 或 profile 输出格式时，更新 profile 协议和结果文档。
- 新增或重跑性能测量时，更新对应模型的 profile 结果：硬件、代码版本、输入、采样配置、runs、warmup、wall、显存、阶段分解和限制都要写清楚。
- 实施性能优化时，先记录 profile 证实的瓶颈，再在同一结果文档中记录优化前后对比；没有对比数据就不把改动称为性能优化。
- 修改公共契约、pipeline kind、deploy 配置或 parity 结论时，同步 README、ADR 或迁移文档，避免代码和教学材料产生不同说法。
- `docs/perf/` 中的大型 trace 可以保留为本地产物；可复现的结论必须进入受版本控制的 `docs/dev/` 或 `docs/adr/` 文档，并注明原始产物位置。

完成标准：每个受影响模型、协议和决策文档都已更新，且文档中的性能数字能追溯到具体硬件、代码版本和原始 profile 产物。

## 对齐契约

对外接口与 `vllm-omni` 对齐（消费者可见的名字、签名、字段、返回形状、HTTP 响应）：
`Omni` / `AsyncOmni` / `SamplingParams` / `OmniEngineArgs` / `OmniRequestOutput` /
`PipelineConfig` / `DeployConfig` / `POST /v1/chat/completions`。

对齐指行为兼容，不指内部实现相同。因本项目范围（单进程、小模型、stdlib serving）
产生的差异是允许的，用测试锁住并在文档里写明，不要声称不存在的 parity。

参考仓库 `/Users/mcig/Projects/vllm-omni` **只读**：不修改、不 import。

## 结构约定

- 模型族拓扑放 `nanovllm_omni/models/<family>/pipeline.py`；采样/资源默认值放 `deploy/<family>.yaml`。两者不混。
- `__init__.py` 只 re-export，不放实现；`config/__init__.py` 保持叶子，不 import engine/model/entrypoint。
- 新增模型族按 `.agents/skills/add-new-model/SKILL.md` 走。

## 测试代码

编写或修改测试时，先读 `docs/test-factories.md`，其中包含测试规范和 stage 工厂用法。

## 命名约定

同一语义只保留一个拼写，全词优先：

- 数量：`num_*`（`num_heads` / `num_layers` / `num_requests`）。不用 `n_*`，唯一例外 `SamplingParams.n`（vLLM 锁名）。
- 序列：`sequence` / `sequence_len` / `max_sequence_len`。不用 `seq`。
- 全词：`token` / `config` / `max_embeddings`。不用 `tok` / `cfg` / `max_emb`。

不改的边界：`trust_remote_code` 加载的远端模型属性（`n_rep` 等）、注释/docstring 里的形状记法、
JSON key / 表头等序列化 schema、上面列出的公开对齐符号。

## 参考指南与领域文档

**把指引用到当前代码上之前，先核对它是否还成立。**

- agent / bot 的 review 意见当建议，不当事实。改之前先确认它在当前代码上仍然适用。
- 改专门区域之前，先读对应的本地指引或 skill。
- 指引和要做的改动冲突时，拒绝冲突的那部分并说明原因。

本仓库的本地 skill：

| 区域 | Skill | 什么时候读 |
|---|---|---|
| 新增模型族 | `.agents/skills/add-new-model/SKILL.md` | 加 / 改模型族、pipeline、stage contract、deploy yaml、`Omni(...)` 生成链路 |
| 提交前检查 | `.agents/skills/ci-safe-delivery/SKILL.md` | 每次 commit / push 之前 |
| 本地 ↔ WSL 闭环 | `.agents/skills/nanovllm-omni-wsl-loop/SKILL.md` | 本地改完要传到 WSL 跑真权重 |
| Colab 远端实验 | `.agents/skills/colab-operator/SKILL.md` | 在 Colab 上开 GPU/TPU 会话跑实验 |

## 贡献卫生

提 PR 或提交 AI 辅助改动前，先读 `CONTRIBUTING.md` 的贡献卫生要求。

## 提交前闸门

- 新增函数或规则前，先搜索项目中是否已有实现。
- 文件大小钩子拒绝提交时，按报错拆分文件后再提交。
- 流程、CI 命令、hook 安装见 `CONTRIBUTING.md`；提交前跑 `scripts/pre-commit`，push 前跑 `scripts/pre-push`。
- 每个公开契约的改动配一个聚焦测试；一个 commit 一件事。
- 没有审过的 diff 不 push。push 前完整读一遍自己的 diff，`scripts/pre-push` 全绿再推。

**这些准则起作用的表现：** diff 里没必要的改动变少，因过度设计而返工变少，澄清发生在动手之前而不是出错之后。
