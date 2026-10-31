# AGENTS.md

## 定位

`nanovllm-omni` 是一个**教学项目**：目标是让人读懂 omni（文本→音频）推理链路是怎么跑起来的，
并在一张 4 GB 的 RTX 3050 上真的跑通。它不是生产推理引擎，不追求吞吐、不追求覆盖所有模型。

一切取舍的判据只有一条：**代码是否更容易读懂**。

这些准则偏向谨慎而不是速度；琐碎任务自行判断。默认的 code agent 指引就是本文件：不要新增 `CLAUDE.md`，否则 Claude Code 会读它而不是本文件，规则就分叉了。

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
- 不做未测量的性能优化。没有 profile 数据支撑的 cache / 批处理 / 并发，一律不写。
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

测试 = 三段式（**BUILD → OPERATE → CHECK**）——构造、操作、检验。读者扫一眼应该立刻看清这三段在哪、断言在哪。

- **三段用空行分开**。不要写 `# build:` / `# operate:` / `# check:` 注释——空行够用，注释是噪音。
- **BUILD ≤ 3×(OPERATE+CHECK)**。超过就抽 helper 或重写；超过 5× 时即便抽出来，读者也得停下来重找主线，几乎不算一个测试了。
- **stdlib import 必须放文件顶部**。`import json` / `import base64` / `import tempfile` 出现在函数体里就是 BOC 被打断，读者得回到 BUILD 重新定位。第三方重包（torch / diffusers / transformers）按 `pytest.importorskip` 守门后仍可放函数体，但要在文件顶部注释解释为什么。
- **fake / stub 只实现被测路径真用到的 API**。`FakeMimi` 只写 `decode()`，不要顺手补 `encode()`——YAGNI 在 fake 上一样适用。
- **fake 命名 `_Fake*`**，靠近使用它的测试文件，不要全局 `_fakes.py`——一处改动只影响本文件，blast radius 小。
- **抽 helper 至少要有第二个真实使用者**。`_ThinkerAudio` / `_PlainAudio` / `_Arr` / `_FakePolicy` 这种单点 inline stub 不动；第二次出现再合并，并加 `# ponytail: …` 注释。
- **`# ponytail:` 注释标故意没合并的重复**（见下面 DSL 词汇表的 `FakeMimi` / `_FakeMimi`）。

### 本仓库的测试 DSL（tuple-registered factories）

`StageConfig.stage_factory` 用 `(module, attr)` 元组引用，让 `import nanovllm_omni.config` 保持无 torch：`models/*/pipeline.py` 只存字符串，不会 import `thinker.py` / `talker.py`（这两个模块顶部就 `import torch`），所以 CPU-only 机器也能检查拓扑。代价是测试不能传闭包，只能模块级工厂 + 捕获槽。所有工厂写在 `tests/_stage_factories.py`，按用途分四类：

| 词 | 形态 | 用途 |
|---|---|---|
| `thinker_simple` / `talker_simple` / `code2wav_simple` / `executor_simple` | 无状态工厂 | 跑通就够，不观察 |
| `identity_process_input` / `bridge_process_input` | bridge hook | 测 `process_input` 是否被调用 |
| `capturing_simple` + `reset_captures` / `get_captures` | 捕获槽 | 测 `SamplingParams` 是否到 stage |
| `observing_factory` + `reset_factory_observations` / `get_factory_observations` | 捕获槽 | 测 `(deploy, args)` 是否到 factory |
| `logged_*` + `reset_log` / `get_log` | 捕获槽 | 测 stage 调用顺序 |
| `logged_diffusion` + `reset_diffusion_log` / `get_diffusion_log` | 捕获槽 + `_LoggedDiffusionPipeline` | diffusion 单 stage |

四个捕获槽共享一个 `_make_slot()` 闭包工厂（`(items, reset, snapshot)`），不要复制 reset/snapshot 对。

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

**不重复劳动，不开琐碎 PR，不提交没人看过一遍的 agent 输出。**

提 PR 之前：

- 查相关 issue、开着的 PR 和区域关键词，确认没人在做同一件事。
- 已经有 PR 在修同一个问题，就不再开一个。
- 你的做法和别人实质性不同时，先说明不同在哪。

不为低价值琐事开 PR：

- 一次性的 typo、孤立的格式抖动、没有实质工作的机械清理，都不单独开。
- 机械清理只有在直接支撑一个有意义的改动时才一起带上。

AI 辅助的工作：

- 提交的人必须理解，并能为每一行改动辩护。
- 提交的人要看过完整 diff，并在发布前跑过相关测试。
- PR 描述里写明用了 AI 辅助、做过重复工作检查、测试命令和结果。

没准备好就停在门口（fail closed）：

- 改动重复、过于琐碎、缺上下文、或拿不出可信的验证路径时，停下来说明缺什么。
- 不要为了继续推进而发明流程例外。

验证脚手架：

- 为证明某个 PR 正确而写的一次性脚本，跑完把命令和结果贴进 PR 的测试记录，**不提交**。仓库的 `tests/` 是长期契约测试，不要把一次性 harness 塞进去。

## 提交前闸门

- 流程、CI 命令、hook 安装见 `CONTRIBUTING.md`；提交前跑 `scripts/pre-commit`，push 前跑 `scripts/pre-push`。
- 每个公开契约的改动配一个聚焦测试；一个 commit 一件事。
- 没有审过的 diff 不 push。push 前完整读一遍自己的 diff，`scripts/pre-push` 全绿再推。

**这些准则起作用的表现：** diff 里没必要的改动变少，因过度设计而返工变少，澄清发生在动手之前而不是出错之后。
