# AGENTS.md

## 定位

`nanovllm-omni` 是一个**教学项目**：目标是让人读懂 omni（文本→音频）推理链路是怎么跑起来的，
并在一张 4 GB 的 RTX 3050 上真的跑通。它不是生产推理引擎，不追求吞吐、不追求覆盖所有模型。

一切取舍的判据只有一条：**代码是否更容易读懂**。

## 代码风格（最重要的一节）

- 先写最直白的版本。能用一个 `for` 说清楚的，不要用生成器 + 闭包。
- 不处理假想的输入。没有测试或调用方触发的 corner case，不加分支；真出现再加。
- 不做未测量的性能优化。没有 profile 数据支撑的 cache / 批处理 / 并发，一律不写。
- 抽象只在出现**第二个真实使用者**时引入。一个实现不要接口、工厂、注册表或配置项。
- 一个函数一件事，名字说清楚就少写注释；需要解释「为什么」时写注释，「做什么」交给代码。
- 宁可多几行显式代码，不要「聪明」写法。凌晨三点能一眼看懂的才算合格。
- 可读性和边界/性能冲突时，选可读的那个，加一行注释说明放弃了什么。
- 删代码优于加代码。改完发现能少一个文件、少一层，就少。

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

`StageConfig.stage_factory` 用 `(module, attr)` 元组引用，让 `PipelineConfig` 完全可序列化（`test_bundle_resolve_snapshot.py` 依赖这点）。所有工厂写在 `tests/_stage_factories.py`，按用途分四类：

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

## 改代码前

流程、CI 命令、hook 安装见 `CONTRIBUTING.md`；提交前跑 `scripts/pre-commit`，push 前 `scripts/pre-push`。
每个公开契约的改动配一个聚焦测试；一个 commit 一件事。
