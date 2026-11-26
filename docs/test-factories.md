# 测试代码

测试 = 三段式（**BUILD → OPERATE → CHECK**）——构造、操作、检验。读者扫一眼应该立刻看清这三段在哪、断言在哪。

- **三段用空行分开**。不要写 `# build:` / `# operate:` / `# check:` 注释——空行够用，注释是噪音。
- **BUILD ≤ 3×(OPERATE+CHECK)**。超过就抽 helper 或重写；超过 5× 时即便抽出来，读者也得停下来重找主线，几乎不算一个测试了。
- **stdlib import 必须放文件顶部**。`import json` / `import base64` / `import tempfile` 出现在函数体里就是 BOC 被打断，读者得回到 BUILD 重新定位。第三方重包（torch / diffusers / transformers）按 `pytest.importorskip` 守门后仍可放函数体，但要在文件顶部注释解释为什么。
- **fake / stub 只实现被测路径真用到的 API**。`FakeMimi` 只写 `decode()`，不要顺手补 `encode()`——YAGNI 在 fake 上一样适用。
- **fake 命名 `_Fake*`**，靠近使用它的测试文件，不要全局 `_fakes.py`——一处改动只影响本文件，blast radius 小。
- **抽 helper 至少要有第二个真实使用者**。`_ThinkerAudio` / `_PlainAudio` / `_Arr` / `_FakePolicy` 这种单点 inline stub 不动；第二次出现再合并，并加 `# ponytail: …` 注释。
- **`# ponytail:` 注释标故意没合并的重复**。

## 测试 DSL（tuple-registered factories）

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

