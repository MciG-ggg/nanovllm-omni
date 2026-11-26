# Contributing to nanovllm-omni

Thank you for your interest. This project is intentionally small: it
exists to teach and to exercise vllm-omni's consumer-visible audio API
on a single 4 GB card. Contributions that move it further from that
purpose will be declined with a pointer to the rationale; everything
that aligns with the contract below is welcome.

The full project contract (alignment rules, naming conventions,
definition of "aligned") lives in [AGENTS.md](AGENTS.md). Read it
before opening a PR — most review feedback traces back to one of the
rules there.

## Quickstart

```bash
git clone https://github.com/MciG-ggg/nanovllm-omni
cd nanovllm-omni
pip install -e ".[dev]"            # lint + test, no torch
pip install -e ".[dev,cuda]"       # Linux/WSL: torch + triton + flash-attn
#   pip install -e ".[dev,mlx]"      # Apple Silicon: mlx + mlx-lm

# Run the full pre-push checks:
scripts/pre-push
# Or run the equivalent checks individually:
ruff check nanovllm_omni/ tests/ tools/
black --check nanovllm_omni/ tests/ tools/
python -m pytest -m "not smoke" -v
python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"
python -m compileall -q nanovllm_omni
```

`scripts/install-hooks.sh` wires `scripts/pre-push` as the git
**pre-push** hook so the full CI-equivalent pass runs automatically
before every push.

## Workflow for any code change

1. **Open or pick a ticket** describing the change. If you're fixing
   something undocumented, open the ticket first.
2. **Inspect both sides**: the current `nanovllm_omni/` implementation
   *and* the corresponding read-only `vllm-omni` reference
   (`/Users/mcig/Projects/vllm-omni`). vllm-omni is read-only; never
   modify it or import from it.
3. **Add a focused test** that locks the changed public contract —
   construction, signature, output fields, error behavior, or HTTP
   JSON shape. The existing `tests/test_registry_resolver.py` is a
   good drift-lock example.
4. **Run the CI-equivalent set** (the `scripts/pre-push` block above)
   until it's green locally. Note: the local env (macOS, Py 3.13,
   torch 2.11) differs from CI (Py 3.11/3.12, fresh-pip torch), so a
   locally-green tree is not proof of a green CI. Keep tests
   version-agnostic: don't bake statistical or numeric assertions
   tight to a specific Python/torch release.
5. **Verify the seam** your change touches end-to-end:
   - Python seam: `Omni(...).generate(...)` returns
     `OmniRequestOutput` with valid audio/WAV bytes.
   - HTTP seam: `POST /v1/chat/completions` returns the required
     OpenAI envelope with decodable base64 WAV audio.
6. **Commit each ticket or coherent change separately** with a
   descriptive message. Push only after all checks pass.

## Adding a new model family

The full runbook is at
[`.agents/skills/add-new-model/SKILL.md`](.agents/skills/add-new-model/SKILL.md).
The short version: add a `nanovllm_omni/models/<family>/pipeline.py`
that builds a `PipelineConfig`, call `register_pipeline(...)` at
import time, ship the per-stage sampling defaults in
`nanovllm_omni/deploy/<family>.yaml`, add a deploy unit test, and
wire an `examples/offline_inference/<family>/` smoke.

## Naming conventions

See [AGENTS.md](AGENTS.md#命名约定) for naming rules and exceptions.

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

## Where to ask

- **Bug / feature**: open an issue using the templates under
  [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
- **Security**: see [SECURITY.md](SECURITY.md).
- **Anything else**: open a discussion-style issue.
