# AGENTS.md

## Project contract

`nanovllm-omni` aligns its user-facing audio API with `vllm-omni`; it does not reproduce the full vLLM-Omni implementation. The reference repository is `/Users/mcig/Projects/vllm-omni` and is read-only. Never modify it or import from it.

## Alignment rules

- Match public names, signatures, field names, return shapes, and observable behavior where the scope requires it:
  - `Omni`, `AsyncOmni`, `OmniBase`
  - `SamplingParams`, `OmniEngineArgs`
  - `OmniRequestOutput`
  - `PipelineConfig`, `DeployConfig`, registry functions
  - `POST /v1/chat/completions` OpenAI response shape
- Scope is any model family that runs locally on the user's hardware (RTX 3050 4 GB). Add a family through the registry contract; the full runbook lives in `.agents/skills/add-new-model/SKILL.md`.
- Prefer package modules with one responsibility. `__init__.py` files should primarily re-export public symbols; implementation classes and runtime logic belong in dedicated modules. The configuration layer lives in `nanovllm_omni/config/` (`registry.py` + `params.py`) behind a thin `__init__.py` facade; keep `config/__init__.py` leaf-only (no engine/model/entrypoint imports) so the import graph stays acyclic.
- Keep deploy/runtime knobs separate from pipeline topology. Pipeline topology belongs in model-family `pipeline.py`; sampling/resource defaults belong in `deploy/*.yaml`.

## Required workflow for interface changes

1. Read the relevant ticket before editing.
2. Inspect the current nanovllm implementation and the corresponding read-only vllm-omni reference.
3. Add a focused test for every changed public contract: construction, signature, output fields, error behavior, or HTTP JSON shape.
5. Run the CI-equivalent checks before commit:
   ```bash
   ruff check nanovllm_omni/ tests/
   black --check nanovllm_omni/ tests/
   python -m pytest -m "not smoke" -v
   python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"
   python -m compileall -q nanovllm_omni
   ```
   Or, equivalently, run the pre-commit hook (install once with
   `./scripts/install-hooks.sh`):
   ```bash
   scripts/pre-commit
   ```
   The hook runs the fast lint, format, and public-API-import checks on
   the whole repo. `scripts/pre-push` runs the same command set as the
   no-torch CI job (adds compileall + pytest -m "not smoke");
   `install-hooks.sh` wires it as the git **pre-push** hook, so a full
   CI-equivalent pass runs automatically before every push.

   Caveat: the local env (macOS, Py 3.13, torch 2.11) differs from CI
   (Py 3.11/3.12, fresh-pip torch), so a locally-green tree is not proof
   of a green CI. Keep tests version-agnostic: don't bake statistical or
   numeric assertions tight to a specific Python/torch release.
6. For Python seam changes, verify `Omni(...).generate(...)` returns `OmniRequestOutput` with valid audio/WAV bytes. For HTTP changes, verify `/v1/chat/completions` has the required OpenAI envelope and decodable base64 WAV audio.
6. For Python seam changes, verify `Omni(...).generate(...)` returns `OmniRequestOutput` with valid audio/WAV bytes. For HTTP changes, verify `/v1/chat/completions` has the required OpenAI envelope and decodable base64 WAV audio.
7. Commit each ticket or coherent change separately with a descriptive message, then push only after the checks pass.

## Definition of aligned

Alignment means consumer-visible compatibility, not identical internals. A difference is acceptable only when it is required by this project's explicit scope (for example, local single-process execution on small models and stdlib serving); document such differences in tests and docs rather than claiming unsupported parity.

已知在范围内差异:内建 registry 的 `PipelineConfig` 以 `name` 为注册键(参考 vllm-omni 以 `model_type` 为键),且同名重复注册静默覆盖(参考为 validate+warn)。`name` 键是为内部一致性做的有意选择,不承诺参考级告警行为;实际行为由 `tests/test_registry_resolver.py` 的 drift-lock 测试锁定。

## 命名约定

函数/变量/字段名:全词优先、`num_*` 优先。同一语义只保留一个拼写:

- 数量:一律 `num_*`(`num_heads`/`num_layers`/`num_requests`/`num_positions`/`num_steps`);不用 `n_*`(唯一例外是 `SamplingParams.n`,vLLM 锁名)。
- 序列:一律 `sequence`/`sequence_len`/`max_sequence_len`;不用 `seq` 缩写。
- 完整词优先:`token` 不用 `tok`、`config` 不用 `cfg`、`max_embeddings` 不用 `max_emb`。

**边界(不纳入重命名,写新代码也不要改这些名字):**

- 远端模型(`AutoModelForCausalLM.from_pretrained(...trust_remote_code=True)` 加载)的属性名是外部契约,如 `n_rep`/`n_local_heads`/`n_local_kv_heads`;只读引用,别改名。
- 字符串/注释/docstring/张量形状记法(如 `[B, seq, kv, d]`)不是标识符,不动。
- 序列化格式键(JSON key、markdown 表头)是外部 schema,不动。
- 公开对齐符号(`Omni`/`SamplingParams` 字段/`OmniRequestOutput` 字段等)是公开契约,走接口变更流程,不因内部统一而改。

内部命名只要求自洽,不要求与 vllm-omni 内部一致——vllm-omni 内部自己就是 `n_*`/`num_*` 混用,没有可对齐的基准。重命名请按轴分 commit,并跑 pre-commit 钩子。
