# AGENTS.md

## Project contract

`nanovllm-omni` aligns its user-facing audio API with `vllm-omni`; it does not reproduce the full vLLM-Omni implementation. The reference repository is `/Users/mcig/Projects/vllm-omni` and is read-only. Never modify it or import from it.

## Alignment rules

- Treat `.scratch/aligned-interfaces/SPEC.md` as the source of truth for locked decisions. Do not change a locked decision silently; ask the user first.
- Match public names, signatures, field names, return shapes, and observable behavior where the scope requires it:
  - `Omni`, `AsyncOmni`, `OmniBase`
  - `SamplingParams`, `OmniEngineArgs`
  - `OmniRequestOutput`
  - `PipelineConfig`, `DeployConfig`, registry functions
  - `POST /v1/chat/completions` OpenAI response shape
- Scope is any model family that runs locally on the user's hardware (RTX 3050 4 GB). Add a family through the registry contract; the full runbook lives in `.agents/skills/add-new-model/SKILL.md`. The must-not list and stack constraints (stdlib-only contracts/serving, no new dependencies, no FastAPI/uvicorn/Pydantic/WebSockets/distributed execution/full-duplex S2S, closed-set `StageExecutionType` unless a ticket) live in that skill, not here.
- Prefer package modules with one responsibility. `__init__.py` files should primarily re-export public symbols; implementation classes and runtime logic belong in dedicated modules. The configuration layer lives in `nanovllm_omni/config/` (`registry.py` + `params.py`) behind a thin `__init__.py` facade; keep `config/__init__.py` leaf-only (no engine/model/entrypoint imports) so the import graph stays acyclic.
- Do not copy vllm-omni code. Inspect it only to verify names, signatures, field shapes, and architecture patterns, then adapt to this repository's smaller scope.
- Keep deploy/runtime knobs separate from pipeline topology. Pipeline topology belongs in model-family `pipeline.py`; sampling/resource defaults belong in `deploy/*.yaml`.
- Preserve the aligned API contract when cleaning up legacy code. Any intentional compatibility break must be explicit in the ticket and reflected in tests/docs.

## Required workflow for interface changes

1. Read the relevant SPEC and ticket before editing.
2. Inspect the current nanovllm implementation and the corresponding read-only vllm-omni reference.
3. Write or update the mapping in `docs/aligned_interfaces.md` when a public symbol or behavior changes.
4. Add a focused test for every changed public contract: construction, signature, output fields, error behavior, or HTTP JSON shape.
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
   The hook runs the lint, format, and public-API-import checks on
   the whole repo. Slow checks (pytest, torch-using unit tests) live
   in the GitHub Actions workflow -- `.github/workflows/ci.yml` --
   not in the hook.
6. For Python seam changes, verify `Omni(...).generate(...)` returns `OmniRequestOutput` with valid audio/WAV bytes. For HTTP changes, verify `/v1/chat/completions` has the required OpenAI envelope and decodable base64 WAV audio.
7. Commit each ticket or coherent change separately with a descriptive message, then push only after the checks pass.
6. For Python seam changes, verify `Omni(...).generate(...)` returns `OmniRequestOutput` with valid audio/WAV bytes. For HTTP changes, verify `/v1/chat/completions` has the required OpenAI envelope and decodable base64 WAV audio.
7. Commit each ticket or coherent change separately with a descriptive message, then push only after the checks pass.

## Definition of aligned

Alignment means consumer-visible compatibility, not identical internals. A difference is acceptable only when it is required by this project's explicit scope (for example, local single-process execution on small models and stdlib serving). Document such differences in `docs/aligned_interfaces.md` rather than claiming unsupported parity.
