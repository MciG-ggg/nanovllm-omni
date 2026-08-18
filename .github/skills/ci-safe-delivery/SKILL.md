---
name: ci-safe-delivery
description: Use before every commit and push to prevent CI failures from lint, formatting, imports, or tests.
---

# CI-safe delivery

Run this checklist before pushing Python changes:

1. Create or activate an isolated virtual environment; install the exact project dev extras (`pip install -e ".[dev]"`).
2. Run the same commands as `.github/workflows/ci.yml`, not a substitute:
   - `ruff check nanovllm_omni/ tests/`
   - `black --check nanovllm_omni/ tests/`
   - `python -m pytest -m "not smoke" -v`
3. If formatting fails, run `black nanovllm_omni/ tests/`, inspect the diff, then rerun both lint and format checks.
4. Verify public imports explicitly:
   `python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"`
5. Check for stale imports after deletions or API migrations:
   `python -m compileall -q nanovllm_omni`
6. Run the focused tests for changed code, then the complete non-smoke suite.
7. Only commit after all checks pass. Push only the commit that contains the verified tree.

## Common failure modes

- `black --check` fails even when pytest passes: formatting is a separate CI gate.
- Ruff F401/I001 errors commonly appear after changing package exports; use explicit `__all__` re-exports and sorted imports.
- Deleted modules may remain imported by tests or adapters; compile/import checks catch this before CI.
- Do not rely on globally installed tools; use the project virtual environment so local and CI tool versions match.

## Delivery record

In the final message, report the exact commit SHA and the exact lint, format, and test commands run, including any intentionally skipped smoke or model-download checks.
