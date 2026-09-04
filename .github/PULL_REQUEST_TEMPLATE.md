## Summary

One paragraph: what changed and why.

## Linked issue

Closes #<issue> or "no issue, opening PR directly because <reason>".

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New model family
- [ ] New capability (registry, serving, optimization)
- [ ] Refactor / cleanup
- [ ] Documentation only
- [ ] Other (describe):

## Public-API impact

- [ ] No public symbol changed
- [ ] Public symbol added (new field / new class / new function)
- [ ] Public symbol changed (field rename, signature change,
      serialization-shape change)
- [ ] Public symbol removed (deprecation handled)

If any box other than the first is checked, paste the relevant
diff snippet here and call out which downstream code/tests need to
track it.

## Tests added

- [ ] `tests/test_<area>.py` updated or added
- [ ] Run `python -m pytest -m "not smoke" -v` locally — green
- [ ] For Python-seam changes: `Omni(...).generate(...)` smoke run
- [ ] For HTTP-seam changes: `POST /v1/chat/completions` smoke run

## CI-equivalent checks

Run `scripts/pre-push` (or the equivalent block in CONTRIBUTING.md)
locally and paste the last line of each command:

- `ruff check nanovllm_omni/ tests/`: ___
- `black --check nanovllm_omni/ tests/`: ___
- `python -m pytest -m "not smoke" -v`: ___
- `python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"`: ___
- `python -m compileall -q nanovllm_omni`: ___

## Notes for reviewer

Anything the reviewer should read first, any intentional divergence
from vllm-omni, any follow-up work you're punting to a later PR.
