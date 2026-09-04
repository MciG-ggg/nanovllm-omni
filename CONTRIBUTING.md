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
pip install -e ".[dev,minimind]"   # adds MiniMind-O path (torch etc.)

# Run the fast feedback loop (same checks as pre-push):
scripts/pre-commit
# Or, equivalently:
ruff check nanovllm_omni/ tests/
black --check nanovllm_omni/ tests/
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

Lifted from AGENTS.md; applies to all new code:

- **Counts**: `num_*` (`num_heads`, `num_layers`, `num_requests`,
  `num_positions`, `num_steps`). No `n_*`. The only exception is
  `SamplingParams.n`, which vLLM locks the name of.
- **Sequences**: `sequence` / `sequence_len` / `max_sequence_len`. No
  `seq` abbreviation.
- **Full words preferred**: `token`, `config`, `max_embeddings`. No
  `tok`, `cfg`, `max_emb`.

**Out of rename scope** (do not change these even if you spot them in
new code):

- Remote-model attributes accessed via
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  (e.g. `n_rep`, `n_local_heads`, `n_local_kv_heads`).
- String / comment / docstring / shape notations like `[B, seq, kv, d]`.
- Serialization-format keys (JSON keys, markdown table headers) — these
  are external schema.
- Public-aligned symbols (`Omni`, `SamplingParams` fields,
  `OmniRequestOutput` fields, etc.) — go through the interface-change
  workflow.

## Where to ask

- **Bug / feature**: open an issue using the templates under
  [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
- **Security**: see [SECURITY.md](SECURITY.md).
- **Anything else**: open a discussion-style issue.
