# examples/

Runnable, weight-bearing examples for the aligned API. Layout mirrors
vllm-omni's `examples/`: split by run mode first (`offline_inference/`,
`online_serving/`), then one folder per model family. See
`.scratch/aligned-interfaces/SPEC.md` for the locked Python / HTTP
seam contracts every example exercises.

## Layout

```
examples/
  AGENTS.md                        # this file
  offline_inference/
    <model>/                       # one folder per registered model_type
      README.md                    # Setup + Run sections
      end2end.py                   # main entrypoint, builds Omni + writes output
      <other>.py                   # additional entries (batched eval, helpers)
      run_*.sh                     # 2-line wrappers, chmod +x
  online_serving/
    <model>/
      README.md
      <client>.py / curl_*.sh      # client for the stdlib HTTP server
      save_*.py                    # decode server JSON -> artifact
```

Rule of thumb: **one folder per model family**, not per script. A
model family with two natural demo shapes (`synthetic_obs.py` +
`libero_eval.py`) lives in **one** folder with two entries, not two
folders.

## Naming

- Folder name = `nanovllm_omni.config.registry.OMNI_PIPELINES` key
  (snake_case). MiniMind-O is registered as `minimind_o`; SmolVLA is
  `smolvla`. New folders must match a registered key — see
  `register_pipeline(...)` in `nanovllm_omni/config/registry.py`.
- README first heading is `# <Model>: <Run mode>`. Use the run-mode
  suffix from vllm-omni (`Offline inference` / `Online serving`).
- Entrypoint `.py` files describe themselves in the **first line of
  the docstring**: `Smoke: Omni.generate -> <artifact>` or
  `L1 demo: ...`. Setup / run instructions go in the README, not
  the docstring (the docstring stays short — `--help` shows it).
- Shell wrappers are `run_<entrypoint>.sh`. Body is always two lines:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  cd "$(dirname "$0")"
  exec python <entrypoint>.py "$@"
  ```

  `cd "$(dirname "$0")"` matters — `deploy/<model>.yaml` resolves
  relative to the repo root only after the `cd`.

## Default `--model` paths

Project convention: **entrypoints default `--model` to a local path
under `pretrained/`, not to a HuggingFace Hub id**. This matches
the offline-first bundle loader (`1278891 fix(bundle): align
_resolve_snapshot with vllm-omni's offline-first semantics`) and
lets WSL run the same script as Mac without a Hub login.

```bash
# Default for every example entrypoint:
--model  pretrained/<model-folder-name>
--mimi   pretrained/<model-folder-name>   # if the model has one
```

The `Setup` section in each model README lists the matching
`hf download --local-dir pretrained/<name>` commands. A user who
prefers Hub ids can still pass them explicitly:

```bash
python .../end2end.py --model jingyaogong/minimind-3o
python .../libero_eval.py --model HuggingFaceVLA/smolvla_libero
```

The bundle resolver accepts both forms; see
`nanovllm_omni/entrypoints/base.py:OmniBase._resolve_pipeline`.

## Required sections in every README

In this order, modeled on vllm-omni's per-model READMEs:

1. **Title + one-line scope** — "MiniMind-O: Offline inference".
2. **Setup** — `hf download <repo> --local-dir <path>` commands for
   every checkpoint the example uses. The bundle loader is
   offline-first (see `tests/test_bundle_resolve_snapshot.py`), so
   these commands are mandatory, not optional. Mention `HF_HUB_OFFLINE=1`.
3. **Run examples** — one `bash` block per natural demo shape. Each
   block runs `cd examples/<...>/<model>` first, then `bash run_*.sh ...`.
   Show the realistic path arguments, not placeholders.
4. **Notes** — pipeline registry key, deploy YAML location, dtype /
   VRAM hints, what tests pin the contract.

Avoid:

- A "Quickstart" section that points at a nonexistent path (the
  project README has the canonical quickstart; per-model READMEs
  describe only themselves).
- Hard-coded machine paths in shell wrappers. Parameterize via
  `--model` / `--mimi` / `--dtype` and document the realistic
  defaults in the README.

## How to add a new model example

1. Register the pipeline first if it's not in
   `nanovllm_omni.config.registry.OMNI_PIPELINES`. Use
   `register_pipeline(...)` and add a `deploy/<model>.yaml` with
   the per-stage sampling / resource defaults — these are read by
   `Omni(...).generate(...)` via `load_deploy_config`.
2. Create the folder:

   ```bash
   mkdir examples/offline_inference/<model>
   ```

3. Add README + entrypoint + `run_*.sh`. Use the existing
   `minimind_o/` or `smolvla/` READMEs as templates — copy the
   "Setup" / "Run examples" / "Notes" skeleton, then fill in the
   model-specific `hf download` commands.
4. chmod +x the shell wrappers.
5. Run pre-commit:

   ```bash
   ruff check examples/
   black --check examples/
   ```

6. Commit on its own:

   ```bash
   git add examples/<...>/<model>/
   git commit -m "examples(<model>): initial offline-inference scaffold"
   ```

Online-serving folders are reserved for models that actually have an
HTTP server endpoint today. MiniMind-O is the only one; SmolVLA is
action-only and has no `online_serving/<model>/` entry.

## What this directory does NOT contain

- Unit tests — those live in `tests/`. Examples are weight-bearing
  smoke surfaces, not test fixtures; the only assertion they make is
  on artifact shape or WAV validity.
- Configuration. Pipeline topology stays in code
  (`nanovllm_omni/config/registry.py`); runtime knobs stay in
  `deploy/<model>.yaml`. Examples pass `--deploy-config` only to
  point at an alternate YAML for experiments.
- A top-level runner script. Each example is independently runnable;
  vllm-omni has the same convention.