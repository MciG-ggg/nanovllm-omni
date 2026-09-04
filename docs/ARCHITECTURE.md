# Architecture: how nanovllm-omni maps to vllm-omni

This document is for readers who want to understand vllm-omni's design
by reading `nanovllm-omni`. It pairs each module in this repository
with its counterpart in vllm-omni, walks through the request lifecycle
end-to-end, and lists the in-scope divergences you should know about.

If you have not read the [README](../README.md) and [AGENTS.md](../AGENTS.md)
yet, read them first. The "Definition of aligned" section in AGENTS.md
defines what counts as a divergence from vllm-omni and what does not.

## Module map

Every `nanovllm_omni/` module has a vllm-omni counterpart the reader can
jump to. Some nanovllm-omni modules have **no** counterpart (they exist
only because the single-process runtime collapses what vllm-omni does in
N subprocesses into one in-process structure); those are flagged with
`— ` below.

| `nanovllm_omni/` | `vllm_omni/` counterpart | Notes |
|---|---|---|
| `__init__.py` | `__init__.py` | Public-API re-exports only |
| `config/registry.py` | `config/pipeline_registry.py`, `config/stage_config.py` | `StageExecutionType`, `register_pipeline`, `resolve_pipeline_config`, `load_deploy_config` |
| `config/params.py` | `config/...`, `engine/arg_utils.py` | `SamplingParams` + `OmniEngineArgs` (deliberately smaller field set; see divergences below) |
| `outputs.py` | `outputs.py` | `OutputModality` / `OutputModalityNames`, `MultimodalPayload`, `OmniRequestOutput`, `AudioPayload` / `TextArtifact` / `ImageArtifact` / `ActionArtifact` |
| `entrypoints/base.py` | `entrypoints/omni_base.py`, `config/config_factory.py` | `OmniBase`, `try_infer_model_type` cascade |
| `entrypoints/omni.py` | `entrypoints/omni.py` | `Omni` class |
| `entrypoints/async_omni.py` | `entrypoints/async_omni.py` | `AsyncOmni` class |
| `engine/runtime.py` | `engine/runtime.py` (or `StageRuntime`) | Single-process `Runtime`; collapses vllm-omni's per-stage `StageEngineCoreProc` pool (see divergence) |
| `engine/runtime_scheduler.py` | `engine/scheduler.py` | Per-stage continuous batching (in-process) |
| `engine/load_balancer.py` | `engine/load_balancer.py` | StagePool with `num_replicas ≥ 2` + RoundRobin LB (in-process) |
| `engine/executor.py` | `engine/executor.py` | Stage executor |
| `serving/openai_adapter.py` | `entrypoints/cli/serve.py` | `POST /v1/chat/completions` adapter; envelope + base64 WAV/PNG |
| `serving/{server.py}` | `experimental/*/serving/*` (FastAPI/uvicorn) | **stdlib `http.server` — divergence** |
| `models/<family>/pipeline.py` | `models/<family>/pipeline.py` | Per-family `PipelineConfig` definition + `register_pipeline(...)` call |
| `models/<family>/bundle.py` (MiniMind-O) | `models/<family>/bundle.py` | Holds the loaded checkpoint + codec + tokenizer in one process |
| `models/<family>/attention.py` | `models/<family>/attention.py` | Fused QKV / gate-up / RMSNorm / RoPE variants |
| `optim/` | `optim/`, `engine/optim/` | Bench + optimization helpers |

For each vllm-omni file path above, `nanovllm-omni`'s read-only reference
clone lives at `/Users/mcig/Projects/vllm-omni` on the developer's
machine; otherwise the same path is reachable from
<https://github.com/vllm-project/vllm-omni>.

## How a request flows

End-to-end request lifecycle, with the module that owns each step:

1. **User constructs `Omni(<model_id>)`** — `entrypoints/omni.py`
   - Resolves to a `PipelineConfig` via `resolve_pipeline_config(...)`,
     which reads `OMNI_PIPELINES` (`config/registry.py`).
   - Calls `OmniBase.try_infer_model_type(...)` to disambiguate HF
     architectures that share a `model_type` (`entrypoints/base.py`).
   - Loads the matching `PipelineConfig` and merges its deploy defaults
     via `merge_pipeline_deploy(...)` (`config/registry.py`).
2. **`engine/runtime.py`** is instantiated with the merged
   `(stage, defaults)` tuple. The runtime holds one in-process
   `ModelsBundle` per family (e.g. `MinimindBundle` for MiniMind-O in
   `models/minimind_omni/bundle.py`).
3. **`Omni.generate([prompt], SamplingParams(...))`** —
   `entrypoints/omni.py` calls into the runtime's per-stage
   scheduler (`engine/runtime_scheduler.py`) for continuous batching,
   and the per-stage load balancer (`engine/load_balancer.py`) picks a
   `(stage_id, replica_id)` for each stage under load.
4. **Each stage executor** (`engine/executor.py`) feeds the prompt or
   the previous stage's hidden states through the stage's loaded
   model(s), wrapping the result in the stage's
   `final_output_type` (`audio` / `image` / `text` / `actions`).
5. **`OmniRequestOutput`** is assembled in `outputs.py`:
   - `multimodal_output` carries the modality payload (audio bytes,
     image bytes, action array).
   - `custom_output` carries optional non-modal extras (e.g. the
     MiniMind-O transcript; see note below).
   - `to_dict()` materializes the JSON envelope: tensors → lists,
     bytes → base64, artifacts include a `<key>_metadata` sibling
     (`sample_rate` for WAV, `width`/`height` for PNG,
     `token_ids` for text).
6. **For the HTTP seam** (`serving/openai_adapter.py` +
   `serving/{server.py}`), the same `to_dict()` output becomes the
   `POST /v1/chat/completions` body, preserving the OpenAI envelope
   (`id`, `object`, `created`, `model`, `choices[].message`,
   `usage`).

## In-scope divergences from vllm-omni

Each divergence here is deliberate, documented, and locked by at least
one test. The complete list lives in AGENTS.md's
"Definition of aligned" section; this table is the developer-facing
checklist.

| Divergence | Where it shows up in nanovllm-omni | Where it's documented | Locked by |
|---|---|---|---|
| `PipelineConfig` registered under `name` (vllm-omni uses `model_type`) | `config/registry.py:register_pipeline` | AGENTS.md "Definition of aligned" | `tests/test_registry_resolver.py` |
| Same-name re-registration **silently overrides** (vllm-omni validates + warns) | `config/registry.py:register_pipeline` | AGENTS.md "Definition of aligned" | `tests/test_registry_resolver.py` (drift-lock) |
| Single-process runtime (no `StageEngineCoreProc` pool, no multi-GPU / tensor-parallel) | `engine/runtime.py` (one `Runtime` per `Omni`) | README "Scope → Single-card, single-process runtime" | `tests/test_batched_generation.py` (Q10a) |
| `OmniEngineArgs` has a deliberately smaller field set than vllm-omni's | `config/params.py:OmniEngineArgs` | AGENTS.md "Alignment rules" | the dataclass itself |
| HTTP server uses stdlib `http.server` (not FastAPI/uvicorn) | `serving/{server.py}` + `serving/openai_adapter.py` | README "Scope" | `tests/test_serving_http.py` |
| `_custom_output` ↔ `custom_output` serialization name on `OmniRequestOutput` | `outputs.py:OmniRequestOutput` | inline property docstring | `tests/test_outputs.py` |

If you spot a divergence that is not in this table, it is either a
bug or an undocumented in-scope difference. Open an issue and reference
this section.

## Reading order for vllm-omni newcomers

If you have never read vllm-omni's source before, this 30-minute path
through `nanovllm-omni` mirrors vllm-omni 1:1:

1. `entrypoints/omni.py` ↔ `vllm_omni/entrypoints/omni.py`
2. `entrypoints/base.py` ↔ `vllm_omni/entrypoints/omni_base.py`
3. `config/registry.py` ↔ `vllm_omni/config/pipeline_registry.py`
4. `config/params.py` ↔ `vllm_omni/engine/arg_utils.py` (focus on `OmniEngineArgs`)
5. `outputs.py` ↔ `vllm_omni/outputs.py` (focus on `OmniRequestOutput.to_dict()`)
6. `engine/runtime.py` ↔ `vllm_omni/engine/runtime.py` (note the single-process divergence)
7. `serving/openai_adapter.py` ↔ `vllm_omni/entrypoints/cli/serve.py`
8. Pick one family — `models/minimind_omni/pipeline.py` is the
   richest — and walk its bundle, attention, and stage modules.

After step 8, you should be able to read any other vllm-omni file
without orientation.

## Adding a model family

See [`.agents/skills/add-new-model/SKILL.md`](../.agents/skills/add-new-model/SKILL.md)
for the full runbook. The short version: each family adds a
`models/<family>/pipeline.py` that builds a `PipelineConfig` and calls
`register_pipeline(...)` at import time; topology goes in code, sampling
defaults go in `deploy/<family>.yaml`; the family auto-loads because
`_load_builtin_pipelines()` imports the module on package init.

## What's intentionally not here

`nanovllm-omni` does not implement:

- Video generation.
- Multi-model pipelines beyond the four listed families
  (MiniMind-O, SmolVLM, SD-Turbo, SmolVLA).
- Distributed execution, multi-card tensor-parallel, pipeline-parallel
  schedulers.
- WebSockets, streaming responses.
- FastAPI/uvicorn serving.

If you need any of those, this is the wrong project — `vllm-omni` is
the upstream that does.
