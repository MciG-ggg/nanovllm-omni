# Aligned interfaces

The public nanovllm-omni API follows the corresponding vllm-omni interfaces where practical.

| nanovllm_omni class | vllm-omni counterpart |
|---|---|
| `nanovllm_omni.entrypoints.Omni` | `vllm_omni.entrypoints.Omni` |
| `nanovllm_omni.entrypoints.AsyncOmni` | `vllm_omni.entrypoints.AsyncOmni` |
| `nanovllm_omni.engine_args.SamplingParams` | `vllm_omni.engine_args.SamplingParams` |
| `nanovllm_omni.engine_args.OmniEngineArgs` | `vllm_omni.engine_args.OmniEngineArgs` |
| `nanovllm_omni.outputs.OmniRequestOutput` | `vllm_omni.outputs.RequestOutput` |
| `nanovllm_omni.config_registry.PipelineConfig` | `vllm_omni.config.PipelineConfig` |
| `nanovllm_omni.config_registry.StageConfig` (new) | `vllm_omni.config.StagePipelineConfig` |
| `nanovllm_omni.config_registry.DeployConfig` | `vllm_omni.config.DeployConfig` |
| `nanovllm_omni.config_registry.DeployStageConfig` | `vllm_omni.config.StageDeployConfig` |
| `nanovllm_omni.config_registry.StageExecutionType` (TK-016 phase 2) | `vllm_omni.config.StageExecutionType` |
| `nanovllm_omni.config_registry.resolve_stage_factory` (TK-016 phase 2) | `vllm_omni.config.resolve_stage_factory` |
| `nanovllm_omni.engine.runner.PipelineRunner` (new) | no vllm-omni analog (replaces `StagePool` role for single-GPU) |
| `nanovllm_omni.engine.executor.PipelineExecutor` (new) | no vllm-omni analog (replaces `Orchestrator` role for single-process) |

## Alignment boundary

This is consumer-visible interface alignment, not an identical implementation. The supported vertical slice is MiniMind-O audio plus Wan2.2 TI2V diffusion (the latter is gated by TICKET-06). It intentionally differs from vllm-omni by omitting:

- Multi-replica routing, load balancing, per-stage replica metrics, dynamic replica membership (`vllm_omni.engine.stage_pool.St2_agePool` — 1281 lines).
- Cross-stage request lifecycle, CFG companion dispatch, duplex session tracking, error recovery, dead-replica detection (`vllm_omni.engine.orchestrator.Orchestrator` — 2428 lines).
- Distributed execution, KV transfer, Mooncake connectors, Ray backend.
- Full-duplex S2S APIs, WebSocket endpoints (`/v1/duplex`, `/v1/realtime`, `/v1/realtime/robot/openpi`).
- Vision understanding (LLaVA / InternVL — single-stage vision-language fused models).
- FastAPI / uvicorn / Pydantic / plugin entry points.
- Multi-Arm action policies, robot control loops, simulators.
- `hf_config_predicate` for HF-config-based model-type disambiguation: we use one-key-per-model-family.

These omissions are deliberate scope boundaries, not compatibility bugs. They are documented at the ticket level so future contributors can re-open them when the scope changes.

## StageConfig fields

The minimal field set for `StageConfig` after TICKET-02 (see `.scratch/aligned-interfaces/issues/02-omni-end-to-end.md` for the design record):

| Field | Type | Purpose | Used by |
|---|---|---|---|
| `stage_id` | `int` | Stable ordering of stages within a pipeline | All models |
| `name` | `str` | Human-readable identifier (e.g. `"thinker"`, `"dit"`) | All models |
| `kind` | `StageExecutionType` | Execution class: `LLM_AR`, `LLM_GENERATION`, `DIFFUSION`, `CODEC`. `StrEnum` mirroring vllm-omni. | All models |
| `factory` | `str` | Dotted-path to the stage factory callable (`"package.module:attr"`). Resolved lazily via `resolve_stage_factory`. | All models |
| `process_input` | `str \| None` | Dotted-path to the bridge hook (`"package.module:attr"`), or `None` for identity pass-through. | minimind_o (TICKET-05), reserved for future multi-stage models |
| `input_sources` | `tuple[int, ...]` | Stage ids whose outputs feed this stage. Empty for the first stage. | All models |
| `is_terminal` | `bool` | Marks the final stage whose output is the user-facing result. | All models |
| `final_output_type` | `str \| None` | One of `"audio"`, `"video"`, `"image"`, `"text"`, `"actions"`. Drives `OmniRequestOutput.multimodal_output` key. SmolVLA uses `"actions"` with an `ActionArtifact`. | All models |
| `model_subdir` | `str \| None` | Subfolder name within the model checkpoint (e.g. `"language_model"`). | Reserved for TICKET-06 / future TTS-same-shape ticket |
| `tokenizer_subdir` | `str \| None` | Subfolder for the tokenizer. | Reserved for future TTS-same-shape ticket |
| `diffusers_class_name` | `str \| None` | Stable class name for diffusers-based diffusion stages. | TICKET-06 (Wan2.2) |

## PipelineConfig fields

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Canonical model identifier (e.g. `"minimind_o"`, `"smolvla"`, `"wan2_2_ti2v"`) |
| `stages` | `tuple[StageConfig, ...]` | Ordered stages. Length 1 = single-stage (diffusion); length N = multi-stage AR pipeline. |
| `default_deploy_config_name` | `str` | Filename in `deploy/` whose contents are loaded into `DeployConfig`. |
| `registration_handles` | `tuple[str, ...]` | Alternate keys under which this pipeline is registered (e.g. HF repo id `"jingyaogong/minimind-3o"`). Defaults to `(name,)`. |

## Engine layering

vllm-omni's `StagePool` (1281 LOC) and `Orchestrator` (2428 LOC) are designed for **multi-replica routing** and **cross-stage request lifecycle management** in a distributed setting. nanovllm-omni targets a **single-process, single-GPU** installation (the user's workstation, plus a rented GPU for diffusion). The corresponding abstractions are split into two narrow classes:

- **`PipelineRunner`**: synchronous single-replica pipeline runner. Runs one request through `PipelineConfig.stages` sequentially. ~100 LOC. Knows nothing about concurrency.
- **`PipelineExecutor`**: async wrapper around `PipelineRunner` for HTTP / `AsyncOmni` use. Uses `asyncio.run_in_executor` to dispatch sync `PipelineRunner.run` calls. `max_concurrent` defaults to `1` (single GPU). ~80 LOC.

The split mirrors vllm-omni's "pool routes to replicas, orchestrator tracks requests" division, but in single-process form: the runner is the only consumer of the pipeline config, and the executor is the only consumer of the runner. No class is named `StagePool` or `Orchestrator` in `nanovllm_omni.engine/` — those names are reserved for the future when multi-replica support is reintroduced (TK-007).

## When an aligned symbol changes

When an aligned public symbol or response field changes, update this table, add a focused contract test, and run the CI-equivalent checks documented in `AGENTS.md`.
