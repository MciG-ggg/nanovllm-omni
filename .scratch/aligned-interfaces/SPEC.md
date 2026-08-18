# nanovllm-omni ↔ vllm-omni Interface Alignment

**Status:** ready-for-agent

## Problem Statement

The user is familiar with vllm-omni's API surface — `Omni`, `AsyncOmni`, `OmniEngineArgs`, `SamplingParams`, `OmniRequestOutput`, `/v1/chat/completions` returning multimodal payloads, the pipeline registry pattern, deploy YAMLs separate from pipeline topology. nanovllm-omni currently exposes a different surface: `Orchestrator.submit()`, `Pipeline.run()`, hardcoded sampling constants in each stage, a single stdlib HTTP route, no sampling-params object, no output object, no deploy YAML. Switching between the two engines today requires rewriting the consumer code from scratch.

Beyond surface mismatch, the README documents features that do not exist (a FastAPI serving app, `examples/chat.py`, `examples/image_gen.py`, `examples/vla.py`, a VLA stage, a diffusion stage, vision-LLM models). The README also advertises "AR / Diffusion / VLA / Omni" as four implemented stage kinds when only the MiniMind-O audio pipeline is actually wired up. The current `docs/design_mapping.md`, `docs/deployment.md`, and `docs/TODO.md` are referenced from the README but do not exist on disk.

## Solution

Build a parallel API surface on top of nanovllm-omni's currently-implemented MiniMind-O audio pipeline that mirrors vllm-omni's class names, signatures, and behaviors 1:1. Two observable seams are verified:

- **Python seam**: `from nanovllm_omni import Omni, SamplingParams, OmniRequestOutput`; `Omni("model_id").generate(prompts, sampling_params)` returns `list[OmniRequestOutput]` with `multimodal_output["audio"]` set to a valid WAV payload.
- **HTTP seam**: `POST /v1/chat/completions` returns an OpenAI-shape `chat.completion` JSON whose `choices[0].message.audio.data` decodes to a valid WAV file.

Both seams are exercised on a single MiniMind-O run.

Concretely, the bespoke `Orchestrator` / `Pipeline` / `Stage[Input,Output]` ABC / typed `Payload` dataclasses are replaced with a layout that follows vllm-omni's module structure: `config/` for topology dataclasses and the pipeline registry, `engine/` for the runtime, `entrypoints/` for the user-facing `Omni` / `AsyncOmni` / `OmniBase` classes, `outputs.py` for `OmniRequestOutput`, `models/<family>/` for per-family pipeline constants and stage factories, `deploy/` for YAML-based runtime config, `core/` for shared errors. Pipeline topology becomes a `PipelineConfig` declared as a module-level constant in the per-family `pipeline.py`, looked up via `resolve_pipeline_config(model_type)`. Per-stage runtime parameters (devices, sampling defaults, memory budgets) move into a `deploy/<model>.yaml` loaded by `load_deploy_config()`. The user-facing `Omni` class accepts a single positional `model` string and forwards everything else to `OmniEngineArgs`.

The README and any docs that reference non-existent files are reconciled with what the code actually does in a separate Phase-1 ticket (TICKET-01) before any code is touched.

## User Stories

1. As a developer familiar with vllm-omni, I want `from nanovllm_omni import Omni`, so that I can use the class name I already know without learning a new entry point.

2. As a developer familiar with vllm-omni, I want `from nanovllm_omni import AsyncOmni`, so that my async code transfers directly to nanovllm-omni.

3. As a developer familiar with vllm-omni, I want `from nanovllm_omni import SamplingParams`, so that sampling parameter construction looks identical to vLLM.

4. As a developer familiar with vllm-omni, I want `from nanovllm_omni import OmniRequestOutput`, so that downstream code reads the same fields it would in vllm-omni.

5. As a developer, I want `Omni("jingyaogong/minimind-3o")` to resolve the right pipeline automatically, so that I do not have to pass a separate config object.

6. As a developer, I want `Omni("model", enforce_eager=True, gpu_memory_utilization=0.8)` to forward engine arguments to the underlying engine, so that deploy knobs work the same way they do in vllm-omni.

7. As a developer, I want `Omni.generate(prompts, sampling_params=...)` to return `list[OmniRequestOutput]`, so that result-handling code is shape-compatible with vllm-omni examples.

8. As a developer, I want `AsyncOmni.generate(...)` to return an async iterator, so that streaming-style flows compose with my existing async code.

9. As a developer, I want `SamplingParams(temperature=0.7, max_tokens=512)` to construct without errors, so that I can copy vLLM examples verbatim.

10. As a developer, I want the per-stage sampling defaults to live in `deploy/minimind_omni.yaml`, so that hardcoded constants in model code disappear.

11. As a deploy operator, I want `load_deploy_config(path)` to return a typed `DeployConfig`, so that YAML mistakes surface at load time and not deep inside the engine.

12. As a developer, I want `resolve_pipeline_config("minimind_o")` to return the same `PipelineConfig` that the engine will use, so that pipeline introspection is consistent.

13. As a contributor, I want `OMNI_PIPELINES["new_model"] = MyPipeline(...)` to register a new model family in one line, so that adding a model is not a multi-file exercise.

14. As an HTTP client, I want `POST /v1/chat/completions` to accept OpenAI-shape requests, so that I can drive it with the OpenAI client SDK.

15. As an HTTP client, I want `choices[0].message.audio.data` to be a base64-encoded WAV, so that audio payloads come back in-band with the response.

16. As an HTTP client, I want the response to include `usage`, `object`, and `created` fields, so that the response is OpenAI-shape and not a custom shape.

17. As a developer, I want a `nanovllm_omni.OmniBase` shared base class, so that the constructor behavior is identical between sync and async entry points.

18. As a developer, I want stage implementations to be factory functions returning runtime objects (not abstract base classes), so that pipeline topology is data-driven.

19. As a developer, I want `nanovllm_omni.config.OmniEngineArgs(**kwargs)` to be the kwargs-to-config seam, so that any keyword I pass to `Omni(...)` flows through it.

20. As a developer, I want `examples/minimind_omni.py` to load MiniMind-O and produce an `audio.wav`, so that I can run a smoke test of the aligned API in under a minute.

21. As a developer, I want `docs/aligned_interfaces.md` to map each nanovllm-omni class to its vllm-omni counterpart, so that I can navigate the two engines side by side.

22. As a CI maintainer, I want `tests/test_sampling_params.py`, `tests/test_engine_args.py`, `tests/test_pipeline_registry.py`, and `tests/test_omni_request_output.py`, so that regressions in the aligned interface surface are caught.

23. As a reader of the README, I want the "Supported models" table to list only models that actually run, so that I do not waste time on aspirational entries.

24. As a reader of the README, I want any reference to a non-existent file (`serving/app.py`, `examples/chat.py`, `diffusion/scheduler.py`, etc.) to be removed, so that the README does not lie to me.

25. As a reader of the README, I want a "Status" section that names what is implemented vs. what is aspirational, so that I can calibrate my expectations before I start using the engine.

## Implementation Decisions

The following decisions are locked. The new agent must not relitigate them; deviations must be raised explicitly with the user.

### Naming and entry shape (mirror vllm-omni)

- Top-level public symbols exported from the package root: `Omni`, `AsyncOmni`, `OmniBase`, `SamplingParams`, `OmniRequestOutput`, plus `OMNI_PIPELINES` and `resolve_pipeline_config` accessible via `nanovllm_omni.config`.
- `Omni.__init__(model: str, **kwargs)`. The single positional argument is the model identifier; every other keyword flows through `OmniEngineArgs`.
- `Omni.generate(prompts, sampling_params=None, use_tqdm=True)` returns `list[OmniRequestOutput]`. Sync wrapper around the async engine.
- `AsyncOmni.generate(...)` returns an async iterator of `OmniRequestOutput`.

### Internal architecture (drop the strong-typed stage contract)

- No `Stage[Input, Output]` ABC, no typed `Payload` dataclasses. Stages are discovered through the registry and configured by data (`PipelineConfig`, `StagePipelineConfig`).
- Pipeline topology is a frozen `PipelineConfig` dataclass held as a module-level constant in the per-family `pipeline.py`.
- Runtime knobs (devices, sampling defaults, memory budgets) live in a separate mutable `DeployConfig` loaded from YAML.
- Stage implementations are factory functions or plain classes that return runtime objects — never an ABC the user subclasses.

### Configuration split

- `OmniEngineArgs` is the programmatic seam — kwargs forwarded to the engine.
- A `StageConfigFactory.create_from_model(model, ...)` builds the runtime per-stage configs by combining the resolved `PipelineConfig` with the `DeployConfig`.
- `load_deploy_config(path)` parses YAML into a `DeployConfig`.
- `merge_pipeline_deploy(pipeline_cfg, deploy_cfg)` produces the final per-stage configs.

### Pipeline registry

- `OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc]` — a module-level Python literal in the registry module, one entry per model_type.
- `resolve_pipeline_config(model_type, hf_config=None)` returns the concrete `PipelineConfig` (or `None` if not registered). If the value is callable, invoke it with `hf_config` to support cases where pipeline shape depends on the HF config.
- `register_pipeline(pipeline, model_type)` is the out-of-tree extension hook.

### Sampling

- `SamplingParams` is a stdlib `@dataclass(frozen=True)` with fields, names matching vLLM exactly: `temperature: float = 1.0`, `top_p: float = 1.0`, `top_k: int = -1`, `max_tokens: int = 16`, `stop: list[str] | None = None`, `seed: int | None = None`, `n: int = 1`, `extra: dict = field(default_factory=dict)`.
- Per-stage sampling defaults live in `deploy/<model>.yaml` under `stages[].default_sampling_params`. Stage code reads its defaults from there; the dataclass fields above are per-request overrides.
- The hardcoded values currently in `models/minimind_omni.py` (thinker temperature=0.7 / `max_new_tokens=512`; talker temperature=0.2 / `watchdog_limit=192`) move into the deploy YAML.

### Output object

- `OmniRequestOutput` is a dataclass mirroring vllm-omni's structure with three branches: pipeline, diffusion, error. Constructors: `from_pipeline(...)`, `from_error(...)`, `from_diffusion(...)`. Properties: `unwrap()`, `is_pipeline_output`, `is_diffusion_output`.
- For MiniMind-O audio, `from_pipeline(..., final_output_type="audio")` returns an `OmniRequestOutput` whose `multimodal_output["audio"]` is the `AudioPayload` with `wav_bytes()` returning valid 16-bit PCM mono.

### HTTP layer

- Stays on stdlib `http.server.ThreadingHTTPServer`. No FastAPI, no uvicorn.
- Single route: `POST /v1/chat/completions`.
- Response shape: OpenAI `chat.completion` JSON. Required fields: `id` (string), `object: "chat.completion"`, `created` (unix seconds), `model` (string), `choices[0].index`, `choices[0].message.role: "assistant"`, `choices[0].message.audio = {"data": "<base64 wav>", "format": "wav", "sample_rate": 24000}`, `usage = {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}`. Field names mirror vllm-omni exactly.

### Module structure

- New top-level subpackages / modules: `config/`, `engine/`, `entrypoints/`, `outputs.py`, `models/`, `core/`. Skip `quantization/`, `distributed/`, `model_executor/` — these have no analog in nanovllm-omni's scope.
- Per-model files: `models/<family>/pipeline.py` holds the module-level `*_PIPELINE` constant; `models/<family>/stages.py` holds stage factory functions.
- Deploy files: `deploy/<model>.yaml`. YAML contents are resource / sampling knobs only — never pipeline topology.

### Validation

- All configs are stdlib `@dataclass` (no Pydantic). Validation is per-class `validate() -> list[str]` methods (mirror vllm-omni's pattern).

### Dependency footprint

- No new third-party dependencies. Existing: `transformers`, `torch`, `numpy`, `pyyaml`. SamplingParams, config, output, registry are all built on stdlib.

### Delivery phasing

This spec is delivered as four tickets (`.scratch/aligned-interfaces/issues/01..04-*.md`), each a vertical slice. The first ticket is doc-only and independent. The next two introduce the new module structure alongside the old `Orchestrator` (expand phase). The last is the contract step that deletes the old API.

## Testing Decisions

A good test for this alignment work exercises the user-observable surface (the two seams), not the implementation details underneath. A test that verifies "Stage ABC has a `run()` method" is bad; a test that verifies "`Omni.generate(prompts, sp)` returns an `OmniRequestOutput` with `multimodal_output["audio"]` set and `wav_bytes()` returning valid 16-bit PCM" is good.

### Two seams, both tested

1. **Python seam**: `Omni(model).generate(prompts, sampling_params=...)` returns a non-empty `list[OmniRequestOutput]`. Each output's `multimodal_output["audio"].wav_bytes()` returns valid 16-bit PCM mono at the model's sample rate (verifiable by writing to disk and re-opening with `wave.open`).

2. **HTTP seam**: `curl -X POST /v1/chat/completions` with a `{"messages": [...]}` body returns a response whose `choices[0].message.audio.data` decodes to a non-empty WAV file openable by Python's `wave.open`.

### Modules tested in isolation

- `tests/test_sampling_params.py` — SamplingParams construction with each field; default values; `extra` dict; frozen behavior.
- `tests/test_engine_args.py` — OmniEngineArgs accepts the kwargs that `Omni` passes through; rejects unknown kwargs or stores them in `extra`.
- `tests/test_pipeline_registry.py` — `OMNI_PIPELINES["minimind_o"]` resolves to the registered `PipelineConfig`; `register_pipeline(...)` adds an entry; `resolve_pipeline_config(...)` returns `None` for unregistered keys.
- `tests/test_omni_request_output.py` — `from_pipeline(...)`, `from_error(...)`, `from_diffusion(...)` constructors; `unwrap()`, `is_pipeline_output`, `is_diffusion_output` properties.

### End-to-end integration

- `examples/minimind_omni.py` runs the Python seam against MiniMind-O with default sampling params, writes `audio.wav`, exits non-zero on failure. Acts as both demo and smoke test.
- An HTTP integration script (curl-based or stdlib-urllib-based) that runs against the server and checks the response shape.

### Prior art

- The existing `tests/` directory in nanovllm-omni (test layout, fixture style).
- vllm-omni's `tests/` directory is a reference for shape compatibility, but its code is not imported — only field names and assertion shapes are inspected.

## Out of Scope

This spec covers only the audio-output alignment for MiniMind-O. The following are explicitly NOT in scope and must not be implemented under the guise of this work:

- **Other modalities**: image generation, video generation, vision understanding, speech-to-text, image-to-image, image-to-video, video editing. `StageConfig.kind ∈ {diffusion, action, vision_ar}` is accepted by config validation but not implemented.
- **Multi-model pipelines**: any pipeline combining more than one independent model. Out of scope until a second model family lands.
- **AsyncOmni duplex APIs** (`open_duplex_session_async`, `append_duplex_input_async`, `signal_duplex_turn_async`, etc.) — vllm-omni's full-duplex S2S surface.
- **WebSocket endpoints** (`/v1/duplex`, `/v1/realtime`, `/v1/realtime/robot/openpi`, etc.).
- **Diffusion sampling params** (`OmniDiffusionSamplingParams` analog) — no diffusion stage implemented.
- **Plugin entry points** (`vllm.general_plugins`) for out-of-tree model registration. The `register_pipeline()` Python API is the only registration surface.
- **Custom chat templates** — Jinja templates live in model repos, not in nanovllm-omni.
- **KV transfer / cross-stage connectors** (`omni_connectors/`, `OmniCoordinator`, Mooncake, etc.) — single-process execution only.
- **Ray backend / distributed execution** — local engine only.
- **Pydantic config models** — stdlib dataclasses only.
- **FastAPI / uvicorn serving** — stdlib `http.server` only.
- **Pip console-script entry points** (`vllm-omni`-style CLI command). The HTTP server runs via `python -m nanovllm_omni.serving.openai_adapter`.
- **Out-of-tree plugin auto-registration** via setuptools entry points. `register_pipeline()` must be called explicitly.

## Further Notes

### Reference for the new agent

When mapping nanovllm-omni's new code to vllm-omni's layout, the new agent should explore `/Users/mcig/Projects/vllm-omni/vllm_omni/` to find these analogs (use `find` and `grep`; the layout is stable but paths are not committed to this spec):

- The user-facing classes (`Omni`, `AsyncOmni`, `OmniBase`) live in vllm-omni's `entrypoints/` package.
- `OmniEngineArgs` and `OmniAsyncEngineArgs` live in vllm-omni's `engine/arg_utils.py`.
- `PipelineConfig`, `StagePipelineConfig`, `StageDeployConfig`, `DeployConfig`, `load_deploy_config`, `merge_pipeline_deploy` live in vllm-omni's `config/stage_config.py`.
- `OMNI_PIPELINES`, `register_pipeline`, `resolve_pipeline_config` live in vllm-omni's `config/pipeline_registry.py`.
- `OmniRequestOutput` lives in vllm-omni's `outputs/__init__.py`.
- Per-family `*_PIPELINE` constants live in `vllm_omni/model_executor/models/<family>/pipeline.py`.

The new agent adapts the structure to nanovllm-omni's smaller scope — it does NOT copy vllm-omni code verbatim, and it does NOT pull in vllm-omni modules.

### Acceptance per ticket

A ticket is done when its acceptance-criteria checkboxes are all green AND the existing `tests/` directory still passes. No ticket may regress `examples/audio.py` until TICKET-04 deliberately rewrites or replaces it.

### Sequence

TICKET-01 → TICKET-02 → {TICKET-03, ...} → TICKET-04. The first is doc-only and can land independently. The second introduces the new module structure alongside the old `Orchestrator` (expand). The third wires the HTTP adapter to the new engine. The fourth is the contract step that deletes the old API and runs the full verification checklist.