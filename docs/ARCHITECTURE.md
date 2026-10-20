# Architecture: how nanovllm-omni maps to vllm-omni

This document describes the implementation that is actually in this
repository. The project mirrors vllm-omni's consumer-facing pipeline shape;
it does not reuse vllm-omni's engine internals.

## Runtime boundary

`nanovllm-omni` has one in-process pipeline runner. `OmniBase` resolves a
registered `PipelineConfig`, loads deploy defaults, and owns one
`PipelineExecutor`. `PipelineExecutor` wraps a synchronous `PipelineRunner`
with an asyncio semaphore and a thread pool.

`third_party/nano-vllm` is a vendored fork of
[`GeeeekExplorer/nano-vllm`](https://github.com/GeeeekExplorer/nano-vllm),
installed as an editable dependency via `pyproject.toml`'s
`[tool.uv.sources] nano-vllm = { path = "third_party/nano-vllm", editable = true }`.
**It is the AR engine behind `Omni.generate`** for the LLM-shaped stages
(MiniMind-O thinker + talker, SmolVLM). We do not
fork it lightly — the vLLM-style paged-KV decoder and CUDA-Graph capture
path live there, and re-implementing them locally would be a multi-month
side quest. What we do **not** take from the fork is its
text-only Qwen3 model class, its `LLMEngine`/`Scheduler`, its
`SamplingParams`, and the asyncio server shell — those are replaced by
`PipelineRunner` and our own dataclasses. The fork's `register_model`
registry is preserved so non-fork model classes (e.g. `MiniMindThinker`,
`MiniMindTalker`, `SmolVLMForConditionalGeneration`) can plug in via
`ModelRunner(model_class=...)`.

SmolVLA's `vlm` stage is **not** on the fork: it calls the lerobot
`SmolVLAPolicy` stack directly (see the SmolVLA section below).

### Fork pieces we use

| Fork module | What it gives us |
|---|---|
| `nanovllm.engine.model_runner.ModelRunner` | Per-stage AR runner: weight loader, KV allocate, optional CUDA-Graph capture, NCCL init, decode-graph reuse. One `ModelRunner` per stage. |
| `nanovllm.config.Config` | Stage configuration (`enforce_eager`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `trust_remote_code`). Each stage builds one. |
| `nanovllm.layers.attention.Attention` | Paged-KV attention with the `store_kvcache` Triton kernel. Used by the fork-AR submodules (thinker blocks, talker blocks, SmolVLM text stack). |
| `nanovllm.layers.linear.{QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear}` | Fused QKV / fused gate-up / row-parallel projection. These are what makes the merged QKV a "real" layer, not a runtime monkey-patch. |
| `nanovllm.layers.layernorm.RMSNorm` | Fused RMSNorm. |
| `nanovllm.layers.rotary_embedding.get_rope` | RoPE. |
| `nanovllm.layers.activation.SiluAndMul` | Fused SwiGLU activation. |
| `nanovllm.layers.embed_head.{ParallelLMHead, VocabParallelEmbedding}` | Tied / untied lm head and embedding. |
| `nanovllm.layers.sampler.Sampler` | Top-k / Gumbel-max sampling used by the per-stage decode loop. |
| `nanovllm.utils.context.{set_context, reset_context}` | Module-level `Context` object that the fork `Attention` layer reads at every forward to learn whether the call is prefill vs decode. The `StageRunner` adapter installs/clears it. |

### The one piece of fork-runtime monkey-patching

`engine/stage_runner.py::_ensure_dist` patches
`torch.distributed.init_process_group` so two `ModelRunner` instances
(thinker + talker) can co-exist in one Python process. The fork assumes
one runner per process and unconditionally calls `nccl`-init; the patch
makes the second call a no-op when a process group is already initialised.
This is unrelated to performance — it's the fork-multi-runner adaptation.

## Module map

| Local module | Responsibility | vllm-omni relationship |
|---|---|---|
| `config/registry.py` | Lazy model registry, pipeline/deploy registry, tuple stage factories | Small local counterpart to vllm-omni's model and pipeline registries |
| `config/params.py` | `OmniEngineArgs`, aligned `SamplingParams` | Consumer-facing subset, not the fork's sampling class |
| `outputs.py` | `OmniRequestOutput` and modality artifacts | Consumer-facing output envelope |
| `entrypoints/base.py` | Model and pipeline resolution; lazy executor setup | Local `OmniBase` counterpart |
| `entrypoints/omni.py` | Synchronous `Omni.generate` | Local public entry point |
| `entrypoints/async_omni.py` | Async submission and streaming wrapper | Local async entry point |
| `engine/runner.py` | Sequential stage execution and deploy merge | Single-replica in-process runner |
| `engine/executor.py` | Thread-pool and semaphore wrapper | Async transport around the runner |
| `engine/stage_runner.py` | Fork-AR adapter (`StageRunner`, `get_stage_config`, `_ensure_dist`) — composes fork `ModelRunner` + `Sampler` and exposes the fork-attention lifecycle as discrete methods | New — fork has no local equivalent |
| `models/minimind_omni/bundle.py` | Bundle loader: HF model + Mimi codec + tokenizer into one `MinimindBundle` | MiniMind-O only |
| `models/minimind_omni/thinker.py` | `MiniMindThinker`: 12-layer AR stack on fork layers; bridge hidden state capture into a registered buffer | MiniMind-O only |
| `models/minimind_omni/talker.py` | `MiniMindTalker`: 4-layer AR stack on fork layers; 8 codebook heads for audio frame generation | MiniMind-O only |
| `models/minimind_omni/code2wav.py` | Mimi codec decode + WAV envelope | MiniMind-O only |
| `models/minimind_omni/pipeline.py` | Three-stage `PipelineConfig` declaration | MiniMind-O only |
| `models/minimind_omni/stage.py` | `ThinkerStage` / `TalkerStage` — high-level stage classes invoked by the factory | MiniMind-O only |
| `models/minimind_omni/stage_runner.py` | MiniMind-O decode recipe (`decode_minimind`, `recapture_bridge_minimind`) composed on `engine.stage_runner.StageRunner` | MiniMind-O only |
| `models/minimind_omni/stage_processors.py` | `ThinkerStageOutput` / `TalkerInputPayload` / `Code2WavInputPayload` — frozen dataclasses + pure-function converters | MiniMind-O only |
| `models/minimind_omni/hf_config.py` | `MiniMindOConfig` — the repo's own config class for `model_type: minimind-o`; registered so the hub's `auto_map` file never runs | MiniMind-O only |
| `models/smolvlm/{smolvlm.py, stage.py, pipeline.py}` | Complete SmolVLM (`SigLIP` vision tower + fork-style `SmolLM2` text stack + connector), prefill/decode stage, single-stage `LLM_AR` pipeline | SmolVLM only |
| `models/sd_turbo/{stage.py, pipeline.py}` | SD-Turbo 1-step image generation as a `DIFFUSION`-kind stage implementing the 4-method contract (`prepare_encode` / `denoise_step` / `step_scheduler` / `post_decode`) | SD-Turbo only |
| `models/smolvla/{vlm_stage.py, action_stage.py, stage_processors.py, pipeline.py, policy.py}` | Two-stage VLA: `vlm` (`LLM_AR`) → bridge payload → `action` (`DIFFUSION`, flow-matching Euler over the action expert) | SmolVLA only |
| `diffusion/interface.py` | `DiffusionPipeline` Protocol / `StepState` / `DiffusionOutput` — 4-method contract for DIFFUSION-kind stages | New — fork has no equivalent |
| `diffusion/runner.py` | `DiffusionRunner` — drives the 4-method loop (`prepare` / `denoise_step` / `step_scheduler` / `post_decode`, exposed via `run_sync`); stdlib `deque` scheduler inline | New |
| `utils/profiling.py` | `profile_range(name)` context manager (`torch.profiler.record_function` + `nvtx.range`) | New |
| `serving/openai_adapter.py` | Stdlib OpenAI-shaped HTTP adapter | Small serving seam, not FastAPI |

There is no top-level `optim/` package in the current tree. The
fusion-style optimisations (merged QKV / gate-up, fused RMSNorm, fused
RoPE, paged-KV attention, CUDA-Graph capture) live in `nanovllm.layers.*`
as proper `nn.Module` subclasses, not as runtime monkey-patches on HF
model classes. Performance evolution history is recorded in
`docs/perf/tk005-rtx3050.md`.

## Request flow

For a normal synchronous request:

1. `Omni.generate(...)` materializes prompts and sampling overrides.
2. `Omni._one(...)` resolves the executor and converts a multimodal prompt
   dictionary into text plus `SamplingParams.extra`.
3. `PipelineExecutor._runner.run(...)` lazily constructs the configured stage
   instances. For stages with `model_architecture`, the runner resolves the
   model class from `OmniModelRegistry` and injects it into the stage wrapper.
4. `PipelineRunner.run(...)` applies each stage's optional `process_input`,
   merges deploy defaults with request sampling, calls the stage instance in
   pipeline order, and adapts terminal output by `final_output_type`.
5. `OmniRequestOutput.from_pipeline(...)` wraps the terminal stage result.

For MiniMind-O the stage order is:

```text
prompt
  -> thinker: token generation and audio-code sequence
  -> talker: hidden-state bridge and MTP audio refinement
  -> code2wav: Mimi audio-code decode and WAV envelope
  -> OmniRequestOutput
```

For SmolVLA the stage order is:

```text
prompt + rgb_obs
  -> vlm (LLM_AR): prefill-only observation encode, no incremental decode
  -> action (DIFFUSION): Euler-integrated flow matching for the action chunk
  -> OmniRequestOutput
```

Inside each AR stage the loop is:

```text
fork ModelRunner.run_model (input_ids, positions, is_prefill)
  -> fork Attention with module-level Context (slot_mapping, block_tables, ...)
  -> Sampler (top-k / Gumbel-max with optional repetition_penalty)
  -> record_token, advance positions, repeat
```

The `engine.stage_runner.StageRunner` adapter wraps the fork
`ModelRunner` + `Sampler` and exposes the fork-attention lifecycle as
discrete methods (`set_context` / `forward` / `sample` /
`reset_context`) so model families stay side-effect free at import
time.

## Deliberate divergences

- Single process and single local replica; no distributed stage pool or
  tensor-parallel fork engine.
- Thread-pool async wrapper around synchronous stages.
- Small consumer-facing `SamplingParams` subset; it is not a re-export of the
  fork's class.
- Stdlib `http.server` instead of FastAPI/uvicorn.
- Diffusion stages are driven by a local `DiffusionRunner`, not the fork —
  the fork has no diffusion path.
- `SmolVLA`'s vlm stage deliberately does not run the fork `ModelRunner`
  decode loop (one-shot prefill only); the action expert is reached
  through lerobot's internal `policy.model.vlm_with_expert.forward`. See
  `docs/dev/nanovllm-omni-smolvla-arflow-migration.md` §3 ADR-029 for
  the rationale.

## Reading order

1. `entrypoints/omni.py` and `entrypoints/base.py`
2. `engine/executor.py` and `engine/runner.py`
3. `engine/stage_runner.py` — the fork-AR adapter
4. `config/registry.py` and `config/params.py`
5. `models/minimind_omni/pipeline.py`
6. `models/minimind_omni/bundle.py`, `thinker.py`, `talker.py`, `code2wav.py`,
   `stage.py`, `stage_runner.py`, `stage_processors.py`, `hf_config.py`
7. `models/sd_turbo/{stage,pipeline}.py` and `diffusion/{interface,runner}.py`
8. `models/smolvlm/{smolvlm,stage,pipeline}.py`
9. `models/smolvla/{vlm_stage,action_stage,stage_processors,pipeline,policy}.py`
10. `outputs.py` and `serving/openai_adapter.py`

## Out of scope

This repository does not implement distributed execution, tensor parallelism,
video generation, a general-purpose text-only vLLM engine, WebSockets,
streaming HTTP responses, or FastAPI/uvicorn serving.
