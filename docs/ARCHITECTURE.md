# Architecture: how nanovllm-omni maps to vllm-omni

This document describes the implementation that is actually in this
repository. The project mirrors vllm-omni's consumer-facing pipeline shape;
it does not reuse vllm-omni's engine internals.

## Runtime boundary

`nanovllm-omni` has one in-process pipeline runner. `OmniBase` resolves a
registered `PipelineConfig`, loads deploy defaults, and owns one
`PipelineExecutor`. `PipelineExecutor` wraps a synchronous `PipelineRunner`
with an asyncio semaphore and a thread pool.

The vendored `third_party/nano-vllm` submodule is a separate, text-only
educational engine. It is **not** the engine behind `Omni.generate`.
`nanovllm-omni` uses only these fork pieces:

- `BlockManager` for paged KV block allocation.
- The fork's `Sequence` for paged KV block-table bookkeeping.

The current attention adapter, K/V writes, and decode loop remain local.
When the separate `flash_attn` wheel is installed, the adapter calls its
paged kernel directly; otherwise it uses the torch SDPA fallback.

The fork's `LLMEngine`, `ModelRunner`, `Scheduler`, `Config`, and
`SamplingParams` are not part of the Omni generation path. In particular,
MiniMind-O's Thinker -> Talker -> Code2Wav generation is not a Qwen3
`LLMEngine` workload, so making `OmniBase` inherit `LLMEngine` would only add a
nominal parent class while violating the model and scheduler contracts.

The fork package initializer is lazy. Importing `nanovllm.engine.block_manager`
does not import or construct `LLM`; constructing the fork's `LLM` remains an
explicit operation with its own CUDA and distributed-runtime requirements.
The paged adapter adds the submodule to `sys.path` only when paged KV is first
used and fails clearly if another package has already claimed the `nanovllm`
module name.

## Module map

| Local module | Responsibility | vllm-omni relationship |
|---|---|---|
| `config/registry.py` | Pipeline and deploy registry; dotted-path stage factories | Small local counterpart to pipeline registry and stage config |
| `config/params.py` | `OmniEngineArgs`, aligned `SamplingParams` | Consumer-facing subset, not the fork's sampling class |
| `outputs.py` | `OmniRequestOutput` and modality artifacts | Consumer-facing output envelope |
| `entrypoints/base.py` | Model and pipeline resolution; lazy executor setup | Local `OmniBase` counterpart |
| `entrypoints/omni.py` | Synchronous `Omni.generate` | Local public entry point |
| `entrypoints/async_omni.py` | Async submission and streaming wrapper | Local async entry point |
| `engine/runner.py` | Sequential stage execution and deploy merge | Single-replica in-process runner |
| `engine/executor.py` | Thread-pool and semaphore wrapper | Async transport around the runner |
| `models/minimind_omni/runtime_scheduler.py` | Per-stage continuous-batching state | MiniMind-O model-family helper |
| `models/minimind_omni/attention.py` | Fixed contiguous KV buffer adapter | MiniMind-O graph optimization |
| `models/minimind_omni/paged_attention.py` | Paged KV pool and fork allocator bridge | MiniMind-O paged KV adapter |
| `models/minimind_omni/cuda_graph.py` | Per-position Thinker CUDA Graph decoder | MiniMind-O-specific optimization |
| `models/minimind_omni/paged_cuda_graph.py` | One-window, replay-many decoder | MiniMind-O-specific paged optimization |
| `models/minimind_omni/talker_cuda_graph.py` | Fixed-shape Talker MTP graph cache | MiniMind-O-specific optimization |
| `models/minimind_omni/` | Stages, schedulers, KV helpers, graph decoders, and bundle | Model-family implementation |
| `serving/openai_adapter.py` | Stdlib OpenAI-shaped HTTP adapter | Small serving seam, not FastAPI |

There is no `engine/runtime.py`, `engine/load_balancer.py`, or top-level
`optim/` package in the current tree. The generic `engine/` package only owns
pipeline execution; MiniMind-O CUDA, KV, scheduler, and sequence helpers live
under `models/minimind_omni/`.

## Request flow

For a normal synchronous request:

1. `Omni.generate(...)` materializes prompts and sampling overrides.
2. `Omni._one(...)` resolves the executor and converts a multimodal prompt
   dictionary into text plus `SamplingParams.extra`.
3. `PipelineExecutor._runner.run(...)` lazily constructs the configured stage
   instances.
4. `PipelineRunner.run(...)` applies each stage's optional `process_input`,
   merges deploy defaults with request sampling, and calls the stage instance
   in pipeline order.
5. `OmniRequestOutput.from_pipeline(...)` wraps the terminal stage result.

For MiniMind-O the stage order is:

```text
prompt
  -> thinker: token generation and audio-code sequence
  -> talker: hidden-state bridge and MTP audio refinement
  -> code2wav: Mimi audio-code decode and WAV envelope
  -> OmniRequestOutput
```

The Thinker stage may select eager decoding, the contiguous fixed-KV CUDA
Graph decoder, or the paged single-graph decoder. These are implementation
choices inside the stage; none of them calls fork `LLMEngine.generate`,
`add_request`, or `step`.

## Paged KV bridge

`models.minimind_omni.paged_attention.enable_paged_kv_cache(...)` discovers attention-like
modules by their projection interface, allocates one shared
`[2, layers, blocks, block_size, kv_heads, head_dim]` tensor, and binds each
layer's K/V views to it. A `PagedKVContext` owns fixed-address metadata
(`slot_mapping`, `context_lens`, and `block_tables`) for graph replay.

The fork `BlockManager` allocates block IDs. The local adapter owns the
per-layer pool wiring, context scratch buffers, attention-forward replacement,
and MiniMind-O request lifecycle. The torch-native SDPA path is used when
flash-attn is unavailable; flash-attn remains optional for the macOS test
path and is expected on the WSL CUDA box for its optimized kernel.

The local `OmniSequence` in `models/minimind_omni/sequence.py` is deliberately named
separately from the fork's `nanovllm.engine.sequence.Sequence`. The former is
scheduler state for the local pipeline; the latter is paged-KV block-table
state. They are not interchangeable.

## Deliberate divergences

- Single process and single local replica; no distributed stage pool or
  tensor-parallel fork engine.
- Thread-pool async wrapper around synchronous stages.
- Small consumer-facing `SamplingParams` subset; it is not a re-export of the
  fork's class.
- Stdlib `http.server` instead of FastAPI/uvicorn.
- MiniMind-O uses its Hugging Face model implementation and local stage
  contracts rather than the fork's Qwen3-only model runner.
- CUDA Graph source adaptation is limited to the MiniMind-O forward contract;
  graph optimization is opt-in through deploy configuration and falls back to
  eager execution when unsupported.

## Reading order

1. `entrypoints/omni.py` and `entrypoints/base.py`
2. `engine/executor.py` and `engine/runner.py`
3. `config/registry.py` and `config/params.py`
4. `models/minimind_omni/pipeline.py`
5. `models/minimind_omni/bundle.py`, `thinker.py`, `talker.py`, and `code2wav.py`
6. `models/minimind_omni/paged_attention.py` and
   `models/minimind_omni/paged_cuda_graph.py`
7. `outputs.py` and `serving/openai_adapter.py`

## Out of scope

This repository does not implement distributed execution, tensor parallelism,
video generation, a general-purpose text-only vLLM engine, WebSockets,
streaming HTTP responses, or FastAPI/uvicorn serving.
