# nanovllm-omni ↔ vllm-omni compatibility surface

This document records **consumer-visible symbols whose names happen to align
with vllm-omni**, plus the operational caveats a vllm-omni consumer should
know before reusing code on nanovllm-omni. It is *not* an internal-
implementation contract — every public symbol below is re-exported from the
nanovllm-omni source so the names match, not because nanovllm-omni ships a
replacement for vllm-omni.

> **Read this before porting code from vllm-omni.** nanovllm-omni is a
> single-card, single-process runtime aimed at learning and small demos;
> it is *not* trying to be a drop-in replacement for vllm-omni. The names
> line up; the operational guarantees are deliberately narrower.

## Symbols aligned (consumer-visible)

| Symbol | Source-of-truth | Notes |
|---|---|---|
| `nanovllm_omni.Omni` | `nanovllm_omni.entrypoints.Omni` | Sync entry point; `generate(...) -> list[OmniRequestOutput]`. |
| `nanovllm_omni.AsyncOmni` | `nanovllm_omni.entrypoints.AsyncOmni` | Async entry point; `generate(...) -> AsyncIterator[OmniRequestOutput]`. |
| `nanovllm_omni.OmniBase` | `nanovllm_omni.entrypoints.OmniBase` | Shared constructor + lazy engine setup for sync / async. |
| `nanovllm_omni.OmniEngineArgs` | `nanovllm_omni.config.params.OmniEngineArgs` | kwargs→config dataclass; field set per the `OmniEngineArgs` field matrix in `docs/aligned_interfaces.md` (only some fields are live; the rest are no-op schema slots). |
| `nanovllm_omni.SamplingParams` | `nanovllm_omni.config.params.SamplingParams` | `frozen=True` dataclass mirroring vllm-omni's surface (`temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `n`, `extra`). |
| `nanovllm_omni.OmniRequestOutput` | `nanovllm_omni.outputs.OmniRequestOutput` | `request_id`, `outputs`, `multimodal_output` (`Mapping[str, Any]`), `error`, `final_output_type`, `images`, `latents`, `custom_output`, `to_dict()`, `from_pipeline()` / `from_diffusion()` / `from_stage_output()` / `from_error()`, `is_pipeline_output` / `is_diffusion_output`. See `docs/contracts/multimodal_output.md` for the locked artifact types. |
| `nanovllm_omni.config.OMNI_PIPELINES` | registry dict | Same role as vllm-omni's `pipeline_registry.OMNI_PIPELINES`; values may be `PipelineConfig` or a `Callable[[Any], PipelineConfig \| None]` resolver. |
| `nanovllm_omni.config.resolve_pipeline_config` | registry helper | Lookup by `name` or `registration_handles` alias; callable resolvers are invoked with `hf_config`. |
| `nanovllm_omni.config.register_pipeline` | registry writer | Register a `PipelineConfig` (or a callable resolver under an explicit `model_type`); callable resolvers may also register `registration_handles` aliases. |

## Operational caveats

### Scope is narrower

nanovllm-omni ships:

- **4 model families**: MiniMind-O (audio, 3-stage), SD-Turbo (image, 1-stage DIFFUSION), SmolVLM-500M-Instruct (text, 1-stage LLM_GENERATION), SmolVLA (actions, 1-stage LLM_GENERATION).
- **1 hardware target**: a single CUDA device with the model in one process. `engine/runner.py` + `engine/orchestrator.py` are deliberately in-process; there is no subprocess `StagePool`, no janus queue, no Mooncake, no Ray.
- **1 HTTP seam**: `python -m nanovllm_omni.serving.openai_adapter` implements `POST /v1/chat/completions` over stdlib `http.server` (no FastAPI / uvicorn / Pydantic). No `/v1/duplex`, no `/v1/realtime`, no WebSocket endpoints.
- **1 ASGI-free async surface**: `AsyncOmni` runs the sync `PipelineRunner` under `asyncio.run_in_executor`; `extra["max_concurrent"]` (default 1) gates concurrent requests.

### Things that *look* like vllm-omni but are not

| Symbol / behaviour | Status |
|---|---|
| `StagePool`, `Orchestrator` with background threads | **deliberately omitted.** nanovllm-omni's `engine/orchestrator.py` keeps the *shape* (request → replica pool → LB → drive to completion) in-process; it does not run a background thread, does not own janus queues, and does not implement distributed membership. |
| `StageRuntime`, lazy per-stage VRAM unload | **not implemented.** README's "Single-card, single-process runtime" section locks the bundle loader as the source-of-truth: MiniMind-O's thinker + talker + code2wav share one `AutoModelForCausalLM` in a single `MinimindBundle`. There is no stage subprocess to lazy-load. |
| Tensor-parallel, pipeline-parallel, multi-GPU dispatch | **not implemented.** `tensor_parallel_size=1` is the only mode; `AGENTS.md` locks single-GPU scope. |
| Per-stage `gpu_memory_utilization` / `max_num_seqs` | **no-op fields.** They exist on `OmniEngineArgs` for schema parity with vllm-omni but nothing reads them. See the field-effect matrix in `docs/aligned_interfaces.md`. |
| Continuous-batching scheduler (TK-004) | **implemented in single-process form.** `engine/runtime_scheduler.py` (`RuntimeScheduler`) + `engine/sequence.py` (`Sequence`, `PrefillChunk`) + `engine/kv_pool.py` (`FixedKvSlotPool`). One scheduler instance per LLM stage; no PagedAttention, no prefix cache, no speculative decoding. |
| Stage replica + RoundRobin LB (TK-007) | **minimal demo.** `StageConfig.num_replicas` + `engine/load_balancer.py` (`RoundRobinBalancer`). No dead-replica detection, no dynamic scaling, no per-replica metrics. |
| Streaming chunks (`AsyncIterator[OmniRequestOutput]`) | **not implemented.** `py_generator=True` on `Omni.generate` returns a sync `Generator` that yields per-prompt results in submission order; that is the closest substitute. `AsyncOmni.generate` returns an `AsyncIterator`, but each yielded value is the complete result of one request, not a token-level stream. |
| FastAPI / uvicorn serving | **explicitly out of scope** (`AGENTS.md`). The HTTP adapter uses stdlib `http.server.BaseHTTPRequestHandler` + `ThreadingHTTPServer`. |
| PagedAttention, prefix caching, KV transfer | **explicitly out of scope.** TK-004 ticket's scope notes. |
| Plugin entry points (`vllm.general_plugins`) | **not implemented.** Registration is the `register_pipeline()` Python API. |

### vllm-omni release cadence

vllm-omni ships a major release roughly every 2 months; its public surface
(modules, dataclass fields, HTTP shape) evolves with each release.
nanovllm-omni tracks the surface that was inspected when the alignment
work landed and does not auto-track upstream. Concretely:

- The contract captured in `docs/aligned_interfaces.md` reflects vllm-omni
  as it was when the file was last updated; the next upstream release may
  have moved a field, renamed a class, or added a `*_metadata` sidecar
  that nanovllm-omni does not emit.
- Re-running the alignment audit (`grep` on `nanovllm_omni/` vs the
  vllm-omni read-only reference at `/Users/mcig/Projects/vllm-omni`) is
  the recommended way to spot a drift; this project does not run that
  audit on a schedule.
- Compatibility breaks in nanovllm-omni are explicit and ticket-tracked;
  they will surface as `OmniRequestOutput` field changes, `OmniEngineArgs`
  kwarg additions/removals, or `register_pipeline` signature changes —
  never silently.

### Reusing code across the two engines

A consumer that targets both engines should:

1. Stay on the **consumer-visible** symbols above. Anything else
   (`StagePool`, `Orchestrator`, `StageRuntime`, ...`) is single-engine
   and will not port.
2. Treat `multimodal_output` as `Mapping[str, Any]` at runtime; the
   closed set of artifact types (`AudioPayload`, `ActionArtifact`,
   `ImageArtifact`, `TextArtifact` per
   `docs/contracts/multimodal_output.md`) is the contract, but a stage
   may emit a raw `PIL.Image` / `str` for backward compatibility, and
   `to_dict()` handles both.
3. Read `OmniEngineArgs` field-effect matrix in
   `docs/aligned_interfaces.md` before assuming a vllm-omni kwarg does
   anything. No-op fields are silent; the matrix is the truth.
4. Run a smoke (`Omni("model_id").generate(["hi"], SamplingParams(max_tokens=8))`)
   on the target hardware before assuming the consumer code will run.
   The runtime contract is "real weights required, offline-first";
   the bundle loader never auto-fetches.

## How to verify the compat surface

```bash
# Public-API import smoke (CI / pre-commit gate):
python -c "from nanovllm_omni import (
    Omni, AsyncOmni, OmniBase, SamplingParams, OmniEngineArgs, OmniRequestOutput,
)"

# Alignment doc reference (single source of truth for symbols + fields):
cat docs/aligned_interfaces.md

# Output contract:
cat docs/contracts/multimodal_output.md

# Field-effect matrix (which OmniEngineArgs kwarg actually does anything):
grep -E '\| `?(enforce_eager|gpu_memory_utilization|max_num_seqs|max_num_batched_tokens|tensor_parallel_size)' \
    docs/aligned_interfaces.md
```

## What this document is *not*

- It is *not* an implementation contract for nanovllm-omni's internals
  (`PipelineRunner`, `PipelineExecutor`, `Orchestrator`, `LoadBalancer`,
  ...). Those are single-engine and live in
  `nanovllm_omni/engine/`; consult that package for behavior.
- It is *not* a substitute for `docs/aligned_interfaces.md`, which is the
  field-by-field truth. This file is the *operational* layer.
- It is *not* a promise of vllm-omni feature parity. The list of
  deliberately omitted features is in
  `docs/aligned_interfaces.md` "对齐边界" / "deliberately omitted"
  section.