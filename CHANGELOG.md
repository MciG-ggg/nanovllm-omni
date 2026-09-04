# Changelog

All notable changes to `nanovllm-omni` are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project is pre-1.0 and does not yet strictly follow Semantic
Versioning.

## [0.1.0] - 2026-09-04

First public pre-release.

### Added

- Four model families on a single 4 GB consumer card (RTX 3050) in one
  Python process:
  - **MiniMind-O** (`jingyaogong/minimind-3o` + `kyutai/mimi`): three-stage
    Thinker → Talker → Code2Wav audio pipeline, ~340 ms p50 on a single
    prompt.
  - **SmolVLM-500M-Instruct** (`HuggingFaceTB/SmolVLM-500M-Instruct`):
    text/VLM.
  - **SD-Turbo** (`stabilityai/sd-turbo`): 1-step image generation,
    512 × 512 PNG.
  - **SmolVLA-256M** (`HuggingFaceTB/SmolVLA-256M`): action-chunk policy
    on LIBERO datasets.
- Unified `OmniRequestOutput` envelope across audio / image / text /
  action modalities, plus a matching JSON shape for
  `POST /v1/chat/completions` (base64 WAV for audio, base64 PNG for
  images, action arrays for actions).
- Public API mirrors vllm-omni's consumer-visible symbols: `Omni`,
  `AsyncOmni`, `OmniBase`, `SamplingParams`, `OmniEngineArgs`,
  `OmniRequestOutput`, `PipelineConfig`, `DeployConfig`,
  `register_pipeline`, `resolve_pipeline_config`, `load_deploy_config`,
  `merge_pipeline_deploy`.
- StagePool pattern with `num_replicas ≥ 2` and RoundRobin load
  balancing.
- Per-stage continuous batching (TK-004) and per-stage replica + LB
  (TK-007) implemented as in-process data structures
  (`nanovllm_omni/engine/runtime_scheduler.py`,
  `nanovllm_omni/engine/load_balancer.py`), not as subprocess pools.
- Fused QKV / gate-up projections, fused RMSNorm, fused RoPE,
  pre-allocated KV buffer, SDPA decode with `is_causal=True` for
  MiniMind-O on the 4 GB card.
- Offline (`examples/offline_inference/`) and online
  (`examples/online_serving/`) examples for all four families.
- Apache-2.0 license; install via `pip install -e ".[dev,minimind]"`.

### In-scope divergences from vllm-omni

Documented in `docs/ARCHITECTURE.md`; listed here for visibility:

- `PipelineConfig` registered under `name` (vllm-omni uses `model_type`).
- Same-name re-registration **silently overrides** (vllm-omni validates
  + warns). Locked by `tests/test_registry_resolver.py`.
- Single-process runtime; no per-stage subprocess pool, no
  `StageRuntime`, no multi-GPU / tensor-parallel worker.
- HTTP seam uses the Python standard library (`http.server`), not
  FastAPI/uvicorn.

### Out of scope (intentionally not implemented)

- Video generation.
- Multi-model pipelines beyond the four families listed above.
- Distributed execution, multi-card tensor-parallel, pipeline-parallel
  schedulers.
- WebSockets, streaming responses.
- FastAPI/uvicorn serving.

[Unreleased]: https://github.com/MciG-ggg/nanovllm-omni/compare/0.1.0...HEAD
