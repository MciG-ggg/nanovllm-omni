# Aligned interfaces

The public nanovllm-omni API follows the corresponding vllm-omni interfaces where practical.

| nanovllm-omni class | vllm-omni counterpart |
|---|---|
| `nanovllm_omni.entrypoints.Omni` | `vllm_omni.entrypoints.Omni` |
| `nanovllm_omni.entrypoints.AsyncOmni` | `vllm_omni.entrypoints.AsyncOmni` |
| `nanovllm_omni.engine_args.SamplingParams` | `vllm_omni.engine_args.SamplingParams` |
| `nanovllm_omni.engine_args.OmniEngineArgs` | `vllm_omni.engine_args.OmniEngineArgs` |
| `nanovllm_omni.outputs.OmniRequestOutput` | `vllm_omni.outputs.RequestOutput` |
| `nanovllm_omni.config_registry.PipelineConfig` | `vllm_omni.config.PipelineConfig` |
| `nanovllm_omni.config_registry.DeployConfig` | `vllm_omni.config.DeployConfig` |
| `nanovllm_omni.config_registry.DeployStageConfig` | `vllm_omni.config.DeployStageConfig` |

## Alignment boundary

This is consumer-visible interface alignment, not an identical implementation. The supported vertical slice is MiniMind-O audio only. It intentionally differs from vllm-omni by omitting diffusion and other modalities, distributed execution, duplex S2S, WebSockets, FastAPI/uvicorn, Pydantic, and plugin entry points. These omissions are deliberate scope boundaries, not compatibility bugs.

When an aligned public symbol or response field changes, update this table, add a focused contract test, and run the CI-equivalent checks documented in `AGENTS.md`.
