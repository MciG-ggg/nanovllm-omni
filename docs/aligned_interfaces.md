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
