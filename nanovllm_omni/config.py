
# Aligned vLLM-Omni compatibility exports (kept alongside legacy schema).
from nanovllm_omni.engine_args import OmniEngineArgs, SamplingParams
from nanovllm_omni.config_registry import (
    OMNI_PIPELINES, register_pipeline, resolve_pipeline_config,
    load_deploy_config, merge_pipeline_deploy, DeployConfig, DeployStageConfig,
)
