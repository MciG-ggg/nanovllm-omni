"""Public configuration registry exports.

Thin facade over the config subsystem. Implementation lives in the sibling
leaf modules ``registry.py`` (model/pipeline/deploy dataclasses + yaml loading +
lazy tuple-based stage resolution) and ``params.py`` (``SamplingParams`` /
``OmniEngineArgs``). Keep this file leaf-only: importing engine, models, or
entrypoints here would reintroduce the import cycle this layout avoids.
"""

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.config.registry import (
    OMNI_MODELS,
    OMNI_PIPELINES,
    DeployConfig,
    DeployStageConfig,
    OmniModelRegistry,
    PipelineConfig,
    StageConfig,
    StageFactoryRegistration,
    load_deploy_config,
    merge_pipeline_deploy,
    register_pipeline,
    resolve_pipeline_config,
)

__all__ = [
    "OMNI_MODELS",
    "OMNI_PIPELINES",
    "OmniModelRegistry",
    "PipelineConfig",
    "StageConfig",
    "StageFactoryRegistration",
    "DeployConfig",
    "DeployStageConfig",
    "register_pipeline",
    "resolve_pipeline_config",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "OmniEngineArgs",
    "SamplingParams",
]
