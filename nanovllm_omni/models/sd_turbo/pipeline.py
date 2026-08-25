"""SD-Turbo diffusion pipeline topology (frozen, declarative).

Single terminal DIFFUSION stage: ``diffusers.StableDiffusionPipeline``
loads ``stabilityai/sd-turbo`` (adversarially-distilled SD 2.1, 1-step
inference). text prompt -> 512x512 image. This is the minimal-diffusion
demo (TK-009 rev2): one stage, no bridging. ``kind=DIFFUSION`` is an
existing closed-set member (no contract change).
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_SD_TURBO_FAMILY = "nanovllm_omni.models.sd_turbo"

SD_TURBO_PIPELINE = PipelineConfig(
    name="sd_turbo",
    stages=(
        StageConfig(
            stage_id=0,
            name="sd_turbo",
            kind=StageExecutionType.DIFFUSION,
            factory=f"{_SD_TURBO_FAMILY}.stage:_sd_turbo_stage",
            process_input=None,
            input_sources=(),
            is_terminal=True,
            final_output_type="image",
        ),
    ),
    default_deploy_config_name="sd_turbo.yaml",
    registration_handles=(
        "sd_turbo",
        "stabilityai/sd-turbo",
    ),
    # ``hf_architectures`` is the layer-6 disambiguator in
    # ``OmniBase.try_infer_model_type``. diffusers indexes pipelines by
    # ``model_index.json._class_name``; SD-Turbo ships as the standard
    # ``StableDiffusionPipeline`` class.
    hf_architectures=("StableDiffusionPipeline",),
)

PIPELINE = SD_TURBO_PIPELINE

register_pipeline(SD_TURBO_PIPELINE)
