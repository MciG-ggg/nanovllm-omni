"""Sana-0.6B diffusion pipeline topology (frozen, declarative).

Single terminal DIFFUSION stage: the diffusers ``SanaPipeline`` is one
checkpoint (Gemma-2-2B text encoder + SanaTransformer2DModel + AutoencoderDC);
text prompt -> 1024x1024 image. This is the minimal-diffusion demo (TK-009):
one stage, no bridging. ``kind=DIFFUSION`` is an existing closed-set member
(no contract change).
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_SANA_FAMILY = "nanovllm_omni.models.sana_06b"

SANA_06B_PIPELINE = PipelineConfig(
    name="sana_06b",
    stages=(
        StageConfig(
            stage_id=0,
            name="sana",
            kind=StageExecutionType.DIFFUSION,
            factory=f"{_SANA_FAMILY}.stage:_sana_stage",
            process_input=None,
            input_sources=(),
            is_terminal=True,
            final_output_type="image",
        ),
    ),
    default_deploy_config_name="sana_06b.yaml",
    registration_handles=(
        "sana_06b",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    ),
    # ``hf_architectures`` is the layer-6 disambiguator in
    # ``OmniBase.try_infer_model_type``. diffusers indexes pipelines by
    # ``model_index.json._class_name``; matching that lets a local snapshot
    # route here when the surrounding repo name does not contain the family.
    hf_architectures=("SanaPipeline",),
)

PIPELINE = SANA_06B_PIPELINE

register_pipeline(SANA_06B_PIPELINE)
