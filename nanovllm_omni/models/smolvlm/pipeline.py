"""SmolVLM-500M-Instruct pipeline topology (frozen, declarative).

Single terminal ``LLM_AR`` stage: vision prefill (HF SigLIP + GeLU
connector) then fork paged-attention AR decode loop on the SmolLM2
text decoder. ``kind=LLM_AR`` (per ADR-018) replaces the prior
``LLM_GENERATION`` placeholder; the stage is genuinely autoregressive
now that it rides the fork ``StageRunner``. ``final_output_type="text"``
so the engine wraps the generated string via
``OmniRequestOutput.from_pipeline(...)``.
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_SMOLVLM_FAMILY = "nanovllm_omni.models.smolvlm"

SMOLVLM_PIPELINE = PipelineConfig(
    name="smolvlm",
    stages=(
        StageConfig(
            stage_id=0,
            name="vlm",
            kind=StageExecutionType.LLM_AR,
            factory=f"{_SMOLVLM_FAMILY}.stage:_vlm_stage",
            process_input=None,
            input_sources=(),
            is_terminal=True,
            final_output_type="text",
        ),
    ),
    default_deploy_config_name="smolvlm.yaml",
    registration_handles=(
        "smolvlm",
        "HuggingFaceTB/SmolVLM-500M-Instruct",
    ),
    # ``hf_architectures`` is the layer-6 disambiguator in
    # ``OmniBase.try_infer_model_type``. transformers resolves the
    # ``AutoModelForVision2Seq`` class to ``SmolVLMForConditionalGeneration``
    # when ``config.json.architectures`` is that name; matching here lets a
    # local snapshot route back to ``smolvlm`` even when the basename is
    # not the HF repo id.
    hf_architectures=("SmolVLMForConditionalGeneration",),
)

register_pipeline(SMOLVLM_PIPELINE)
