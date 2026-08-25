"""SmolVLM-500M-Instruct pipeline topology (frozen, declarative).

Single terminal LLM_GENERATION stage: ``transformers.AutoModelForVision2Seq``
loads ``HuggingFaceTB/SmolVLM-500M-Instruct`` (BF16, image+text -> text).
``kind=LLM_GENERATION`` is the closest existing StageExecutionType; the
vllm-omni taxonomy has no pure-VLM member and the closed-set test locks
the four names. ``final_output_type="text"`` so the engine wraps the
generated string via ``OmniRequestOutput.from_pipeline(...)``.
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
            kind=StageExecutionType.LLM_GENERATION,
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

PIPELINE = SMOLVLM_PIPELINE

register_pipeline(SMOLVLM_PIPELINE)
