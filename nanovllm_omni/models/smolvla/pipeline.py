"""SmolVLA pipeline topology (frozen, fully declarative).

Single terminal stage: the LeRobot SmolVLA policy is one checkpoint
(SigLIP + SmolVLM + action expert). Splitting vision / language / action
into separate engine stages needs TensorHandle and is not
faked here.

kind=LLM_GENERATION -- closest existing StageExecutionType; the vllm-omni
taxonomy has no VLA member and the closed-set test locks the four names.
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_SMOLVLA_FAMILY = "nanovllm_omni.models.smolvla"

SMOLVLA_PIPELINE = PipelineConfig(
    name="smolvla",
    stages=(
        StageConfig(
            stage_id=0,
            name="vla",
            kind=StageExecutionType.LLM_GENERATION,
            factory=f"{_SMOLVLA_FAMILY}.stage:_vla_stage",
            process_input=None,
            input_sources=(),
            is_terminal=True,
            final_output_type="actions",
        ),
    ),
    default_deploy_config_name="smolvla.yaml",
    registration_handles=(
        "smolvla",
        "HuggingFaceVLA/smolvla_libero",
        "lerobot/smolvla_base",
    ),
    # ``hf_architectures`` is the layer-6 disambiguator in
    # ``OmniBase.try_infer_model_type``. LeRobot names the policy class
    # ``SmolVLAPolicy`` in its modeling file; matching on it lets a local
    # snapshot route here even when ``config.json`` is malformed.
    hf_architectures=("SmolVLAPolicy",),
)

PIPELINE = SMOLVLA_PIPELINE

register_pipeline(SMOLVLA_PIPELINE)
