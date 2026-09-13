"""SmolVLA pipeline topology (frozen, fully declarative).

Two stages so AR and flow-matching execution modes are visible at the
pipeline level:

| stage | name   | kind         | what runs                       |
|-------|--------|--------------|---------------------------------|
| 0     | vlm    | LLM_AR       | SigLIP + SmolVLM backbone       |
| 1     | action | DIFFUSION    | flow-matching action expert     |

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md``.
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_SMOLVLA_FAMILY = "nanovllm_omni.models.smolvla"
_VLM_FACTORY = (_SMOLVLA_FAMILY + ".vlm_stage", "_vlm_stage")
_ACTION_FACTORY = (_SMOLVLA_FAMILY + ".action_stage", "_action_stage")
_PROCESSOR_MODULE = _SMOLVLA_FAMILY + ".stage_processors"

SMOLVLA_PIPELINE = PipelineConfig(
    name="smolvla",
    stages=(
        StageConfig(
            stage_id=0,
            name="vlm",
            kind=StageExecutionType.LLM_AR,
            stage_factory=_VLM_FACTORY,
            process_input=None,
            input_sources=(),
            is_terminal=False,
            final_output_type=None,
        ),
        StageConfig(
            stage_id=1,
            name="action",
            kind=StageExecutionType.DIFFUSION,
            stage_factory=_ACTION_FACTORY,
            process_input=(_PROCESSOR_MODULE, "vlm2action"),
            input_sources=(0,),
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
    hf_architectures=("SmolVLAPolicy",),
)

register_pipeline(SMOLVLA_PIPELINE)
