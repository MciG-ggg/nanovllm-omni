"""SmolVLA pipeline topology (frozen, fully declarative).

ADR-026: split into two stages so AR and flow-matching execution modes
are visible at the pipeline level:

| stage | name   | kind         | what runs                       |
|-------|--------|--------------|---------------------------------|
| 0     | vlm    | LLM_AR       | SigLIP + SmolVLM backbone       |
| 1     | action | DIFFUSION    | flow-matching action expert     |

ADR-031: legacy single-stage path stays available via deploy YAML toggle
``use_split_stages: false`` (default) until numerical alignment passes.
When ``use_split_stages: true``, the two-stage pipeline is registered
under the same handles.

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

# Legacy single-stage pipeline (kept for backward compatibility).
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
    hf_architectures=("SmolVLAPolicy",),
)


def _smolvla_split_resolver(hf_config: object | None) -> PipelineConfig | None:
    """Resolver for the split-stages pipeline (ADR-031).

    Returns the two-stage pipeline. Kept separate from the legacy
    PipelineConfig so callers can choose via deploy YAML.
    """
    return SMOLVLA_SPLIT_PIPELINE


# Two-stage split pipeline (ADR-026, ADR-031).
SMOLVLA_SPLIT_PIPELINE = PipelineConfig(
    name="smolvla_split",
    stages=(
        StageConfig(
            stage_id=0,
            name="vlm",
            kind=StageExecutionType.LLM_AR,
            factory=f"{_SMOLVLA_FAMILY}.vlm_stage:_vlm_stage",
            process_input=None,
            input_sources=(),
            is_terminal=False,
            final_output_type=None,
        ),
        StageConfig(
            stage_id=1,
            name="action",
            kind=StageExecutionType.DIFFUSION,
            factory=f"{_SMOLVLA_FAMILY}.action_stage:_action_stage",
            process_input=f"{_SMOLVLA_FAMILY}.stage_processors:vlm2action",
            input_sources=(0,),
            is_terminal=True,
            final_output_type="actions",
        ),
    ),
    default_deploy_config_name="smolvla.yaml",
    registration_handles=("smolvla_split",),
    hf_architectures=("SmolVLAPolicy",),
)


register_pipeline(SMOLVLA_PIPELINE)
# Register split pipeline under a dedicated handle; callers opt in via
# ``extra["pipeline"] = "smolvla_split"`` (ADR-031).
register_pipeline(SMOLVLA_SPLIT_PIPELINE)
