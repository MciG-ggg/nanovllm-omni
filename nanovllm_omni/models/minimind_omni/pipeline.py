"""MiniMind-O pipeline topology: thinker -> talker -> code2wav.

Stage factories are dotted-path strings resolved at construction;
per-stage implementation lives in ``thinker.py`` / ``talker.py`` /
``code2wav.py``. Public symbol: ``MINIMIND_OMNI_PIPELINE``.
"""

from __future__ import annotations

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_MINIMIND_OMNI_FAMILY = "nanovllm_omni.models.minimind_omni"

MINIMIND_OMNI_PIPELINE = PipelineConfig(
    name="minimind_o",
    stages=(
        StageConfig(
            stage_id=0,
            name="thinker",
            kind=StageExecutionType.LLM_AR,
            factory=f"{_MINIMIND_OMNI_FAMILY}.thinker:_thinker_stage",
            process_input=None,
            input_sources=(),
        ),
        StageConfig(
            stage_id=1,
            name="talker",
            kind=StageExecutionType.LLM_AR,
            factory=f"{_MINIMIND_OMNI_FAMILY}.talker:_talker_stage",
            process_input=f"{_MINIMIND_OMNI_FAMILY}.stage_processors:thinker2talker",
            input_sources=(0,),
        ),
        StageConfig(
            stage_id=2,
            name="code2wav",
            kind=StageExecutionType.CODEC,
            factory=f"{_MINIMIND_OMNI_FAMILY}.code2wav:_code2wav_stage",
            process_input=f"{_MINIMIND_OMNI_FAMILY}.stage_processors:talker2code2wav",
            input_sources=(1,),
            is_terminal=True,
            final_output_type="audio",
        ),
    ),
    default_deploy_config_name="minimind_omni.yaml",
    registration_handles=("minimind_o", "jingyaogong/minimind-3o"),
    supported_pipeline_kinds=("full",),
)

PIPELINE = MINIMIND_OMNI_PIPELINE


register_pipeline(MINIMIND_OMNI_PIPELINE)
