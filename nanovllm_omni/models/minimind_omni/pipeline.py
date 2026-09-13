"""MiniMind-O pipeline topology: thinker -> talker -> code2wav.

The resolver selects the full or thinker-only topology from HF config;
per-stage implementation lives in ``thinker.py`` / ``talker.py`` /
``code2wav.py``. Public symbol: ``MINIMIND_OMNI_PIPELINE``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanovllm_omni.config.registry import (
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)

_MINIMIND_OMNI_FAMILY = "nanovllm_omni.models.minimind_omni"
_THINKER_MODEL = "MiniMindThinker"
_TALKER_MODEL = "MiniMindTalker"
_THINKER_FACTORY = (_MINIMIND_OMNI_FAMILY + ".thinker", "_thinker_stage")
_TALKER_FACTORY = (_MINIMIND_OMNI_FAMILY + ".talker", "_talker_stage")
_CODE2WAV_FACTORY = (_MINIMIND_OMNI_FAMILY + ".code2wav", "_code2wav_stage")
_PROCESSOR_MODULE = _MINIMIND_OMNI_FAMILY + ".stage_processors"

MINIMIND_OMNI_PIPELINE = PipelineConfig(
    name="minimind_o",
    stages=(
        StageConfig(
            stage_id=0,
            name="thinker",
            kind=StageExecutionType.LLM_AR,
            stage_factory=_THINKER_FACTORY,
            model_architecture=_THINKER_MODEL,
            process_input=None,
            input_sources=(),
        ),
        StageConfig(
            stage_id=1,
            name="talker",
            kind=StageExecutionType.LLM_AR,
            stage_factory=_TALKER_FACTORY,
            model_architecture=_TALKER_MODEL,
            process_input=(_PROCESSOR_MODULE, "thinker2talker"),
            input_sources=(0,),
        ),
        StageConfig(
            stage_id=2,
            name="code2wav",
            kind=StageExecutionType.CODEC,
            stage_factory=_CODE2WAV_FACTORY,
            process_input=(_PROCESSOR_MODULE, "talker2code2wav"),
            input_sources=(1,),
            is_terminal=True,
            final_output_type="audio",
        ),
    ),
    default_deploy_config_name="minimind_omni.yaml",
    registration_handles=("minimind_o", "jingyaogong/minimind-3o"),
    supported_pipeline_kinds=("full",),
)

MINIMIND_OMNI_THINKER_ONLY_PIPELINE = PipelineConfig(
    name="minimind_o_thinker_only",
    stages=(
        StageConfig(
            stage_id=0,
            name="thinker",
            kind=StageExecutionType.LLM_AR,
            stage_factory=_THINKER_FACTORY,
            model_architecture=_THINKER_MODEL,
            is_terminal=True,
            final_output_type="text",
        ),
    ),
    default_deploy_config_name="minimind_omni.yaml",
    registration_handles=("minimind_o_thinker_only",),
    supported_pipeline_kinds=("thinker_only",),
)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_minimind_pipeline(hf_config: Any) -> PipelineConfig:
    """Select MiniMind full or thinker-only topology from HF config."""
    if hf_config is not None and not bool(_config_value(hf_config, "enable_audio_output", True)):
        return MINIMIND_OMNI_THINKER_ONLY_PIPELINE
    return MINIMIND_OMNI_PIPELINE


PIPELINE = MINIMIND_OMNI_PIPELINE


register_pipeline(
    resolve_minimind_pipeline,
    model_type="minimind_o",
    registration_handles=("jingyaogong/minimind-3o",),
)
