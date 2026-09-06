"""MiniMind-O pipeline topology (frozen, fully declarative).

Stage 0: thinker  — kind=LLM_AR    (text multimodal understanding + generation)
Stage 1: talker   — kind=LLM_AR    (thinker hidden state -> Mimi codec codes)
Stage 2: code2wav — kind=CODEC     (Mimi codec codes -> 24 kHz mono PCM)

This file contains NO static imports from per-stage modules: factory and
process_input are dotted-path strings resolved by
``nanovllm_omni.config.registry.resolve_stage_factory`` at construction
(``StageConfig.__post_init__``). The pipeline topology file is the
declarative contract; per-stage implementation lives in
``thinker.py`` / ``talker.py`` / ``code2wav.py``.

The thinker factory emits a ``ThinkerStageOutput`` with bridge hidden
states, ``thinker2talker`` turns that into a ``TalkerInputPayload``, the
talker stage drives the talker wrapper + ``talker_mtp`` over the bridge to
produce code rows, ``talker2code2wav`` turns them into a
``Code2WavInputPayload``, and the code2wav stage decodes them to 24 kHz
mono WAV.

Full is the only supported pipeline kind: the three-stage path executes
end-to-end and was validated against real MiniMind-3o / Mimi weights on
RTX 3050 (see ``tests/test_full_pipeline.py``). The legacy collapsed
(single-thinker) pipeline is retired.
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
    # The three-stage thinker -> talker -> code2wav path is the only runtime
    # mode; the legacy collapsed pipeline is retired.
    supported_pipeline_kinds=("full",),
)

PIPELINE = MINIMIND_OMNI_PIPELINE


register_pipeline(MINIMIND_OMNI_PIPELINE)
