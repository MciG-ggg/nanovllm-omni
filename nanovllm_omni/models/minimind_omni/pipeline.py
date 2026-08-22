"""MiniMind-O pipeline topology (frozen, fully declarative).

Stage 0: thinker  — kind=LLM_AR    (text multimodal understanding + generation)
Stage 1: talker   — kind=LLM_AR    (thinker hidden state -> Mimi codec codes)
Stage 2: code2wav — kind=CODEC     (Mimi codec codes -> 24 kHz mono PCM)

This file contains NO static imports from per-stage modules: factory and
process_input are dotted-path strings resolved by
``nanovllm_omni.config_registry.resolve_stage_factory`` at construction
(``StageConfig.__post_init__``). The pipeline topology file is the
declarative contract; per-stage implementation lives in
``thinker.py`` / ``talker.py`` / ``code2wav.py``.

For TICKET-02, the three stage factories are a happy-path glue layer that
invokes the existing ``generate_audio`` end-to-end wrapper via the
thinker. The post-EOS state machine, talker watchdog, and bridge
hidden-state conversion are deferred to TICKET-05; field set / topology
shape match vllm-omni's pipeline-registry pattern.
"""

from __future__ import annotations

from nanovllm_omni.config_registry import (
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
            process_input=f"{_MINIMIND_OMNI_FAMILY}.talker:_identity_process_input",
            input_sources=(0,),
        ),
        StageConfig(
            stage_id=2,
            name="code2wav",
            kind=StageExecutionType.CODEC,
            factory=f"{_MINIMIND_OMNI_FAMILY}.code2wav:_code2wav_stage",
            process_input=f"{_MINIMIND_OMNI_FAMILY}.talker:_identity_process_input",
            input_sources=(1,),
            is_terminal=True,
            final_output_type="audio",
        ),
    ),
    default_deploy_config_name="minimind_omni.yaml",
    registration_handles=("minimind_o", "jingyaogong/minimind-3o"),
)

PIPELINE = MINIMIND_OMNI_PIPELINE


register_pipeline(MINIMIND_OMNI_PIPELINE)
