"""MiniMind-O pipeline topology (frozen).

Stage 0: thinker  — kind="ar"     (text multimodal understanding + generation)
Stage 1: talker   — kind="ar"     (thinker hidden state -> Mimi codec codes)
Stage 2: code2wav — kind="codec"  (Mimi codec codes -> 24 kHz mono PCM)

For TICKET 02, the three stage factories are a happy-path glue layer that
invokes the existing ``generate_audio`` end-to-end wrapper. The
post-EOS state machine, talker watchdog, and bridge hidden-state
conversion are deferred to TICKET 05. Field set / topology shape are
final and match vllm-omni's PR #3796 layout.

Stage factory functions live in the per-stage modules (``thinker.py``,
``talker.py``, ``code2wav.py``) -- this file assembles the topology only.
"""

from __future__ import annotations

from nanovllm_omni.config_registry import PipelineConfig, StageConfig, register_pipeline

from .code2wav import _code2wav_stage
from .talker import _identity_process_input, _talker_stage
from .thinker import _thinker_stage

MINIMIND_OMNI_PIPELINE = PipelineConfig(
    name="minimind_o",
    stages=(
        StageConfig(
            stage_id=0,
            name="thinker",
            kind="ar",
            factory=_thinker_stage,
            process_input=None,
            input_sources=(),
        ),
        StageConfig(
            stage_id=1,
            name="talker",
            kind="ar",
            factory=_talker_stage,
            process_input=_identity_process_input,
            input_sources=(0,),
        ),
        StageConfig(
            stage_id=2,
            name="code2wav",
            kind="codec",
            factory=_code2wav_stage,
            process_input=_identity_process_input,
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
