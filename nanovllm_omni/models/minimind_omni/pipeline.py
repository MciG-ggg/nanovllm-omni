"""MiniMind-O pipeline topology (frozen).

Stage 0: thinker  — kind="ar"     (text multimodal understanding + generation)
Stage 1: talker   — kind="ar"     (thinker hidden state -> Mimi codec codes)
Stage 2: code2wav — kind="codec"  (Mimi codec codes -> 24 kHz mono PCM)

For TICKET 02, the three stage factories below are a happy-path glue layer
that invokes the existing ``generate_audio`` end-to-end wrapper. The
post-EOS state machine, talker watchdog, and bridge hidden-state
conversion are deferred to TICKET 05. Field set / topology shape are
final and match vllm-omni's PR #3796 layout.
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.config_registry import PipelineConfig, StageConfig, register_pipeline


def _identity_process_input(payload: Any, prompt: str) -> Any:
    """Default process_input: pass the previous stage's output through unchanged.

    Used by TICKET 02's happy-path glue layer. TICKET 05 will replace this
    with real bridge hidden-state conversion.
    """
    return payload


def _thinker_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: returns a callable that runs the thinker.

    For TICKET 02, the "thinker" invokes the entire end-to-end pipeline via
    ``generate_audio`` so that the field topology is exercised without
    requiring the 3-stage split (TICKET 05).
    """
    from nanovllm_omni.models.minimind_omni.stages import create_bundle, generate_audio

    extra_args = dict(getattr(args, "extra", None) or {})
    mimi_model_id = extra_args.pop("mimi_model_id", None) or extra_args.pop("mimi", None)
    bundle_kwargs: dict[str, Any] = {}
    if mimi_model_id:
        bundle_kwargs["mimi_model_id"] = mimi_model_id
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    def thinker_forward(payload: Any, sampling: Any) -> Any:
        prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
        extra = (
            (sampling.extra or {}) if sampling is not None and hasattr(sampling, "extra") else {}
        )
        return generate_audio(
            bundle,
            prompt,
            max_tokens=int(sampling.max_tokens) if sampling is not None else 16,
            temperature=float(sampling.temperature) if sampling is not None else 0.7,
            top_p=float(sampling.top_p) if sampling is not None else 1.0,
            open_thinking=bool(extra.get("open_thinking", False)),
        )

    return thinker_forward


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: identity pass-through for TICKET 02.

    In TICKET 05 this will consume the thinker's bridge hidden state and
    emit Mimi codec codes. For now the thinker's end-to-end output already
    includes the audio, so the talker is a no-op.
    """

    def talker_forward(payload: Any, sampling: Any) -> Any:
        return payload

    return talker_forward


def _code2wav_stage(deploy: Any, args: Any) -> Any:
    """Stage 2 factory: identity pass-through for TICKET 02.

    The end-to-end ``generate_audio`` already returns ``AudioPayload``; the
    codec decode happened inside the thinker glue layer.
    """

    def code2wav_forward(payload: Any, sampling: Any) -> Any:
        return payload

    return code2wav_forward


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
