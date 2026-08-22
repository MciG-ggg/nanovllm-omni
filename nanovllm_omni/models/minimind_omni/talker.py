"""MiniMind-O talker stage.

Stage 1 of the 3-stage pipeline. For TICKET-02 this is an identity
pass-through because the thinker's end-to-end ``generate_audio`` already
emits the audio payload. TICKET-05 (``.scratch/aligned-interfaces/issues/
05-minimind-stage-split``) will replace the body with real talker
forwarding + post-EOS state machine + watchdog.
"""

from __future__ import annotations

from typing import Any


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: identity pass-through for TICKET-02.

    In TICKET-05 this will consume the thinker's bridge hidden state and
    emit Mimi codec codes. For now the thinker's end-to-end output already
    includes the audio, so the talker is a no-op.
    """

    def talker_forward(payload: Any, sampling: Any) -> Any:
        return payload

    return talker_forward


def _identity_process_input(payload: Any, prompt: str) -> Any:
    """Default process_input: pass the previous stage's output through unchanged.

    Used by TICKET-02's happy-path glue layer. TICKET-05 will replace this
    with real bridge hidden-state conversion.
    """
    return payload


__all__ = ["_identity_process_input", "_talker_stage"]
