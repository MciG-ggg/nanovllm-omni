"""Public synchronous submission operation for the model-free pipeline.

Issue #3 layers the MiniMind-Omni post-EOS state machine on top of the
issue #2 tracer bullet. Each request owns a ``PerRequestPostEOSState``
that walks ``pending -> visible_eos -> forced_padding -> downstream_ready
-> done`` while the orchestrator observes the Thinker's bridge stream.
State is registered with the orchestrator on entry and cleared in a
``finally`` block, so concurrent or failing requests cannot leak.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from uuid import uuid4

from nanovllm_omni.payloads import (
    THINKER_FORCED_PADDING_DEFAULT,
    AudioPayload,
    BridgePayload,
    ThinkerRun,
)
from nanovllm_omni.pipeline import Pipeline


class _PostEOSPhase(StrEnum):
    """Phases of the per-request Thinker post-EOS state machine."""

    PENDING = "pending"
    VISIBLE_EOS = "visible_eos"
    FORCED_PADDING = "forced_padding"
    DOWNSTREAM_READY = "downstream_ready"
    DONE = "done"


@dataclass
class PerRequestPostEOSState:
    """Per-request Thinker post-EOS state machine.

    Implements ``pending -> visible_eos -> forced_padding -> downstream_ready
    -> done``. The orchestrator creates one instance per request, feeds it
    the Thinker bridges in order, and clears it on completion, failure, or
    cancellation. ``forced_padding_count`` defaults to the configured
    reference value and is updated once the Thinker's actual count is
    observed; until then it acts as a placeholder.
    """

    request_id: str
    forced_padding_count: int = THINKER_FORCED_PADDING_DEFAULT
    bridges: list[BridgePayload] = field(default_factory=list)
    phase: _PostEOSPhase = _PostEOSPhase.PENDING

    def observe_bridge(self, bridge: BridgePayload) -> None:
        """Advance the state machine using one bridge state from the Thinker."""
        if self.phase == _PostEOSPhase.PENDING:
            # First bridge is the visible step; it must end in EOS by contract.
            self.bridges.append(bridge)
            self.phase = _PostEOSPhase.VISIBLE_EOS
            if self.forced_padding_count <= 0:
                self.phase = _PostEOSPhase.DOWNSTREAM_READY
            else:
                self.phase = _PostEOSPhase.FORCED_PADDING
        elif self.phase == _PostEOSPhase.FORCED_PADDING:
            self.bridges.append(bridge)
            if len(self.bridges) - 1 >= self.forced_padding_count:
                self.phase = _PostEOSPhase.DOWNSTREAM_READY
        else:
            raise ValueError(f"cannot observe bridge in phase {self.phase}")

    def mark_done(self) -> None:
        """Finalize the state machine once downstream readiness is reached."""
        if self.phase != _PostEOSPhase.DOWNSTREAM_READY:
            raise ValueError(f"cannot mark done in phase {self.phase}")
        self.phase = _PostEOSPhase.DONE


@dataclass(frozen=True)
class PipelineResult:
    """Final output and observable stage order for one submitted request."""

    audio: AudioPayload
    stage_names: tuple[str, ...]
    elapsed_seconds: float


class Orchestrator:
    """Own request entrypoints; asynchronous scheduling is intentionally deferred."""

    def __init__(self) -> None:
        # Tracks every in-flight request's post-EOS state so concurrent
        # submits cannot bleed into one another and so cleanup is
        # observable from tests without inspecting private counters.
        self._active_states: dict[str, PerRequestPostEOSState] = {}

    def active_request_count(self) -> int:
        """Return the number of in-flight requests (testing/observability)."""
        return len(self._active_states)

    def submit(self, pipeline: Pipeline, request: str) -> PipelineResult:
        """Run a request through ``pipeline`` and return its typed final output.

        Registers a per-request ``PerRequestPostEOSState`` on entry and
        clears it in a ``finally`` block, so the post-EOS counters and
        bridge states are isolated across concurrent requests and removed
        after completion, failure, or cancellation.
        """
        request_id = uuid4().hex
        state = PerRequestPostEOSState(
            request_id=request_id,
            forced_padding_count=THINKER_FORCED_PADDING_DEFAULT,
        )
        self._active_states[request_id] = state
        try:
            started = monotonic()
            audio, trace = pipeline.run(request)
            thinker_output = trace[0][1]
            if not isinstance(thinker_output, ThinkerRun):
                raise TypeError("first pipeline stage must emit a ThinkerRun")
            # Replace the placeholder forced_padding_count with the Thinker's
            # actual value, then drive the state machine through the bridge
            # stream the Thinker produced.
            state.forced_padding_count = thinker_output.forced_padding_count
            state.bridges.clear()
            state.phase = _PostEOSPhase.PENDING
            for bridge in thinker_output.bridges:
                state.observe_bridge(bridge)
            state.mark_done()
            return PipelineResult(
                audio=audio,
                stage_names=tuple(name for name, _ in trace),
                elapsed_seconds=monotonic() - started,
            )
        finally:
            self._active_states.pop(request_id, None)
