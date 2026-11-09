"""Backend seam: one AR execution path per array framework.

The scheduler layer is already framework-free. ``engine/stage_engine.py``
and ``engine/runner.py`` import no torch at all, and so do the fork's
``engine/scheduler.py`` / ``engine/sequence.py`` / ``engine/block_manager.py``
-- batch formation, slot reuse, preemption and chunked prefill all live
in plain Python.

What is *not* framework-free is the physical step: ``prepare_*`` builds
the batch tensors, ``forward`` runs the model, ``sample`` draws a token.
Those three are CUDA-only today, because the fork's attention kernel is.

This package names that seam. A backend supplies the device, the dtype,
what shapes it can batch, and a ``StageRunner`` that runs one step.

See ``docs/adr/0013-backend-abstraction.md`` for why the seam is drawn
here and not one level up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do. Stages read this instead of guessing."""

    varlen: bool
    """Can one forward carry ragged sequences?

    True means the backend builds ``cu_seqlens`` + a paged KV cache and
    runs a variable-length attention kernel, so a batch of different
    lengths costs no padding. False means the backend pads to the longest
    member of the batch.
    """

    graph: bool
    """Has a graph-capture equivalent.

    CUDA Graph on CUDA, ``mx.compile`` on MLX. False means every step
    re-dispatches its ops, which is the dominant cost on the small models
    this project runs (ADR-0007).
    """

    max_batch: int | None = None
    """How many sequences one forward may carry; ``None`` means no limit."""


@runtime_checkable
class StageRunner(Protocol):
    """One AR step, framework-free.

    The sequence objects handed in and out are the fork's ``Sequence``:
    plain Python, shared by every backend. Only the values crossing this
    boundary are framework-specific -- batch tensors on the way in,
    logits and token ids on the way out.
    """

    def prepare_prefill(self, seqs: list[Any]) -> tuple[Any, Any]:
        """Build the batch tensors for a prefill step."""

    def prepare_decode(self, seqs: list[Any]) -> tuple[Any, Any]:
        """Build the batch tensors for a decode step."""

    def prepare_sample(self, seqs: list[Any]) -> Any:
        """Build the per-sequence sampling parameters (temperatures)."""

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        is_prefill: bool,
        inputs_embeds: Any | None = None,
    ) -> Any:
        """Run the model for one step; returns logits."""

    def sample(
        self,
        logits: Any,
        temperatures: Any,
        sampler: Any = None,
    ) -> Any:
        """Draw one token per sequence.

        ``sampler`` overrides the backend's own sampler when given. It is
        called as ``sampler(logits, temperatures)`` and must return a
        flat token-id batch. Families that need a vendor sampling recipe
        (repetition penalty, staggered codebooks) pass one instead of
        reaching into the runner.
        """

    def bridge_hidden(self) -> Any:
        """Hidden states the model produced this step, if it produces any.

        MiniMind-O's thinker hands these to the talker stage; families
        without a second stage return ``None``.
        """

    @property
    def model(self) -> Any:
        """The model object behind this runner.

        Families that run a non-AR path over the same weights reach for
        it directly -- SmolVLM's vision tower and connector.
        """


@runtime_checkable
class Backend(Protocol):
    """A device plus the way AR steps run on it."""

    name: str
    device: str
    dtype: str
    capabilities: BackendCapabilities

    def make_config(self, model_path: str, **kwargs: Any) -> Any:
        """Build the backend's per-stage config object.

        The torch backend returns the fork's ``Config``; the MLX backend
        returns its own. Stage code asks the backend instead of importing
        a framework's config class.
        """

    def make_stage_runner(self, model_class: type, config: Any) -> StageRunner:
        """Build the runner that executes AR steps for one stage."""


__all__ = ["Backend", "BackendCapabilities", "StageRunner"]
