"""Module-level test factory functions for tuple-based StageConfig tests.

``StageConfig.stage_factory`` and ``process_input`` use module/attribute
tuples instead of direct callables. Tests therefore keep module-level
factory functions that can be resolved by ``resolve_stage_factory``.

Conventions:
  - Generic, stateless factories live alongside this docstring
    (``thinker_simple``, ``talker_simple``, ``code2wav_simple``,
    ``executor_simple``, ``diffusion_dit``).
  - Per-test state observation uses a shared module-level slot keyed by
    string. Tests that share a slot MUST call its reset helper at the top
    of the test body so the captured state is fresh. Pytest runs tests
    sequentially by default, so the slot pattern is sufficient.
  - The ``identity_process_input`` and ``bridge_process_input`` are
    reusable tuple-registration targets for ``StageConfig.process_input``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Generic (no-state) stage factories
# ---------------------------------------------------------------------------


def thinker_simple(deploy: Any, args: Any) -> Any:
    """Stage 0 stub: returns ``f"th|{payload}"``."""

    def forward(payload: Any, sampling: Any) -> Any:
        return f"th|{payload}"

    return forward


def talker_simple(deploy: Any, args: Any) -> Any:
    """Stage 1 stub: returns ``f"tk|{payload}"``."""

    def forward(payload: Any, sampling: Any) -> Any:
        return f"tk|{payload}"

    return forward


def code2wav_simple(deploy: Any, args: Any) -> Any:
    """Stage 2 stub: identity pass-through (same shape as the real CODEC)."""

    def forward(payload: Any, sampling: Any) -> Any:
        return payload

    return forward


class FakeRegisteredModel:
    """Small class used to prove model-registry injection without torch."""


class TextTerminal:
    text = "terminal text"


def model_aware_factory(deploy: Any, args: Any, model_class: Any = None) -> Any:
    """Factory that exposes the class supplied by the model registry."""

    def forward(payload: Any, sampling: Any) -> Any:
        return model_class

    return forward


def text_terminal_factory(deploy: Any, args: Any) -> Any:
    """Factory returning a wrapper-shaped text terminal output."""

    def forward(payload: Any, sampling: Any) -> Any:
        return TextTerminal()

    return forward


def executor_simple(deploy: Any, args: Any) -> Any:
    """Single-stage async-executor stub: ``f"out({payload})"``."""

    def forward(payload: Any, sampling: Any) -> Any:
        return f"out({payload})"

    return forward


# ---------------------------------------------------------------------------
# Generic process_input hooks (tuple-registration targets)
# ---------------------------------------------------------------------------


def identity_process_input(payload: Any, prompt: str) -> Any:
    """Identity bridge hook used by ``StageConfig.process_input``."""
    return payload


def bridge_process_input(payload: Any, prompt: str) -> Any:
    """Bridge hook: prepends ``"br|"`` to ``payload`` for tests that
    want to observe that ``process_input`` ran between stages."""
    return f"br|{payload}"


# ---------------------------------------------------------------------------
# Capture slots: shared state holders for the tuple-registered factories below.
# Each slot is ``(list, reset, snapshot)`` -- the factories write to the list,
# tests call reset/snapshot around the operation under test.
# ---------------------------------------------------------------------------


def _make_slot():
    """Build one capture slot: a backing list plus paired reset/snapshot closures."""
    items: list = []

    def reset() -> None:
        items.clear()

    def snapshot() -> list:
        return list(items)

    return items, reset, snapshot


_captures, reset_captures, get_captures = _make_slot()
_factory_observations, reset_factory_observations, get_factory_observations = _make_slot()
_log, reset_log, get_log = _make_slot()
_diffusion_log, reset_diffusion_log, get_diffusion_log = _make_slot()


def observing_factory(deploy: Any, args: Any) -> Any:
    _factory_observations.append((deploy, args))

    def forward(payload: Any, sampling: Any) -> Any:
        return payload

    return forward


def capturing_simple(deploy: Any, args: Any) -> Any:
    """Capturing factory: appends ``sampling`` to a shared list.

    Tests that want to observe the ``SamplingParams`` seen by the runner
    call ``reset_captures()`` first and ``get_captures()`` after.
    """

    def forward(payload: Any, sampling: Any) -> Any:
        _captures.append(sampling)
        return payload

    return forward


def logged_thinker(deploy: Any, args: Any) -> Any:
    def forward(payload: Any, sampling: Any) -> Any:
        _log.append(("thinker", payload, sampling))
        return f"{payload}->thinker"

    return forward


def logged_talker(deploy: Any, args: Any) -> Any:
    def forward(payload: Any, sampling: Any) -> Any:
        _log.append(("talker", payload, sampling))
        return f"{payload}->talker"

    return forward


def logged_code2wav(deploy: Any, args: Any) -> Any:
    def forward(payload: Any, sampling: Any) -> Any:
        _log.append(("code2wav", payload, sampling))
        return f"{payload}->code2wav"

    return forward


class _LoggedDiffusionPipeline:
    """Fake DiffusionPipeline that logs calls and returns a constant."""

    supports_step_execution = True

    def __init__(self, log_target: list) -> None:
        self._log = log_target

    def prepare_encode(self, request: Any) -> Any:
        from nanovllm_omni.diffusion.interface import StepState

        return StepState(
            request_id="test",
            latents=0.0,
            encoder_hidden_states=None,
        )

    def denoise_step(self, state: Any, *, step: int, num_steps: int) -> Any:
        return 0.1

    def step_scheduler(self, state: Any, noise_pred: Any) -> None:
        state.latents = state.latents + noise_pred

    def post_decode(self, state: Any) -> Any:
        from nanovllm_omni.diffusion.interface import DiffusionOutput

        self._log.append(("dit", state.latents, None))
        return DiffusionOutput(images=["video"], finished=True)


def logged_diffusion(deploy: Any, args: Any) -> Any:
    """Diffusion (single-stage) factory: returns a ``DiffusionRunner``."""
    from nanovllm_omni.diffusion.runner import DiffusionRunner

    return DiffusionRunner(_LoggedDiffusionPipeline(_diffusion_log))
