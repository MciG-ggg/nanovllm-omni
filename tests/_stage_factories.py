"""Module-level test factory functions for ``StageConfig`` string-path tests.

After Phase 2 (TK-016), ``StageConfig.factory`` is a dotted-path string
(``"package.module:attr"``) instead of a direct callable. Tests that
construct a ``StageConfig`` therefore need module-level factory functions
that can be resolved by ``resolve_stage_factory``.

Conventions:
  - Generic, stateless factories live alongside this docstring
    (``thinker_simple``, ``talker_simple``, ``code2wav_simple``,
    ``executor_simple``, ``diffusion_dit``).
  - Per-test state observation uses a shared module-level slot keyed by
    string. Tests that share a slot MUST call its reset helper at the top
    of the test body so the captured state is fresh. Pytest runs tests
    sequentially by default, so the slot pattern is sufficient.
  - The ``identity_process_input`` and ``bridge_process_input`` are
    reusable dotted-path targets for ``StageConfig.process_input``.
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


def executor_simple(deploy: Any, args: Any) -> Any:
    """Single-stage async-executor stub: ``f"out({payload})"``."""

    def forward(payload: Any, sampling: Any) -> Any:
        return f"out({payload})"

    return forward


# ---------------------------------------------------------------------------
# Generic process_input hooks (dotted-path targets)
# ---------------------------------------------------------------------------


def identity_process_input(payload: Any, prompt: str) -> Any:
    """Identity bridge hook used by ``StageConfig.process_input``."""
    return payload


def bridge_process_input(payload: Any, prompt: str) -> Any:
    """Bridge hook: prepends ``"br|"`` to ``payload`` for tests that
    want to observe that ``process_input`` ran between stages."""
    return f"br|{payload}"


# ---------------------------------------------------------------------------
# Capturing slot: shared single state holder for ``capturing_simple``
# ---------------------------------------------------------------------------

_captures: list[Any] = []
_factory_observations: list[tuple[Any, Any]] = []


def reset_factory_observations() -> None:
    _factory_observations.clear()


def get_factory_observations() -> list[tuple[Any, Any]]:
    return list(_factory_observations)


def observing_factory(deploy: Any, args: Any) -> Any:
    _factory_observations.append((deploy, args))

    def forward(payload: Any, sampling: Any) -> Any:
        return payload

    return forward


def reset_captures() -> None:
    """Reset the capturing slot before a test runs."""
    _captures.clear()


def get_captures() -> list[Any]:
    """Snapshot the capturing slot after a test runs."""
    return list(_captures)


def capturing_simple(deploy: Any, args: Any) -> Any:
    """Capturing factory: appends ``sampling`` to a shared list.

    Tests that want to observe the ``SamplingParams`` seen by the runner
    call ``reset_captures()`` first and ``get_captures()`` after.
    """

    def forward(payload: Any, sampling: Any) -> Any:
        _captures.append(sampling)
        return payload

    return forward


# ---------------------------------------------------------------------------
# Logged slot: shared per-stage order log for the multi-stage runner test
# ---------------------------------------------------------------------------

_log: list[tuple[str, Any, Any]] = []


def reset_log() -> None:
    _log.clear()


def get_log() -> list[tuple[str, Any, Any]]:
    return list(_log)


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


# ---------------------------------------------------------------------------
# Diffusion slot: shared state for the single-stage diffusion test
# ---------------------------------------------------------------------------

_diffusion_log: list[tuple[str, Any, Any]] = []


def reset_diffusion_log() -> None:
    _diffusion_log.clear()


def get_diffusion_log() -> list[tuple[str, Any, Any]]:
    return list(_diffusion_log)


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
    """Diffusion (single-stage) factory: returns a logged pipeline."""
    return _LoggedDiffusionPipeline(_diffusion_log)
