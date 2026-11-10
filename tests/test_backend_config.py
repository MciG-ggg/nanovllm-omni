"""The backend must be pickable from the public argument surface.

It used to be reachable only through ``extra={"backend": ...}``, an untyped
catch-all: a typo surfaced inside a stage constructor, after the pipeline was
half built. The declared knob validates the name at the argument boundary,
and the deploy YAML can override it per stage.
"""

from __future__ import annotations

import pytest

from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import DeployStageConfig, load_deploy_config


def test_the_default_is_each_stage_s_own_choice() -> None:
    assert OmniEngineArgs("x").backend is None


@pytest.mark.parametrize("name", ["torch", "mlx", "mlx-vlm"])
def test_a_registered_backend_is_accepted(name: str) -> None:
    assert OmniEngineArgs("x", backend=name).backend == name


def test_an_unknown_backend_fails_at_the_argument_boundary() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        OmniEngineArgs("x", backend="mlx_vlm")


def test_the_deploy_yaml_carries_a_backend(tmp_path) -> None:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "stages:\n"
        "  - name: vlm\n"
        "    backend: mlx-vlm\n"
        "    default_sampling_params:\n"
        "      max_new_tokens: 32\n"
    )

    stage = load_deploy_config(path).stages[0]

    assert stage.backend == "mlx-vlm"
    assert stage.default_sampling_params == {"max_new_tokens": 32}


def test_a_deploy_stage_without_a_backend_leaves_it_open() -> None:
    assert DeployStageConfig(name="vlm").backend is None


def test_a_non_string_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be a string"):
        DeployStageConfig(name="vlm", backend=1)  # type: ignore[arg-type]


def test_the_declared_backend_wins_over_the_extra_escape_hatch() -> None:
    """``args.backend`` is validated; ``extra["backend"]`` is not, so it loses."""
    from nanovllm_omni.models.minimind_omni.stage import _stage_backend

    args = OmniEngineArgs("x", backend="mlx")

    assert _stage_backend(args, {"backend": "torch"}) == "mlx"
    assert _stage_backend(OmniEngineArgs("x"), {"backend": "mlx-vlm"}) == "mlx-vlm"
    assert _stage_backend(OmniEngineArgs("x"), {}) == "torch"
