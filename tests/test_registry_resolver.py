"""Test: pipeline registry parallelism with vllm-omni.

Covers the two vllm-omni registry shapes we now mirror:

  - ``OMNI_PIPELINES`` values may be a ``PipelineConfig`` (unchanged) OR a
    callable resolver ``(hf_config) -> PipelineConfig | None``.
  - ``resolve_pipeline_config(model_type, hf_config=None)``: when the mapping
    holds a callable it is invoked with ``hf_config``; a ``None`` result
    means "no pipeline for that config", which lets one key select among
    variants.
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import replace

import pytest

from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.registry import (
    OMNI_PIPELINES,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    register_pipeline,
)


def _mini_pipeline(name: str) -> PipelineConfig:
    return PipelineConfig(
        name=name,
        stages=(
            StageConfig(
                stage_id=0,
                name="stage",
                kind=StageExecutionType.LLM_AR,
                factory="nanovllm_omni.models.smolvla.stage:_vla_stage",
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="minimind_omni.yaml",
    )


class _FakeConfig:
    def __init__(self, model_type: str = "base"):
        self.model_type = model_type


def test_registry_value_may_be_a_callable_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_pipeline accepts a callable; resolve invokes it with hf_config."""
    calls: list[str] = []

    def resolver(hf_config: object | None) -> PipelineConfig | None:
        calls.append(getattr(hf_config, "model_type", "none"))
        if hf_config is not None and getattr(hf_config, "model_type", "") == "variant":
            return _mini_pipeline("variant")
        return None

    with monkeypatch.context() as m:
        m.setitem(OMNI_PIPELINES, "resolver_only", resolver)
        result = resolve_pipeline_config("resolver_only", hf_config=_FakeConfig("variant"))
    assert calls == ["variant"]
    assert isinstance(result, PipelineConfig)
    assert result.name == "variant"


def test_resolver_returning_none_means_no_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolver that returns None is 'no match' -- registry lookup yields None."""

    def resolver(hf_config: object | None) -> PipelineConfig | None:
        return None

    with monkeypatch.context() as m:
        m.setitem(OMNI_PIPELINES, "empty", resolver)
        assert resolve_pipeline_config("empty", hf_config=_FakeConfig("x")) is None


def test_resolve_still_accepts_plain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain PipelineConfig values are untouched; hf_config is ignored."""
    cfg = _mini_pipeline("plain")
    with monkeypatch.context() as m:
        m.setitem(OMNI_PIPELINES, "plain", cfg)
        resolved = resolve_pipeline_config("plain", hf_config=_FakeConfig("x"))
    assert resolved is cfg  # identity: not wrapped, not invoked


def test_register_pipeline_rejects_non_pipeline_and_non_callable() -> None:
    with pytest.raises(TypeError):
        register_pipeline(42)  # type: ignore[arg-type]


def test_single_variants_keep_existing_behavior() -> None:
    """Existing family lookups (minimind_o / smolvla) still resolve to configs."""
    for key in ("minimind_o", "smolvla", "HuggingFaceVLA/smolvla_libero"):
        assert isinstance(resolve_pipeline_config(key), PipelineConfig)


def test_unknown_key_returns_none() -> None:
    assert resolve_pipeline_config("not_a_pipeline") is None


def test_kwarg_param_name_is_model_type() -> None:
    """Aligned-surface contract: the first parameter is named ``model_type``
    (matches vllm-omni's keyword call surface), not ``name``."""
    # Behavioural proof: keyword name both works and resolves.
    assert isinstance(resolve_pipeline_config(model_type="minimind_o"), PipelineConfig)
    # Structural proof: the parameter is literally called model_type.
    params = inspect.signature(resolve_pipeline_config).parameters
    assert list(params)[0] == "model_type"


def test_register_pipeline_keys_by_name_and_clobbers_silently() -> None:
    """Lock a documented in-scope divergence from vllm-omni.

    The reference keys ``PipelineConfig`` by ``model_type`` and runs
    validate + warn on duplicate keys; nanovllm-omni keys by ``name``
    and silently overwrites a same-name registration. AGENTS.md
    ("Definition of aligned") treats this as in-scope, so this test
    pins the actual behavior to keep the docs honest.
    """
    name = f"drift-lock-{uuid.uuid4().hex[:8]}"
    first = _mini_pipeline(name)
    second = replace(first, default_deploy_config_name="other.yaml")
    assert second is not first
    try:
        register_pipeline(first)
        assert OMNI_PIPELINES[name] is first
        register_pipeline(second)  # documented: silent overwrite
        assert OMNI_PIPELINES[name] is second
    finally:
        OMNI_PIPELINES.pop(name, None)
