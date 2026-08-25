"""Mimir-1.6B-Instruct pipeline wiring (no real weights / lcm / sonar).

Mirrors ``test_sd_turbo.py`` shape: registry resolution, stage topology,
import-guard on the heavy deps, and ``OmniRequestOutput.from_pipeline``
final-output-type plumbing. The real-weight smoke is run via
``examples/offline_inference/mimir_1_6b/run.py`` and tagged ``smoke``
in pytest; default ``-m "not smoke"`` skips it.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from nanovllm_omni.config import resolve_pipeline_config
from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import (
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from nanovllm_omni.models.mimir_1_6b import mimir_lcm, prompt_encoder, text_decoder
from nanovllm_omni.outputs import OmniRequestOutput

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "mimir_1_6b.yaml"


def test_pipeline_registry_resolves_mimir_1_6b():
    config = resolve_pipeline_config("mimir_1_6b")
    assert config is not None
    assert config.name == "mimir_1_6b"
    assert [s.name for s in config.stages] == ["prompt_encoder", "mimir_lcm", "text_decoder"]
    kinds = [s.kind for s in config.stages]
    assert kinds == [
        StageExecutionType.CODEC,
        StageExecutionType.DIFFUSION,
        StageExecutionType.CODEC,
    ]
    assert config.stages[-1].is_terminal is True
    assert config.stages[-1].final_output_type == "text"
    assert config.default_deploy_config_name == "mimir_1_6b.yaml"


def test_pipeline_registry_resolves_hf_handle():
    for handle in ("mimir-lcm/Mimir-1.6B-Instruct", "mimir_1_6b_instruct"):
        config = resolve_pipeline_config(handle)
        assert config is not None
        assert config.name == "mimir_1_6b"


def test_process_input_bridges_resolve_to_callables():
    """Each ``process_input`` must be a live callable on the family
    module -- the registry's ``StageConfig.__post_init__`` already
    enforced this at registration time, but we double-check the named
    symbols actually exist for the bridge contract."""
    assert callable(getattr(prompt_encoder, "_embeddings_to_batch", None))
    assert callable(getattr(mimir_lcm, "_generator_output_to_embeddings", None))


def test_from_pipeline_text_key():
    """Terminal stage returns text via from_pipeline(final_output_type="text")."""
    text = "hello"
    output = OmniRequestOutput.from_pipeline(text, final_output_type="text")
    assert output.multimodal_output is not None
    assert output.multimodal_output["text"] == text


def test_stages_require_heavy_deps(monkeypatch):
    """All three stage factories must fail cleanly with ImportError when
    ``lcm`` / ``sonar`` / ``wtpsplit`` are missing, instead of
    poisoning the registry at import time. We monkeypatch ``__import__``
    to raise on the heavy deps; the call to ``_prompt_encoder_stage``
    triggers the import inside the factory."""
    real_import = builtins.__import__
    heavy = {
        "lcm",
        "sonar",
        "sonar.inference_pipelines.text",
        "wtpsplit",
        "fairseq2",
    }

    def fake_import(name, *args, **kwargs):
        if name in heavy or name.split(".", 1)[0] in {"lcm", "sonar", "wtpsplit", "fairseq2"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="No module named"):
        prompt_encoder._prompt_encoder_stage(None, OmniEngineArgs(model="mimir_1_6b"))
    with pytest.raises(ImportError, match="No module named"):
        mimir_lcm._mimir_lcm_stage(None, OmniEngineArgs(model="mimir_1_6b"))
    with pytest.raises(ImportError, match="No module named"):
        text_decoder._text_decoder_stage(None, OmniEngineArgs(model="mimir_1_6b"))


def test_deploy_defaults_per_stage():
    deploy = load_deploy_config(DEPLOY)
    merged = merge_pipeline_deploy(resolve_pipeline_config("mimir_1_6b"), deploy)
    _, enc_defaults = merged[0]
    _, lcm_defaults = merged[1]
    _, dec_defaults = merged[2]
    assert enc_defaults["sat_threshold"] == 0.02
    assert lcm_defaults["inference_timesteps"] == 40
    assert lcm_defaults["guidance_scale"] == 1.5
    assert dec_defaults == {}


def test_try_infer_model_type_basename():
    from nanovllm_omni.entrypoints.base import try_infer_model_type

    # The bare handle matches itself (no other handle is a superstring).
    assert try_infer_model_type("mimir_1_6b") == "mimir_1_6b"
    # A local snapshot dir's basename ``Mimir-1.6B-Instruct`` normalizes
    # to ``mimir16binstruct`` and matches the longest registered alias
    # ``mimir_1_6b_instruct`` (which points at the same pipeline).
    assert try_infer_model_type("/checkpoints/Mimir-1.6B-Instruct") == "mimir_1_6b_instruct"
    from nanovllm_omni.config.registry import resolve_pipeline_config

    assert (
        resolve_pipeline_config(try_infer_model_type("/checkpoints/Mimir-1.6B-Instruct")).name
        == "mimir_1_6b"
    )
