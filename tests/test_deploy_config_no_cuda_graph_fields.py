"""DeployConfig no longer exposes ``use_*_cuda_graph`` fields.

ADR-0005: comparison code lives in bench/, not in deploy config.
Each stage now picks its own default
(thinker -> graph fails on RTX 3050 with the minimind-3o fork Config;
talker -> n/a, uses fork Context directly;
smolvlm -> graph fails on its shell, hardcodes eager).

Contract locks:

1. ``DeployConfig`` has no ``use_thinker_cuda_graph`` /
   ``use_talker_cuda_graph`` fields.
2. ``load_deploy_config`` is lenient: a YAML carrying the dropped
   keys still loads without error (the loader uses ``data.get(...)``
   for known fields only; extras are silently dropped by the named-kwarg
   ``DeployConfig(...)`` constructor call).
3. A minimal fresh YAML still works, so writing new deploy configs
   does not require the dropped keys.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from nanovllm_omni.config.registry import DeployConfig, load_deploy_config


def test_deploy_config_has_no_cuda_graph_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(DeployConfig)}
    assert "use_thinker_cuda_graph" not in field_names
    assert "use_talker_cuda_graph" not in field_names


def test_load_deploy_config_lenient_on_legacy_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "legacy_minimind_omni.yaml"
    yaml_path.write_text(
        "max_batch: 2\n"
        "use_thinker_cuda_graph: false\n"
        "use_talker_cuda_graph: false\n"
        "post_eos_padding_count: 128\n",
        encoding="utf-8",
    )
    deploy = load_deploy_config(yaml_path)
    assert deploy.max_batch == 2
    assert deploy.post_eos_padding_count == 128


def test_load_deploy_config_minimal_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("max_batch: 4\n", encoding="utf-8")
    deploy = load_deploy_config(yaml_path)
    assert deploy.max_batch == 4


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
