"""DeployConfig carries only knobs the engine actually reads.

ADR-0005: comparison code lives in bench/, not in deploy config. Each
stage picks its own graph-vs-eager default.

Contract locks:

1. ``DeployConfig`` has no ``use_*_cuda_graph`` fields.
2. ``load_deploy_config`` is lenient: a YAML carrying dropped or unknown
   keys still loads, so legacy configs keep working.
3. A minimal fresh YAML still works, so writing new deploy configs does
   not require any top-level key.
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
        "post_eos_padding_count: 128\n"
        "stages:\n"
        "  - name: thinker\n",
        encoding="utf-8",
    )
    deploy = load_deploy_config(yaml_path)
    assert [stage.name for stage in deploy.stages] == ["thinker"]


def test_load_deploy_config_minimal_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("stages: []\n", encoding="utf-8")
    assert load_deploy_config(yaml_path).stages == ()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
