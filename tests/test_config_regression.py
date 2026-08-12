"""Regression tests for the YAML configuration loader (Issue #2).

Complements ``tests/test_config.py`` with extra coverage that exercises:

* direct dataclass invariants (so callers that bypass ``load_config`` are
  still validated),
* YAML-level type/structure edge cases that the existing happy-path tests
  do not cover,
* deployment defaults and validation corners identified by inspecting
  ``config.py``,
* compatibility behaviors (alias keys, ``deploy`` vs ``deployment``
  priority, ignored extra keys, defaults).

These tests are CPU-only, deterministic, and do not import torch/vllm/
diffusers so they run on any platform the project supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nanovllm_omni.config import (
    ConnectorSpec,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    load_config,
)

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def write_config(tmp_path: Path, value: Any, name: str = "config.yaml") -> Path:
    """Serialize ``value`` to a YAML file under ``tmp_path`` and return it."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# StageConfig dataclass invariants (direct construction)                      #
# --------------------------------------------------------------------------- #


class TestStageConfigDataclass:
    """Direct construction must enforce the same invariants as the loader."""

    def test_minimal_valid_construction(self) -> None:
        stage = StageConfig(name="thinker", kind="ar", model_id="example/x")
        assert stage.model_kwargs == {}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "kind": "ar", "model_id": "x"}, "name must be a non-empty string"),
            ({"name": "   ", "kind": "ar", "model_id": "x"}, "name must be a non-empty string"),
            ({"name": 0, "kind": "ar", "model_id": "x"}, "name must be a non-empty string"),
            ({"name": "s", "kind": "", "model_id": "x"}, "invalid stage kind"),
            ({"name": "s", "kind": "AR", "model_id": "x"}, "invalid stage kind"),
            ({"name": "s", "kind": "text", "model_id": "x"}, "invalid stage kind"),
            ({"name": "s", "kind": "ar", "model_id": ""}, "model_id must be a non-empty string"),
            ({"name": "s", "kind": "ar", "model_id": "  "}, "model_id must be a non-empty string"),
            (
                {"name": "s", "kind": "ar", "model_id": "x", "model_kwargs": [1, 2]},
                "model_kwargs must be a mapping",
            ),
            (
                {"name": "s", "kind": "ar", "model_id": "x", "model_kwargs": "not a dict"},
                "model_kwargs must be a mapping",
            ),
        ],
    )
    def test_invalid_construction(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            StageConfig(**kwargs)


# --------------------------------------------------------------------------- #
# ConnectorSpec dataclass invariants                                          #
# --------------------------------------------------------------------------- #


class TestConnectorSpecDataclass:
    """``ConnectorSpec`` rejects empty/whitespace and non-string fields."""

    def test_defaults_payload_type(self) -> None:
        c = ConnectorSpec(source="a", target="b")
        assert c.payload_type == "auto"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"source": "", "target": "b"}, "source must be a non-empty string"),
            ({"source": "a", "target": ""}, "target must be a non-empty string"),
            (
                {"source": "a", "target": "b", "payload_type": ""},
                "payload_type must be a non-empty string",
            ),
            ({"source": None, "target": "b"}, "source must be a non-empty string"),
            (
                {"source": "a", "target": "b", "payload_type": None},
                "payload_type must be a non-empty string",
            ),
        ],
    )
    def test_invalid_construction(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ConnectorSpec(**kwargs)


# --------------------------------------------------------------------------- #
# PipelineConfig dataclass invariants                                         #
# --------------------------------------------------------------------------- #


class TestPipelineConfigDataclass:
    """Direct construction enforces unique names and connector references."""

    def test_zero_ar_stages_is_allowed(self) -> None:
        pipeline = PipelineConfig(
            stages=[
                StageConfig(name="img", kind="diffusion", model_id="x"),
                StageConfig(name="wav", kind="audio_decode", model_id="y"),
            ]
        )
        assert pipeline.connectors == []

    def test_one_ar_stage_is_allowed(self) -> None:
        pipeline = PipelineConfig(stages=[StageConfig(name="thinker", kind="ar", model_id="x")])
        assert len(pipeline.stages) == 1

    def test_duplicate_stage_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage names must be unique"):
            PipelineConfig(
                stages=[
                    StageConfig(name="dup", kind="ar", model_id="x"),
                    StageConfig(name="dup", kind="diffusion", model_id="y"),
                ]
            )

    def test_multiple_ar_stages_are_allowed(self) -> None:
        pipeline = PipelineConfig(
            stages=[
                StageConfig(name="thinker", kind="ar", model_id="x"),
                StageConfig(name="talker", kind="ar", model_id="y"),
            ]
        )
        assert [stage.name for stage in pipeline.stages] == ["thinker", "talker"]

    def test_connector_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="connector references unknown stage"):
            PipelineConfig(
                stages=[StageConfig(name="real", kind="ar", model_id="x")],
                connectors=[ConnectorSpec(source="ghost", target="real")],
            )

    def test_connector_unknown_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="connector references unknown stage"):
            PipelineConfig(
                stages=[StageConfig(name="real", kind="ar", model_id="x")],
                connectors=[ConnectorSpec(source="real", target="ghost")],
            )

    def test_self_loop_connector_is_allowed(self) -> None:
        """No explicit validation forbids source == target; document the choice."""
        pipeline = PipelineConfig(
            stages=[StageConfig(name="s", kind="ar", model_id="x")],
            connectors=[ConnectorSpec(source="s", target="s")],
        )
        assert pipeline.connectors[0].source == pipeline.connectors[0].target == "s"


# --------------------------------------------------------------------------- #
# DeployConfig dataclass invariants                                           #
# --------------------------------------------------------------------------- #


class TestDeployConfigDataclass:
    """``DeployConfig`` defaults and validation corners (incl. bool/int trap)."""

    def test_defaults(self) -> None:
        assert DeployConfig() == DeployConfig(device="cuda", lazy_load=True, max_active_stages=1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"device": ""}, "device must be a non-empty string"),
            ({"device": "  "}, "device must be a non-empty string"),
            ({"device": 0}, "device must be a non-empty string"),
            ({"lazy_load": 1}, "lazy_load must be a boolean"),
            ({"lazy_load": "true"}, "lazy_load must be a boolean"),
            ({"max_active_stages": True}, "max_active_stages must be an integer"),
            ({"max_active_stages": 0}, "max_active_stages must be at least 1"),
            ({"max_active_stages": -1}, "max_active_stages must be at least 1"),
            ({"max_active_stages": 1.5}, "max_active_stages must be an integer"),
            ({"max_active_stages": "1"}, "max_active_stages must be an integer"),
        ],
    )
    def test_invalid_construction(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DeployConfig(**kwargs)


# --------------------------------------------------------------------------- #
# YAML loader: missing required fields                                        #
# --------------------------------------------------------------------------- #


class TestLoadConfigRequiredFields:
    """Loader surfaces missing/None stage fields with stage-indexed labels."""

    @pytest.mark.parametrize(
        "missing",
        ["name", "kind", "model_id"],
    )
    def test_missing_stage_field_reports_index(self, tmp_path: Path, missing: str) -> None:
        stage = {"name": "s", "kind": "ar", "model_id": "x"}
        stage.pop(missing)
        path = write_config(tmp_path, {"stages": [stage]})
        with pytest.raises(ValueError, match=f"pipeline.stages\\[0\\].{missing}"):
            load_config(path)

    @pytest.mark.parametrize(
        "field",
        ["name", "kind", "model_id"],
    )
    def test_null_stage_field_rejected(self, tmp_path: Path, field: str) -> None:
        stage = {"name": "s", "kind": "ar", "model_id": "x", field: None}
        path = write_config(tmp_path, {"stages": [stage]})
        with pytest.raises(ValueError, match=f"pipeline.stages\\[0\\].{field}"):
            load_config(path)

    def test_whitespace_only_stage_name_rejected(self, tmp_path: Path) -> None:
        # Bypass yaml.safe_dump's quoting (which would have produced a string
        # anyway) and write raw to make sure whitespace survives serialization.
        path = tmp_path / "config.yaml"
        path.write_text(
            'stages:\n  - name: "  "\n    kind: ar\n    model_id: x\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="pipeline.stages\\[0\\].name"):
            load_config(path)

    def test_case_mismatched_kind_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"stages": [{"name": "s", "kind": "AR", "model_id": "x"}]})
        with pytest.raises(ValueError, match="invalid stage kind"):
            load_config(path)


# --------------------------------------------------------------------------- #
# YAML loader: structural / type validation                                   #
# --------------------------------------------------------------------------- #


class TestLoadConfigStructure:
    """Loader must reject non-mapping/scalar roots and non-list fields."""

    def test_root_is_a_list_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="config must be a mapping"):
            load_config(path)

    def test_root_is_a_scalar_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("42\n", encoding="utf-8")
        with pytest.raises(ValueError, match="config must be a mapping"):
            load_config(path)

    def test_pipeline_value_is_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"pipeline": [1, 2, 3]})
        with pytest.raises(ValueError, match="pipeline must be a mapping"):
            load_config(path)

    def test_stages_is_not_a_list(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"stages": "not a list"})
        with pytest.raises(ValueError, match="pipeline.stages must be a non-empty list"):
            load_config(path)

    def test_empty_pipeline_mapping_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"pipeline": {}})
        with pytest.raises(ValueError, match="pipeline.stages must be a non-empty list"):
            load_config(path)

    def test_empty_stages_list_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"stages": []})
        with pytest.raises(ValueError, match="pipeline.stages must be a non-empty list"):
            load_config(path)

    def test_stage_entry_is_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"stages": ["just a string"]})
        with pytest.raises(ValueError, match="pipeline.stages\\[0\\] must be a mapping"):
            load_config(path)

    def test_model_kwargs_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x", "model_kwargs": [1, 2]}]},
        )
        with pytest.raises(ValueError, match="model_kwargs must be a mapping"):
            load_config(path)

    def test_connectors_not_a_list(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "s", "kind": "ar", "model_id": "x"}],
                "connectors": "oops",
            },
        )
        with pytest.raises(ValueError, match="pipeline.connectors must be a list"):
            load_config(path)

    def test_connector_entry_is_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "s", "kind": "ar", "model_id": "x"}],
                "connectors": ["just a string"],
            },
        )
        with pytest.raises(ValueError, match="pipeline.connectors\\[0\\] must be a mapping"):
            load_config(path)


# --------------------------------------------------------------------------- #
# YAML loader: connector alias priority and required keys                     #
# --------------------------------------------------------------------------- #


class TestLoadConfigConnectors:
    """Connector alias precedence: source/from/from_stage/stage_from, etc."""

    @pytest.mark.parametrize("alias", ["source", "from", "from_stage", "stage_from"])
    def test_source_aliases_resolve_to_same_value(self, tmp_path: Path, alias: str) -> None:
        """Any of the source-alias keys (``source``, ``from``,
        ``from_stage``, ``stage_from``) is accepted as the connector source."""
        single = {"target": "beta", alias: "alpha"}
        path = write_config(
            tmp_path,
            {
                "stages": [
                    {"name": "alpha", "kind": "ar", "model_id": "x"},
                    {"name": "beta", "kind": "diffusion", "model_id": "y"},
                ],
                "connectors": [single],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors[0].source == "alpha"

    def test_source_first_match_wins_when_multiple_aliases_present(
        self,
        tmp_path: Path,
    ) -> None:
        """``_first`` walks ``("source", "from", "from_stage", "stage_from")``
        in order; whichever alias appears first in the tuple is taken even
        if other aliases with different values are also present."""
        mapping = {
            "target": "beta",
            "from": "should_be_ignored",
            "from_stage": "should_be_ignored",
            "stage_from": "should_be_ignored",
            "source": "alpha",
        }
        path = write_config(
            tmp_path,
            {
                "stages": [
                    {"name": "alpha", "kind": "ar", "model_id": "x"},
                    {"name": "beta", "kind": "diffusion", "model_id": "y"},
                ],
                "connectors": [mapping],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors[0].source == "alpha"

    @pytest.mark.parametrize("alias", ["target", "to", "to_stage", "stage_to"])
    def test_target_aliases(self, tmp_path: Path, alias: str) -> None:
        """Any of the target-alias keys resolves to ``ConnectorSpec.target``."""
        single = {"source": "alpha", alias: "beta"}
        path = write_config(
            tmp_path,
            {
                "stages": [
                    {"name": "alpha", "kind": "ar", "model_id": "x"},
                    {"name": "beta", "kind": "diffusion", "model_id": "y"},
                ],
                "connectors": [single],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors[0].target == "beta"

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("payload_type", "hidden_states"),
            ("payload", "hidden_states"),
            ("type", "hidden_states"),
            ("kind", "hidden_states"),
        ],
    )
    def test_payload_type_aliases(self, tmp_path: Path, alias: str, expected: str) -> None:
        """Any of the payload-type aliases resolves to ``payload_type``."""
        single = {"source": "alpha", "target": "beta", alias: expected}
        path = write_config(
            tmp_path,
            {
                "stages": [
                    {"name": "alpha", "kind": "ar", "model_id": "x"},
                    {"name": "beta", "kind": "diffusion", "model_id": "y"},
                ],
                "connectors": [single],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors[0].payload_type == expected

    def test_connector_missing_source_raises(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "a", "kind": "ar", "model_id": "x"}],
                "connectors": [{"target": "a"}],
            },
        )
        with pytest.raises(
            ValueError, match="requires one of: source, from, from_stage, stage_from"
        ):
            load_config(path)

    def test_connector_missing_target_raises(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "a", "kind": "ar", "model_id": "x"}],
                "connectors": [{"source": "a"}],
            },
        )
        with pytest.raises(ValueError, match="requires one of: target, to, to_stage, stage_to"):
            load_config(path)

    def test_connector_missing_payload_defaults_to_auto(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "a", "kind": "ar", "model_id": "x"}],
                "connectors": [{"source": "a", "target": "a"}],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors[0].payload_type == "auto"

    def test_connector_extra_keys_are_ignored(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "a", "kind": "ar", "model_id": "x"}],
                "connectors": [{"source": "a", "target": "a", "future_field": 42}],
            },
        )
        pipeline, _ = load_config(path)
        assert not hasattr(pipeline.connectors[0], "future_field")

    def test_connector_empty_source_string_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "a", "kind": "ar", "model_id": "x"}],
                "connectors": [{"source": "", "target": "a"}],
            },
        )
        with pytest.raises(ValueError, match="connector source must be a non-empty string"):
            load_config(path)


# --------------------------------------------------------------------------- #
# YAML loader: deployment defaults, separation, alias priority                 #
# --------------------------------------------------------------------------- #


class TestLoadConfigDeployment:
    """Deployment defaults, ``deploy`` vs ``deployment``, flat-layout scrape."""

    def test_full_pipeline_with_defaults(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "pipeline": {
                    "stages": [
                        {"name": "thinker", "kind": "ar", "model_id": "x"},
                        {"name": "renderer", "kind": "diffusion", "model_id": "y"},
                    ],
                }
            },
        )
        pipeline, deploy = load_config(path)
        assert [s.name for s in pipeline.stages] == ["thinker", "renderer"]
        assert deploy == DeployConfig()

    def test_deploy_takes_priority_over_deployment(self, tmp_path: Path) -> None:
        """``raw.get('deploy', raw.get('deployment'))`` -> ``deploy`` wins."""
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
                "deploy": {"device": "cuda", "lazy_load": True, "max_active_stages": 7},
                "deployment": {"device": "cpu", "lazy_load": False, "max_active_stages": 3},
            },
        )
        _, deploy = load_config(path)
        assert deploy == DeployConfig(device="cuda", lazy_load=True, max_active_stages=7)

    def test_deployment_used_when_deploy_absent(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
                "deployment": {"device": "cpu", "lazy_load": False, "max_active_stages": 3},
            },
        )
        _, deploy = load_config(path)
        assert deploy == DeployConfig(device="cpu", lazy_load=False, max_active_stages=3)

    def test_flat_layout_partial_keys_use_defaults_for_missing(self, tmp_path: Path) -> None:
        """When only ``device`` is at root, ``lazy_load``/``max_active_stages``
        come from the ``DeployConfig`` defaults."""
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "s", "kind": "ar", "model_id": "x"}],
                "device": "cpu",
            },
        )
        _, deploy = load_config(path)
        assert deploy == DeployConfig(device="cpu", lazy_load=True, max_active_stages=1)

    def test_root_pipeline_key_wins_over_flat_stages(self, tmp_path: Path) -> None:
        """``raw.get('pipeline', raw)`` — if ``pipeline`` is present, the flat
        ``stages`` key is ignored."""
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "from_pipeline", "kind": "ar", "model_id": "x"}]},
                "stages": [{"name": "from_root", "kind": "ar", "model_id": "y"}],
            },
        )
        pipeline, _ = load_config(path)
        assert [s.name for s in pipeline.stages] == ["from_pipeline"]

    def test_extra_deploy_keys_are_ignored(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
                "deploy": {
                    "device": "cpu",
                    "lazy_load": False,
                    "max_active_stages": 2,
                    "note": "hi",
                },
            },
        )
        _, deploy = load_config(path)
        assert not hasattr(deploy, "note")
        assert deploy == DeployConfig(device="cpu", lazy_load=False, max_active_stages=2)

    def test_extra_stage_keys_are_ignored(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x", "future_field": 42}]},
        )
        pipeline, _ = load_config(path)
        assert not hasattr(pipeline.stages[0], "future_field")

    def test_deploy_block_must_be_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
                "deploy": ["not", "a", "mapping"],
            },
        )
        with pytest.raises(ValueError, match="deploy must be a mapping"):
            load_config(path)

    def test_deployment_alias_block_must_be_a_mapping(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
                "deployment": ["not", "a", "mapping"],
            },
        )
        with pytest.raises(ValueError, match="deploy must be a mapping"):
            load_config(path)

    @pytest.mark.parametrize(
        ("deploy_block", "expected"),
        [
            (
                {"device": "", "lazy_load": True, "max_active_stages": 1},
                "device must be a non-empty string",
            ),
            (
                {"device": "cpu", "lazy_load": 1, "max_active_stages": 1},
                "lazy_load must be a boolean",
            ),
            (
                {"device": "cpu", "lazy_load": True, "max_active_stages": 0},
                "max_active_stages must be at least 1",
            ),
        ],
    )
    def test_deploy_block_validates_each_key(
        self, tmp_path: Path, deploy_block: dict[str, Any], expected: str
    ) -> None:
        cfg = {
            "pipeline": {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
            "deploy": deploy_block,
        }
        path = write_config(tmp_path, cfg)
        with pytest.raises(ValueError, match=expected):
            load_config(path)


# --------------------------------------------------------------------------- #
# YAML loader: compatibility / path-input handling                             #
# --------------------------------------------------------------------------- #


class TestLoadConfigCompat:
    """Loader accepts both ``str`` and ``Path``; preserves nested kwargs."""

    def test_accepts_pathlib_path(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
        )
        pipeline, _ = load_config(path)
        assert pipeline.stages[0].name == "s"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
        )
        pipeline, _ = load_config(str(path))
        assert pipeline.stages[0].name == "s"

    def test_nested_model_kwargs_preserved(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [
                    {
                        "name": "s",
                        "kind": "ar",
                        "model_id": "x",
                        "model_kwargs": {"a": {"nested": True}, "b": [1, 2]},
                    }
                ]
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.stages[0].model_kwargs == {"a": {"nested": True}, "b": [1, 2]}

    def test_connectors_omitted_defaults_to_empty(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors == []

    def test_empty_connectors_list_is_accepted(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "stages": [{"name": "s", "kind": "ar", "model_id": "x"}],
                "connectors": [],
            },
        )
        pipeline, _ = load_config(path)
        assert pipeline.connectors == []

    def test_stage_model_kwargs_default_to_empty_dict(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {"stages": [{"name": "s", "kind": "ar", "model_id": "x"}]},
        )
        pipeline, _ = load_config(path)
        assert pipeline.stages[0].model_kwargs == {}

    def test_yaml_null_at_root_treated_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("null\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline.stages must be a non-empty list"):
            load_config(path)
