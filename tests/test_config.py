from pathlib import Path

import pytest
import yaml

from nanovllm_omni.config import DeployConfig, PipelineConfig, load_config


def write_config(tmp_path: Path, value: object, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_load_config_reads_pipeline_and_deployment(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "pipeline": {
                "stages": [
                    {
                        "name": "thinker",
                        "kind": "ar",
                        "model_id": "example/thinker",
                        "model_kwargs": {"dtype": "float32", "max_model_len": 128},
                    },
                    {"name": "renderer", "kind": "diffusion", "model_id": "example/renderer"},
                ],
                "connectors": [{"source": "thinker", "target": "renderer", "payload_type": "text"}],
            },
            "deploy": {"device": "cpu", "lazy_load": False, "max_active_stages": 2},
        },
    )

    pipeline, deploy = load_config(path)

    assert isinstance(pipeline, PipelineConfig)
    assert pipeline.stages[0].name == "thinker"
    assert pipeline.stages[0].model_kwargs == {"dtype": "float32", "max_model_len": 128}
    assert pipeline.connectors[0].source == "thinker"
    assert pipeline.connectors[0].target == "renderer"
    assert pipeline.connectors[0].payload_type == "text"
    assert deploy == DeployConfig(device="cpu", lazy_load=False, max_active_stages=2)


@pytest.mark.parametrize("kind", ["ar", "diffusion", "action", "audio_decode"])
def test_load_config_accepts_each_stage_kind(tmp_path: Path, kind: str) -> None:
    path = write_config(
        tmp_path,
        {"stages": [{"name": "stage", "kind": kind, "model_id": "example/model"}]},
    )

    pipeline, _ = load_config(path)

    assert [stage.kind for stage in pipeline.stages] == [kind]


def test_load_config_rejects_duplicate_stage_names(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "stages": [
                {"name": "stage", "kind": "ar", "model_id": "example/one"},
                {"name": "stage", "kind": "diffusion", "model_id": "example/two"},
            ]
        },
    )

    with pytest.raises(ValueError, match="stage names must be unique"):
        load_config(path)


def test_load_config_rejects_invalid_stage_kind(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {"stages": [{"name": "stage", "kind": "unknown", "model_id": "example/model"}]},
    )

    with pytest.raises(ValueError, match="invalid stage kind"):
        load_config(path)


def test_load_config_allows_multiple_ar_stages_for_thinker_and_talker(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "stages": [
                {"name": "thinker", "kind": "ar", "model_id": "example/thinker"},
                {"name": "talker", "kind": "ar", "model_id": "example/talker"},
            ]
        },
    )

    pipeline, _ = load_config(path)

    assert [stage.name for stage in pipeline.stages] == ["thinker", "talker"]


def test_load_config_rejects_unknown_connector_references(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "stages": [{"name": "source", "kind": "ar", "model_id": "example/model"}],
            "connectors": [{"source": "source", "target": "missing"}],
        },
    )

    with pytest.raises(ValueError, match="connector references unknown stage"):
        load_config(path)


@pytest.mark.parametrize(
    ("connector", "expected_source", "expected_target", "expected_payload"),
    [
        ({"from": "one", "to": "two"}, "one", "two", "auto"),
        ({"from_stage": "one", "to_stage": "two", "payload": "hidden"}, "one", "two", "hidden"),
        ({"stage_from": "one", "stage_to": "two", "type": "tokens"}, "one", "two", "tokens"),
        ({"stage_from": "one", "stage_to": "two", "kind": "codec"}, "one", "two", "codec"),
    ],
)
def test_load_config_accepts_legacy_connector_keys(
    tmp_path: Path,
    connector: dict[str, str],
    expected_source: str,
    expected_target: str,
    expected_payload: str,
) -> None:
    path = write_config(
        tmp_path,
        {
            "stages": [
                {"name": "one", "kind": "ar", "model_id": "example/one"},
                {"name": "two", "kind": "audio_decode", "model_id": "example/two"},
            ],
            "connectors": [connector],
        },
    )

    pipeline, _ = load_config(path)

    assert pipeline.connectors[0].source == expected_source
    assert pipeline.connectors[0].target == expected_target
    assert pipeline.connectors[0].payload_type == expected_payload


def test_load_config_uses_deployment_alias_and_keeps_pipeline_separate(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "pipeline": {
                "stages": [{"name": "stage", "kind": "action", "model_id": "example/model"}]
            },
            "deployment": {"device": "cpu", "lazy_load": False, "max_active_stages": 3},
        },
    )

    pipeline, deploy = load_config(path)

    assert [stage.name for stage in pipeline.stages] == ["stage"]
    assert deploy == DeployConfig(device="cpu", lazy_load=False, max_active_stages=3)
    assert not hasattr(pipeline, "device")


def test_load_config_uses_deployment_defaults_when_omitted(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {"pipeline": {"stages": [{"name": "stage", "kind": "ar", "model_id": "example/model"}]}},
    )

    _, deploy = load_config(path)

    assert deploy == DeployConfig()


def test_load_config_preserves_flat_layout_deployment_keys(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "stages": [{"name": "stage", "kind": "ar", "model_id": "example/model"}],
            "device": "cpu",
            "lazy_load": False,
            "max_active_stages": 4,
        },
    )

    pipeline, deploy = load_config(path)

    assert len(pipeline.stages) == 1
    assert deploy == DeployConfig(device="cpu", lazy_load=False, max_active_stages=4)


def test_load_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("pipeline:\n  stages: [\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="pipeline.stages must be a non-empty list"):
        load_config(path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
