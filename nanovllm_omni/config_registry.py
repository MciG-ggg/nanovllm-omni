from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import yaml

@dataclass
class PipelineConfig:
    name: str
    stages: list[str] = field(default_factory=list)
    default_deploy_config_name: str = "minimind_omni.yaml"

@dataclass
class DeployStageConfig:
    name: str
    default_sampling_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class DeployConfig:
    stages: list[DeployStageConfig] = field(default_factory=list)

OMNI_PIPELINES: dict[str, PipelineConfig | Callable] = {"minimind_o": PipelineConfig("minimind_o", ["thinker", "talker", "code2wav"])}
def register_pipeline(name, config): OMNI_PIPELINES[name] = config
def resolve_pipeline_config(name): return OMNI_PIPELINES.get(name)
def load_deploy_config(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    return DeployConfig([DeployStageConfig(s.get("name", ""), s.get("default_sampling_params", {})) for s in data.get("stages", [])])
def merge_pipeline_deploy(pipeline_cfg, deploy_cfg):
    return [(stage, next((x for x in deploy_cfg.stages if x.name == stage), DeployStageConfig(stage))) for stage in pipeline_cfg.stages]
