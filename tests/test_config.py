from pathlib import Path

from nanovllm_omni.config import (
    DeployConfig,
    DeployStageConfig,
    OmniEngineArgs,
    PipelineConfig,
    SamplingParams,
    load_deploy_config,
    merge_pipeline_deploy,
    resolve_pipeline_config,
)


def test_engine_args_accepts_deploy_kwargs():
    args = OmniEngineArgs(enforce_eager=True, gpu_memory_utilization=0.8, max_num_seqs=4)
    assert args.enforce_eager is True
    assert args.gpu_memory_utilization == 0.8
    assert args.max_num_seqs == 4


def test_sampling_params_matches_aligned_surface():
    params = SamplingParams(temperature=0.7, max_tokens=8, top_p=0.9, top_k=50, seed=42)
    assert params.max_tokens == 8
    assert params.top_k == 50
    assert params.seed == 42


def test_pipeline_registry_resolves_minimind():
    config = resolve_pipeline_config("minimind_o")
    assert isinstance(config, PipelineConfig)
    assert config.stages == ["thinker", "talker", "code2wav"]
    assert config.default_deploy_config_name == "minimind_omni.yaml"


def test_deploy_config_merges_stage_defaults(tmp_path: Path):
    path = tmp_path / "deploy.yaml"
    path.write_text("stages:\n  - name: thinker\n    default_sampling_params:\n      temperature: 0.7\n", encoding="utf-8")
    deploy = load_deploy_config(path)
    assert isinstance(deploy, DeployConfig)
    merged = merge_pipeline_deploy(resolve_pipeline_config("minimind_o"), deploy)
    assert merged[0][1].default_sampling_params["temperature"] == 0.7
    assert isinstance(merged[1][1], DeployStageConfig)
