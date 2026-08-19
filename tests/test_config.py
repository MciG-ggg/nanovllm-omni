from pathlib import Path

from nanovllm_omni.config import (
    DeployConfig,
    OmniEngineArgs,
    PipelineConfig,
    SamplingParams,
    StageConfig,
    load_deploy_config,
    merge_pipeline_deploy,
    resolve_pipeline_config,
)


def test_engine_args_accepts_deploy_kwargs():
    args = OmniEngineArgs(enforce_eager=True, gpu_memory_utilization=0.8, max_num_seqs=4)
    assert args.enforce_eager is True
    assert args.gpu_memory_utilization == 0.8
    assert args.max_num_seqs == 4


def test_engine_args_swallows_unknown_kwargs():
    args = OmniEngineArgs(unknown_flag=1, quantum_entanglement=True)
    assert args.extra["unknown_flag"] == 1
    assert args.extra["quantum_entanglement"] is True


def test_sampling_params_matches_aligned_surface():
    params = SamplingParams(temperature=0.7, max_tokens=8, top_p=0.9, top_k=50, seed=42)
    assert params.max_tokens == 8
    assert params.top_k == 50
    assert params.seed == 42


def test_pipeline_registry_resolves_minimind():
    config = resolve_pipeline_config("minimind_o")
    assert isinstance(config, PipelineConfig)
    assert [s.name for s in config.stages] == ["thinker", "talker", "code2wav"]
    assert config.default_deploy_config_name == "minimind_omni.yaml"
    assert config.registration_handles == ("minimind_o", "jingyaogong/minimind-3o")


def test_pipeline_registry_resolves_by_hf_handle():
    config = resolve_pipeline_config("jingyaogong/minimind-3o")
    assert isinstance(config, PipelineConfig)
    assert config.name == "minimind_o"


def test_pipeline_registry_returns_none_for_unknown():
    assert resolve_pipeline_config("nonexistent_model") is None


def test_stage_config_is_frozen():
    cfg = StageConfig(
        stage_id=0,
        name="thinker",
        kind="ar",
        factory=lambda deploy, args: None,
    )
    try:
        cfg.name = "talker"
    except Exception:
        pass
    else:
        raise AssertionError("StageConfig must be frozen")


def test_deploy_config_merges_stage_defaults(tmp_path: Path):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "stages:\n"
        "  - name: thinker\n"
        "    default_sampling_params:\n"
        "      temperature: 0.7\n"
        "      max_tokens: 512\n"
        "  - name: talker\n"
        "    default_sampling_params:\n"
        "      temperature: 0.2\n"
        "      watchdog_limit: 192\n"
        "  - name: code2wav\n"
        "    default_sampling_params: {}\n",
        encoding="utf-8",
    )
    deploy = load_deploy_config(path)
    assert isinstance(deploy, DeployConfig)
    assert {s.name for s in deploy.stages} == {"thinker", "talker", "code2wav"}

    pipeline = resolve_pipeline_config("minimind_o")
    assert pipeline is not None
    merged = merge_pipeline_deploy(pipeline, deploy)
    by_name = {stage.name: defaults for stage, defaults in merged}
    assert by_name["thinker"]["temperature"] == 0.7
    assert by_name["thinker"]["max_tokens"] == 512
    assert by_name["talker"]["temperature"] == 0.2
    assert by_name["talker"]["watchdog_limit"] == 192
    assert by_name["code2wav"] == {}


def test_register_pipeline_adds_entry():
    from nanovllm_omni.config_registry import (
        OMNI_PIPELINES,
        register_pipeline,
    )

    def my_factory(deploy, args):
        return None

    fake = PipelineConfig(
        name="fake_o",
        stages=(
            StageConfig(stage_id=0, name="only", kind="ar", factory=my_factory, is_terminal=True),
        ),
        default_deploy_config_name="fake_o.yaml",
    )
    register_pipeline(fake)
    try:
        assert resolve_pipeline_config("fake_o") is fake
    finally:
        OMNI_PIPELINES.pop("fake_o", None)
