# 02 — `Omni(model).generate(prompts, sp)` end-to-end (Python seam)

**What to build:** A user can `from nanovllm_omni import Omni, SamplingParams, OmniRequestOutput` and run MiniMind-O audio end-to-end via the aligned Python API. The result is an `OmniRequestOutput` whose `multimodal_output["audio"]` is a valid WAV payload. The old `Orchestrator` API continues to work for any existing callers during this ticket — this is the expand phase of the contract.

**Blocked by:** TICKET-01 (clean README must land first).

**Status:** ready-for-agent

- [ ] `from nanovllm_omni import Omni, AsyncOmni, OmniBase, SamplingParams, OmniRequestOutput` succeeds.
- [ ] `OmniEngineArgs(**kwargs)` accepts at least: `enforce_eager`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `dtype`, `tensor_parallel_size`, `trust_remote_code`, plus an `extra: dict` for unknown kwargs.
- [ ] `SamplingParams(temperature=0.7, max_tokens=8, top_p=0.9, top_k=50, seed=42, n=1, extra={"stage": "thinker"})` constructs without error. Field names match vLLM's exactly.
- [ ] `SamplingParams` is `@dataclass(frozen=True)`.
- [ ] `Omni("jingyaogong/minimind-3o", enforce_eager=True).generate(["hi"], SamplingParams(max_tokens=8))` returns a non-empty `list[OmniRequestOutput]`.
- [ ] The returned `OmniRequestOutput` exposes `.multimodal_output["audio"]` as an `AudioPayload` whose `.wav_bytes()` returns valid 16-bit PCM mono audio (verifiable by writing to disk and reading back with `wave.open`).
- [ ] `OmniRequestOutput` has `.is_pipeline_output`, `.is_diffusion_output`, `.unwrap()` properties matching vllm-omni's contract.
- [ ] `resolve_pipeline_config("minimind_o")` returns a `PipelineConfig` whose `default_deploy_config_name` points to a YAML in `deploy/` (filename: `minimind_omni.yaml` or compatible).
- [ ] `OMNI_PIPELINES` is a module-level `dict[str, PipelineConfig | PipelineResolverFunc]` with at least one entry for `minimind_o`.
- [ ] `load_deploy_config(<deploy_yaml_path>)` returns a `DeployConfig` with `stages` list containing entries for the 3 stages (thinker / talker / code2wav), each with `default_sampling_params` carrying the previous hardcoded values (thinker temperature=0.7, max_tokens=512; talker temperature=0.2, watchdog_limit=192).
- [ ] `merge_pipeline_deploy(pipeline_cfg, deploy_cfg)` produces per-stage configs the engine can consume.
- [ ] `examples/minimind_omni.py` exists and runs end-to-end against MiniMind-O using the new `Omni` API, writing `audio.wav` and exiting 0 on success. Old `examples/audio.py` is left untouched in this ticket.
- [ ] Old `from nanovllm_omni.runtime import Orchestrator` (or equivalent) still imports without error and the unchanged `examples/audio.py` smoke-test path still runs successfully.
- [ ] New tests exist and pass: `tests/test_sampling_params.py`, `tests/test_engine_args.py`, `tests/test_pipeline_registry.py`, `tests/test_omni_request_output.py`.
- [ ] All previously existing tests still pass.
- [ ] No new third-party dependency introduced.