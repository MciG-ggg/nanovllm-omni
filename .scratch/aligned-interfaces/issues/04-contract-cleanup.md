# 04 — Contract: delete old code, full verification checklist green

**What to build:** The old API surface (`Orchestrator`, `Pipeline`, `Stage[Input,Output]` ABC, typed `Payload` dataclasses, original YAML `config.py`) is removed. `examples/audio.py` is deleted in favor of `examples/minimind_omni.py` (created in TICKET-02). The full cross-phase verification checklist from the locked plan is green.

**Blocked by:** TICKET-02, TICKET-03.

**Status:** ready-for-agent

- [x] `nanovllm_omni/stage.py` deleted.
- [x] `nanovllm_omni/payloads.py` deleted.
- [x] `nanovllm_omni/runtime/` directory (containing `pipeline.py`, `build.py`, `orchestrator.py`) deleted.
- [x] `nanovllm_omni/config.py` (the original dataclass module) deleted.
- [x] `nanovllm_omni/models/minimind_omni.py` (the single-file implementation) deleted; replaced by the package created in TICKET-02.
- [x] `examples/audio.py` deleted.
- [ ] `grep -rE "Orchestrator\b|Stage\[|TokenPayload|TensorPayload|BridgePayload|ThinkerRun|CodecTokenPayload" nanovllm_omni/ examples/ tests/` returns 0 matches.
- [ ] `grep -rE "from nanovllm_omni\.runtime|from nanovllm_omni\.payloads|from nanovllm_omni\.stage" .` returns 0 matches (excluding `.scratch/` and `node_modules/` if present).
- [ ] Cross-phase verification checklist green:
  - [ ] `python -c "from nanovllm_omni import Omni, AsyncOmni, SamplingParams, OmniRequestOutput"` succeeds.
  - [ ] `Omni("jingyaogong/minimind-3o", enforce_eager=True).generate(["hi"], SamplingParams(max_tokens=8))` returns a non-empty `list[OmniRequestOutput]` with valid audio payload.
  - [ ] `resolve_pipeline_config("minimind_o")` returns the registered `PipelineConfig`.
  - [ ] `python examples/minimind_omni.py` writes a non-empty `audio.wav`.
  - [ ] HTTP server starts via `python -m nanovllm_omni.serving.openai_adapter`; `POST /v1/chat/completions` returns OpenAI-shape response with audio.
  - [ ] All tests in `tests/` pass.
- [ ] No new third-party dependency introduced (stdlib only).
- [ ] No references to deleted classes or modules in README, docs, or examples.
- [x] `docs/aligned_interfaces.md` exists and maps each aligned nanovllm-omni class to its vllm-omni counterpart (TICKET-06 of the docs-backfill phase may add detail; this ticket requires the file to exist with at least the class-name mapping table).