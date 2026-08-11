# nanovllm-omni — Implementation TODO

> **Status**: scaffolded. Implementation begins Phase 1, Issue #2.
> **Roadmap**: 6 weeks, 40 issues across 6 phases.
> **Last updated**: see git log

---

## Conventions

- **Issue IDs**: `#N` for sequential issues; `#M1`-`#M10` for the MiniMind-Omni phase (inserted between phases 4 and 5 in the original roadmap; now main multi-stage demo).
- **Acceptance criteria** are listed under each issue. PR is mergeable only when all checkboxes are ticked.
- **LOC budget** is a soft cap; if you exceed it, leave a `ponytail:` comment explaining why the simpler version loses information.
- **Tests**: every PR must keep `pytest tests/` green on a CPU runner (smoke tests are marked `@pytest.mark.smoke` and skipped by default).

---

## Phase 1 — Foundation (Week 1)

**Goal**: repo builds, has a runnable skeleton, CI is green.

### #1 — Initialize repo ✅ (this PR)
- [x] `pyproject.toml` with vllm/diffusers/transformers/moshi/torchaudio/gradio deps
- [x] `LICENSE` (Apache 2.0)
- [x] `README.md` (skeleton with design mapping table)
- [x] `.gitignore` (Python + weights)
- [x] `TODO.md` (this file)

### #2 — Define `PipelineConfig` + `DeployConfig` in `nanovllm_omni/config.py`
- [ ] `PipelineConfig` dataclass: `stages: list[StageConfig]`, `connectors: list[ConnectorSpec]`
- [ ] `DeployConfig` dataclass: `device`, `lazy_load: bool`, `max_active_stages: int`
- [ ] `StageConfig`: `name`, `kind` (`ar`|`diffusion`|`action`|`audio_decode`), `model_id`, `model_kwargs`
- [ ] YAML loader (`load_config(path)`) returning `(PipelineConfig, DeployConfig)`
- [ ] Validation: stage names unique, kinds valid, at most one AR stage per pipeline
- [ ] **LOC budget: 150**

### #3 — Stage ABC + stage skeletons in `nanovllm_omni/stage.py`
- [ ] `Stage` ABC with: `name`, `kind`, `load()`, `unload()`, `execute(request) -> StageOutput`, `state_dict()`
- [ ] `ARStage` skeleton (delegates to `vllm.LLM`; concrete impl in #7)
- [ ] `DiffusionStage` skeleton (concrete impl in #16)
- [ ] `ActionStage` skeleton (concrete impl in #19)
- [ ] `AudioDecodeStage` skeleton (concrete impl in #M2)
- [ ] All skeletons raise `NotImplementedError("see issue #N")` so missing impls fail loudly
- [ ] **LOC budget: 250**

### #4 — `StageRuntime` minimal lifecycle in `nanovllm_omni/runtime.py`
- [ ] `StageRuntime` class holds `dict[str, Stage]`
- [ ] `load(name)` / `unload(name)` with VRAM tracking (`torch.cuda.memory_allocated`)
- [ ] `unload_all()` for HF Spaces tab switching
- [ ] Lazy loading: stage only materializes weights on first `execute()`
- [ ] **LOC budget: 150**

### #5 — CI workflow in `.github/workflows/ci.yml`
- [ ] Runs on push/PR
- [ ] Python 3.11 + 3.12 matrix
- [ ] Steps: install dev deps, `ruff check`, `black --check`, `pytest -m "not smoke"`
- [ ] Caches pip + HuggingFace downloads

### #6 — Draft `notebooks/01_stage_pipeline_walkthrough.ipynb`
- [ ] Concepts section with Mermaid diagram
- [ ] Concrete walkthrough uses MiniMind-Omni as canonical example (forward refs Phase 5; uses mocked stage objects for now)
- [ ] "Why stage pipeline?" sidebar explaining the limit of single-stage vLLM

---

## Phase 2 — AR Engine (Week 2)

**Goal**: serve Qwen2.5-VL-3B via vllm, expose chat tab.

### #7 — `models/ar.py`: Qwen2.5-VL-3B loader
- [ ] `load_qwen25_vl(model_id, **kwargs) -> vllm.LLM`
- [ ] Wraps `vllm.LLM` with sensible defaults: `dtype=bfloat16`, `max_model_len=8192`, `gpu_memory_utilization=0.85`
- [ ] Returns a thin handle exposing `generate(prompts, sampling_params) -> list[RequestOutput]`
- [ ] **LOC budget: 100**

### #8 — `stage.py::ARStage` implementation
- [ ] `load()` calls `models.ar.load_qwen25_vl`
- [ ] `execute(request)` runs prefill + decode loop using vllm
- [ ] Captures intermediate hidden states (for downstream stage); only saves `last_hidden_state` per request to limit memory
- [ ] Handles multimodal inputs (text + image) via vllm's `multi_modal_data`
- [ ] **LOC budget: 200** (across ARStage + helpers)

### #9 — `pipeline.py`: Pipeline = ordered list of stages
- [ ] `Pipeline` class: `stages: list[Stage]`, `connectors: list[Connector]`
- [ ] `__call__(request)` walks stages in order
- [ ] Validates: stage kinds must chain (e.g., AR → Diffusion OK, AR → AR requires connector kind)
- [ ] **LOC budget: 100**

### #10 — `orchestrator.py`: request lifecycle state machine
- [ ] `RequestState`: `pending → running(stage_i) → done` (+ `failed`, `cancelled`)
- [ ] `Orchestrator.submit(pipeline, request) -> Future[PipelineOutput]`
- [ ] Per-request metadata: start time, current stage, accumulated outputs
- [ ] **LOC budget: 200**

### #11 — `serving/chat.py` + `serving/app.py` minimal Gradio chat
- [ ] `app.py`: launches Gradio with lazy-load dispatch (HF Spaces friendly)
- [ ] `chat.py`: single tab, calls `Orchestrator.submit(ar_pipeline, request)`
- [ ] Streams tokens via Gradio's `gr.ChatInterface(stream=True)`
- [ ] **LOC budget: 60 + 100**

### #12 — `notebooks/02_design_mapping.ipynb` (executable)
- [ ] Live table mapping each vllm-omni module to its nanovllm-omni location
- [ ] Each row runs `!head -5 nanovllm_omni/<file>` to show the real code
- [ ] "If you read vllm-omni, read this first" sidebar

---

## Phase 3 — Diffusion Engine (Week 3)

**Goal**: serve SD3.5-medium (T2I) + Qwen-Image-Edit (I2I), expose image tabs.

### #13 — `models/diffusion.py`: SD3.5-medium + Qwen-Image-Edit loaders
- [ ] `load_sd35(model_id) -> diffusers.StableDiffusion3Pipeline`
- [ ] `load_qwen_image_edit(model_id) -> diffusers.QwenImageEditPipeline`
- [ ] Both expose `.generate(prompt, image=None, **kwargs) -> list[Image]`
- [ ] Use `diffusers` directly; do NOT implement DiT attention from scratch
- [ ] **LOC budget: 150**

### #14 — `diffusion/sampler.py`: minimal flow matching + DDPM
- [ ] `flow_matching_sample(model, x_T, sigmas, callback=None) -> x_0`
  - Euler integration over `sigmas`; default 30 steps
  - ~50 LOC, no library deps
- [ ] `ddpm_sample(model, x_T, scheduler) -> x_0`
  - Thin wrapper over `diffusers.DDPMScheduler`; only used for SD3.5
  - ~30 LOC
- [ ] Both honor a `progress_callback(step, total)` for Gradio progress bar
- [ ] **LOC budget: 200**

### #15 — `diffusion/scheduler.py`: request-level batching
- [ ] `DiffusionScheduler` accepts N requests, returns outputs as they finish
- [ ] Each request: independent diffusion run on its own batch slot
- [ ] `submit(prompt, callback) -> Future[Image]`
- [ ] NO step-level batching yet (deferred; see Future Work in docs)
- [ ] **LOC budget: 150**

### #16 — `stage.py::DiffusionStage` implementation
- [ ] `load()` calls `models.diffusion.load_sd35` or `load_qwen_image_edit`
- [ ] `execute(request)` returns PIL image (or batched images)
- [ ] Honors `request.connector_spec.conditions` (text from prior stage)
- [ ] **LOC budget: 100**

### #17 — `serving/image.py`: T2I + Edit tabs
- [ ] Two Gradio tabs sharing the same app
- [ ] T2I: textbox + Generate → image gallery
- [ ] Edit: image upload + textbox + Generate → edited image
- [ ] **LOC budget: 100**

### #18 — `notebooks/03_action_diffusion.ipynb`
- [ ] Explains flow matching ODE vs DDPM Markov chain (with diagrams)
- [ ] Live demo: 30-step Euler integration on a toy 2D vector field, then on SD3.5
- [ ] "Why does InternVLA-A1 use flow matching, not DDPM?" sidebar

---

## Phase 4 — VLA (Week 4)

**Goal**: serve InternVLA-A1, expose VLA tab with action visualization.

### #19 — `models/vla.py`: InternVLA-A1 loader
- [ ] `load_internvla(model_id) -> InternVLAPipeline` (vllm-omni-style wrapper)
- [ ] Pipeline exposes `infer(obs) -> actions`
- [ ] Reads vllm-omni's `InternVLAA1Pipeline` source as reference
- [ ] **LOC budget: 150**

### #20 — `stage.py::ActionStage` implementation
- [ ] `load()` calls `models.vla.load_internvla`
- [ ] `execute(request)` runs `obs → actions` flow matching loop
- [ ] Reuses `diffusion/sampler.py::flow_matching_sample` from #14
- [ ] **LOC budget: 100**

### #21 — `connector.py`: shared-memory tensor transport
- [ ] `Connector` ABC with `send(stage_from, payload) -> Future[Payload]` and `recv(stage_to, name) -> Payload`
- [ ] `SharedMemoryConnector`: uses `multiprocessing.shared_memory`
- [ ] Supports payloads: `TokenPayload`, `TensorPayload` (incl. hidden states), `ImagePayload`
- [ ] **LOC budget: 100**

### #22 — `orchestrator.py`: cross-stage cancellation + state propagation
- [ ] Cancel propagates from `pipeline[i]` to all `pipeline[>i]` futures
- [ ] Per-request intermediate outputs retained until downstream stages consume them
- [ ] **LOC budget: 100** (additions to #10)

### #23 — `configs/multi_stage_demo.yaml` + `examples/multi_stage.py`
- [ ] Single-stage defaults still the primary configs (ar.yaml, text_to_image.yaml, etc.)
- [ ] `multi_stage_demo.yaml` references MiniMind-Omni (defined in Phase 5)
- [ ] **LOC budget: 30 (YAML) + 50 (example)**

### #24 — `serving/vla.py`: VLA tab
- [ ] Image upload (camera view) + instruction textbox
- [ ] Run InternVLA-A1, visualize predicted action as 2D/3D vector field
- [ ] **LOC budget: 80**

---

## Phase 5 — MiniMind-Omni: multi-stage audio (Week 5) ⭐ MAIN MULTI-STAGE DEMO

**Goal**: serve MiniMind-Omni as a real 3-stage pipeline (Thinker → Talker → Code2Wav). This is the canonical demonstration of stage-based serving in nanovllm-omni.

Reference: [vllm-omni PR #3796](https://github.com/vllm-project/vllm-omni/pull/3796)

### #M1 — `models/audio.py`: MiniMind-Omni loader
- [ ] `MiniMindOmniConfig` (dataclass; mirrors HF `MiniMindOmniConfig` from `jingyaogong/minimind-3o`)
- [ ] `load_minimind_omni(model_id) -> MiniMindOmniModel` with three sub-modules: `thinker`, `talker`, `code2wav`
- [ ] Weight mapping: thin wrapper that loads `thinker.safetensors`, `talker.safetensors`, `code2wav.safetensors` separately
- [ ] **LOC budget: 120**

### #M2 — `stage.py::AudioDecodeStage` implementation
- [ ] `load()` instantiates the Code2Wav component (Mimi codec decoder)
- [ ] `execute(request)` consumes `MimiCodecToken` payload, returns `AudioPayload` (waveform tensor + sample rate)
- [ ] **LOC budget: 80**

### #M3 — `connector.py`: tensor + hidden-state payload support
- [ ] Extend `Connector` to carry `TensorPayload` (with dtype/shape metadata)
- [ ] `SharedMemoryTensor`: zero-copy via `multiprocessing.shared_memory` for tensors
- [ ] Used by `thinker2talker` to pass Thinker's intermediate hidden states
- [ ] **LOC budget: 100** (additions to #21)

### #M4 — `orchestrator.py::PerRequestPostEOSState` ⭐ design teaching point
- [ ] After Thinker emits EOS, orchestrator forces N=128 padding steps
- [ ] Each padding step still runs through Thinker; hidden state captured and passed to Talker
- [ ] Per-request counter; state cleared when request finishes
- [ ] **LOC budget: 100**
- [ ] Doc comment explains WHY (Talker needs bridge states beyond visible EOS)

### #M5 — `models/audio.py::TalkerMTP` ⭐ design teaching point
- [ ] MTP codebook prediction: predict residual codebooks for each layer-0 audio token
- [ ] Dynamic `active_mask` per step; inactive codebooks replaced with audio padding token
- [ ] ~80 LOC direct port from vllm-omni PR #3796's Talker implementation
- [ ] **LOC budget: 80**

### #M6 — `diffusion/audio_codec.py`: Mimi codec decoder integration
- [ ] `MimiCodec.decode(codec_tokens) -> waveform` using `moshi.Mimi`
- [ ] Lazy import; only loaded when MiniMind-Omni stage is activated
- [ ] **LOC budget: 100**

### #M7 — `serving/audio.py`: Gradio audio tab
- [ ] Textbox + Generate → Gradio `gr.Audio` widget (with playback)
- [ ] Streams intermediate progress (Thinker decoding → Talker MTP → Code2Wav)
- [ ] **LOC budget: 80**

### #M8 — `configs/audio.yaml` + `examples/audio.py`
- [ ] `audio.yaml`: defines 3-stage pipeline (thinker → talker → code2wav) + post-EOS config
- [ ] `examples/audio.py`: standalone CLI runner (no Gradio)
- [ ] **LOC budget: 30 + 40**

### #M9 — `notebooks/04_minimind_omni_internals.ipynb`
- [ ] Walk through 3-stage pipeline with live code cells (mocked stages)
- [ ] Diagram: post-EOS state machine (128 padding steps)
- [ ] MTP codebook active mask visualization (toy example + real MiniMind-Omni trace)
- [ ] "Why N=128 padding steps?" — discussion of bridge state semantics

### #M10 — `tests/test_audio_stage.py`
- [ ] Mock 3-stage pipeline; verify post-EOS state machine produces correct padding-step count
- [ ] Verify connector passes hidden state (tensor payload) between stages
- [ ] **LOC budget: 50**

---

## Phase 6 — Deployment + Polish (Week 6)

**Goal**: shippable repo with HF Spaces URL and complete notebooks.

### #25 — `Dockerfile` for HF Spaces GPU
- [ ] Based on `python:3.11-slim`
- [ ] Installs `.[dev]` deps
- [ ] Sets `HF_HOME=/data` for Spaces persistent storage
- [ ] `CMD ["python", "-m", "nanovllm_omni.serving.app"]`

### #26 — `serving/app.py`: lazy model loading on tab switch
- [ ] Single shared Gradio app; tabs share runtime
- [ ] On tab switch: unload previous model if VRAM pressure detected, load new
- [ ] VRAM probe: `torch.cuda.memory_allocated()` before each tab activation

### #27 — README polish
- [ ] Demo GIF: record full Gradio session (chat + image gen + edit + VLA + audio)
- [ ] Quickstart commands actually run on a fresh clone
- [ ] Hardware notes match real measured numbers

### #28 — `tests/test_smoke.py`: end-to-end with real weights
- [ ] `@pytest.mark.smoke` markers; skipped by default
- [ ] One test per model family: load weights, run single request, check output shape
- [ ] Documented as manual-only in README

### #29 — Deploy to HF Spaces
- [ ] Push to `hf.co/spaces/<your-org>/nanovllm-omni`
- [ ] Verify A10G Space boots and serves Gradio
- [ ] Pin to A10G Small tier in Space settings

### #30 — `notebooks/05_adding_a_new_model.ipynb`
- [ ] Template notebook: reader can fork and add a new model in <1 hour
- [ ] Steps: subclass `Stage`, add `models/<your_model>.py`, register in `models/__init__.py`, write YAML config
- [ ] Worked example: add Stable-Diffusion-XL as a new DiffusionStage

---

## Future work (post-v1)

These are deliberately cut from v1. Listed for transparency; not commitments.

- **Cross-device / cross-node stage placement** (vllm-omni supports this via `StageRuntime`)
- **MoE token-level expert parallelism** for MiniMind-3o-MoE variant
- **Step-level diffusion batching** (currently only request-level)
- **CUDA graph capture** for diffusion stages
- **Async chunk streaming** for MiniMind-Omni (vllm-omni PR notes this gap)
- **Speculative decoding** in AR stage
- **Multi-LoRA** adapter swapping
- **Quantization** (FP8 / NVFP4 / AWQ) for diffusion stages
- **Distributed layerwise offload** for video diffusion (MiniMax-H3-style)

Each of these would ~2× the LOC and add a new debugging surface; explicitly excluded to keep `nanovllm-omni` a learning-first artifact.

---

## Open questions (not blocking, but worth answering)

- Do we want to keep the original `multi_stage_demo.yaml` synthetic Qwen→SD3.5 path as an "alternate demo", or fully retire it in favor of MiniMind-Omni?  → **Decision**: retire; MiniMind-Omni is canonical.
- Should `examples/audio.py` stream audio chunks live (chunked Code2Wav) or only emit the final waveform?  → **Default**: final waveform; streaming is Future Work.
- License for model-specific code (e.g., Talker MTP port): follow MiniMind's repo license?  → **Default**: Apache 2.0 with attribution in source headers.

---

## Progress tracker

| Phase | Started | Completed | Issues closed |
|---|---|---|---|
| 1 — Foundation | today | — | #1 |
| 2 — AR | — | — | — |
| 3 — Diffusion | — | — | — |
| 4 — VLA | — | — | — |
| 5 — MiniMind-Omni ⭐ | — | — | — |
| 6 — Deploy | — | — | — |

Update by ticking the boxes above as issues close.