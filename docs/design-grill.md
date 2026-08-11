# nanovllm-omni Design Grill

> This document records the design interview that shaped the project. It is intentionally opinionated: the goal is to make the implementation small enough to read while still exercising the important runtime boundaries in vLLM-Omni.

## Project Goal

Build `nanovllm-omni`, a learning-first and portfolio-ready minimal implementation of the basic design of [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

The project should:

- cover autoregressive multimodal LLM inference;
- cover image diffusion and image editing;
- cover vision-language-action inference;
- cover a real multi-stage omni-modal pipeline;
- run real weights rather than toy-only models;
- avoid low-level optimizations unless they are essential to the design being demonstrated;
- explain the mapping from `nanovllm-omni` to `vllm-omni` through Markdown and Jupyter notebooks.

The implementation target is a single-process, single-device reference runtime. It is not intended to reproduce vLLM-Omni's production transport, distributed placement, quantization, or CUDA-graph feature set.

## Decisions At A Glance

| Topic | Decision |
|---|---|
| Main architectural idea | Stage-based pipeline with per-stage execution policies |
| Base dependency | Use vLLM interfaces where they fit; use vLLM-Omni source as the design reference |
| AR model | Qwen2.5-VL-3B-Instruct |
| Image generation | SD3.5-medium |
| Image editing | Qwen-Image-Edit |
| VLA model | InternVLA-A1-3B |
| Main multi-stage demo | MiniMind-Omni: Thinker → Talker → Code2Wav |
| Target hardware | 24 GB VRAM, with A10G Small as the HF Spaces target |
| UI | Gradio |
| Deployment | Local plus Hugging Face Spaces |
| Tests | Unit/smoke coverage plus manual real-weight end-to-end checks |
| Documentation | README, design mapping docs, and Jupyter notebooks |
| License | Apache 2.0 |
| Python | 3.11+ |

## Interview Log

### Q1 — What does “cover vLLM-Omni's basic design” mean?

**Answer:** The stage pipeline design.

The project should model heterogeneous logical stages, cross-stage state, and stage-aware execution rather than only exposing three unrelated HTTP endpoints.

### Q2 — Which model families should be covered?

**Answer:** A multimodal LLM, diffusion, and VLA, choosing models with a reasonable quality/VRAM trade-off.

The initial choices were:

- Qwen2.5-VL-3B-Instruct for the general multimodal AR path;
- SD3.5-medium for image diffusion;
- InternVLA-A1-3B for the VLA path;
- MiniMind-Omni was later added as the main omni/audio pipeline.

### Q3 — Which optimizations should be implemented?

**Answer:** Reuse vLLM APIs where possible. If an important capability is not available through an existing interface, implement only the smallest version needed to demonstrate the design.

Explicitly out of scope for v1:

- speculative decoding;
- multi-LoRA;
- FP8/NVFP4/AWQ quantization;
- CUDA graph capture for every stage;
- distributed transport and multi-node placement;
- broad production compatibility work.

### Q4 — Who is the project for?

**Answer:** Both a learning project and a portfolio project.

That requires two things at once:

- source code that can be read end-to-end;
- a working Gradio demo and clear architecture documentation.

### Q5 — Toy models or real weights?

**Answer:** Real weights.

Toy or mocked stages are allowed in unit tests and notebooks, but the supported model paths must be designed around real model loading and inference.

### Q6 — What should be the implementation base?

**Answer:** Use `vllm` as the runtime dependency and use `vllm-omni` as the reference implementation, writing a minimal stage pipeline ourselves.

This deliberately avoids both extremes:

- importing vLLM-Omni as a black box would make the project too shallow;
- reimplementing the entire vLLM runtime would make it too large.

The intended boundary is a thin, readable orchestration layer over existing model runtimes.

### Q7 — How should diffusion fit into the project?

The initial discussion considered three options:

1. standalone text-to-image;
2. an invented AR prompt-rewrite stage followed by diffusion;
3. both standalone and multi-stage modes.

The clarification against the actual vLLM-Omni architecture was important:

- vLLM-Omni does support standalone text-to-image through diffusion pipelines such as SD3, FLUX, and Qwen-Image;
- pure text-to-image is normally one diffusion stage;
- image editing is also normally one diffusion stage with image conditioning;
- a generic “Qwen rewrites a prompt, then SD3 renders it” pipeline is a possible educational composition, but it is not the normal vLLM-Omni model topology;
- model-specific AR → DiT topologies exist, such as HunyuanImage, but are not a good 24 GB default.

**Final decision:** cover both standalone text-to-image and image editing. Use a real multi-stage model as the main pipeline demo instead of pretending the generic prompt-rewrite composition is a native vLLM-Omni topology.

### Q8 — What is Gradio?

Gradio is a Python library that turns model functions into a browser UI. It supports text, image, audio, chat, tabs, and progress reporting without requiring a separate frontend implementation.

**Final decision:** use Gradio for the local and Hugging Face Spaces demos.

### Q9 — Should the design be documented in notebooks?

**Answer:** Yes.

The documentation plan includes:

- a stage pipeline walkthrough;
- an executable `nanovllm-omni` ↔ `vllm-omni` design mapping;
- flow matching/action diffusion internals;
- MiniMind-Omni internals;
- a guide for adding another model.

### Q10 — What is the VRAM target?

**Answer:** 24 GB.

The local target is an RTX 3090/4090-class card. The hosted target is an A10G Small Hugging Face Space. Models should be loaded lazily so the demo does not require every model to stay resident simultaneously.

### Q11 — How should the VLA action decoder be implemented?

The investigation of vLLM-Omni showed that robot-policy serving uses a policy-serving path and keeps model-specific transforms/state in the diffusion pipeline. vLLM-Omni has native support for models such as InternVLA-A1 and GR00T-N1.7, while OpenPI provides a separate protocol path for policies such as π0.

**Final decision:** use InternVLA-A1-3B. It is a direct vLLM-Omni reference point and avoids making the first version depend on a separate OpenPI integration path.

The action sampler should reuse the minimal flow-matching implementation where this teaches the relevant concept. It should not duplicate the whole upstream policy stack.

### Q12 — How much testing is needed?

**Answer:** Both lightweight tests and real-weight end-to-end checks.

- CI runs CPU-safe unit tests for stage transitions, config validation, connectors, and orchestration.
- Real-weight smoke tests are explicitly marked and run manually or in a scheduled environment.
- The smoke path should load one model from each family, run one request, and validate output shape/type rather than benchmark quality.

### Q13 — What does “local + Hugging Face Spaces” mean?

The local app runs with Gradio on `localhost:7860`. The same app can run in a Hugging Face Space using a Docker image and an A10G GPU.

The v1 deployment strategy is:

- keep weights in the Hugging Face Hub rather than the repository;
- download models on first use;
- lazily load one model family per tab;
- unload inactive models when VRAM pressure requires it;
- provide a Space URL as the portfolio demo.

## MiniMind-Omni Decision

### Reference

[vLLM-Omni PR #3796](https://github.com/vllm-project/vllm-omni/pull/3796)

### Why it became the main multi-stage demo

MiniMind-Omni is a real three-stage pipeline and directly exercises the design the project is meant to teach:

```text
Thinker → Talker → Code2Wav
```

- **Thinker** performs text and multimodal understanding and produces text/bridge hidden states.
- **Talker** consumes the Thinker bridge states and produces Mimi audio codec tokens.
- **Code2Wav** decodes Mimi codec tokens into a waveform.

This is better than making the synthetic Qwen → SD3.5 composition the primary demo because MiniMind-Omni is an actual model topology reflected in the vLLM-Omni reference implementation.

### Design details to preserve

#### Post-EOS bridge-state generation

The Thinker cannot stop immediately when it emits visible EOS. The Talker needs bridge hidden states generated after EOS. The reference implementation therefore keeps a per-request state machine, forces a configurable number of padding steps (128 by default), and then emits an internal stop condition.

The minimal implementation should preserve the semantics:

- post-EOS state is per request;
- every forced step runs through the Thinker;
- bridge hidden states are captured for the Talker;
- state is cleared when the request finishes;
- concurrent requests must not share this counter.

#### Talker MTP codebooks

The Talker predicts residual Mimi codebooks with a multi-token prediction path. Codebooks become valid according to a delayed activation pattern, so each step has an `active_mask`; inactive positions receive the audio padding token before the next decode step.

The minimal implementation should keep the mask and state transition visible, without adding general CUDA-graph support.

#### Connector payloads

MiniMind-Omni means the connector cannot only carry token IDs. It must support typed payloads for:

- tokens;
- hidden-state tensors;
- codec tokens;
- audio waveforms and metadata.

The first implementation can use in-process/shared-memory transport and skip Mooncake/Mori/Yuanrong.

## Final Architecture Scope

### Included

- single-process, single-device runtime;
- ordered N-stage pipelines;
- per-stage `load`, `unload`, and `execute` lifecycle;
- typed connector payloads;
- per-request orchestration state;
- lazy model loading;
- Qwen2.5-VL, SD3.5, Qwen-Image-Edit, InternVLA-A1, and MiniMind-Omni;
- Gradio local/HF Spaces demo;
- real-weight manual smoke tests;
- notebooks explaining the design.

### Deliberately excluded from v1

- cross-node stage placement;
- distributed connectors;
- full vLLM-Omni async chunk streaming;
- generic CUDA graph capture;
- step-level diffusion batching;
- speculative decoding;
- quantization;
- multi-LoRA;
- production-grade fault recovery;
- a full reimplementation of vLLM or diffusers.

## Milestones

1. **Foundation**: `PipelineConfig`, `DeployConfig`, `Stage`, `StageRuntime`, CI.
2. **AR**: Qwen2.5-VL through vLLM and a chat tab.
3. **Diffusion**: SD3.5 text-to-image and Qwen-Image-Edit.
4. **VLA**: InternVLA-A1 and action visualization.
5. **MiniMind-Omni**: the canonical three-stage audio pipeline.
6. **Deployment**: lazy loading, Docker, HF Spaces, notebooks, and real-weight smoke tests.

## Key References

- [vLLM-Omni repository](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni architecture overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/)
- [vLLM-Omni supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
- [MiniMind-Omni support PR #3796](https://github.com/vllm-project/vllm-omni/pull/3796)
