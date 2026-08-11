# nanovllm-omni v1: MiniMind-Omni Stage Runtime

## Problem Statement

As a learner and portfolio project author, I want to understand and demonstrate the basic runtime design of vLLM-Omni without reproducing its production-scale implementation.

The current repository contains the project scaffold, design decisions, roadmap, and dependency declarations, but no executable stage runtime. It cannot yet represent a heterogeneous request, move state between stages, load a real model, or demonstrate the central multi-stage behavior.

The most important gap is the absence of a complete, real, readable pipeline. A collection of independent AR, diffusion, and VLA wrappers would not demonstrate the defining idea of vLLM-Omni: a request whose model components have different execution policies and communicate through explicit stage boundaries.

The first complete path must therefore be a small real omni-modal pipeline that can run on the 24 GB target hardware and expose the important runtime concepts without requiring distributed placement, production transport, or low-level kernel optimization.

## Solution

Build a minimal single-process, single-device stage runtime and use MiniMind-Omni as the canonical multi-stage demonstration.

The runtime will represent an ordered pipeline of typed stages. An `Orchestrator` will own per-request lifecycle and stage progression. A `StageRuntime` will own stage loading, unloading, and lazy materialization. An `OmniConnector`-style boundary will carry typed payloads, including tokens, bridge hidden states, Mimi codec tokens, and audio output metadata.

The primary real-weight path will be:

```text
Thinker → Talker → Code2Wav
```

- **Thinker** performs text and multimodal understanding and produces text plus bridge hidden states.
- **Talker** consumes bridge hidden states and produces Mimi codec tokens.
- **Code2Wav** decodes Mimi codec tokens into a waveform.

The Thinker will preserve the MiniMind-Omni post-EOS behavior: after visible EOS, a per-request state machine forces the configured padding steps so downstream Talker bridge states remain available. The Talker will preserve its delayed MTP codebook activation mask.

The same runtime contracts will support the other planned model families:

- Qwen2.5-VL-3B-Instruct as a single AR stage;
- SD3.5-medium as a single diffusion stage for text-to-image;
- Qwen-Image-Edit as a single diffusion stage for image editing;
- InternVLA-A1-3B as the VLA/action path.

Model runtimes will be reused where practical: vLLM for AR execution, diffusers for image diffusion, and existing Mimi/audio tooling for codec decoding. The project will implement only the orchestration and teaching-critical behavior that is not already provided by those runtimes.

## User Stories

1. As a learner of inference runtimes, I want to define an ordered pipeline of heterogeneous stages, so that I can see how AR, codec, diffusion, and action workloads compose into one request.
2. As a learner of vLLM-Omni, I want each stage to have an explicit execution policy and lifecycle, so that I can distinguish model topology from runtime placement.
3. As a learner, I want the pipeline to carry typed payloads between stages, so that I can understand why token IDs, hidden states, codec tokens, images, and waveforms need different contracts.
4. As a user, I want to submit text to MiniMind-Omni and receive an audio waveform, so that I can verify the entire Thinker → Talker → Code2Wav path with real weights.
5. As a user, I want Thinker-generated text and bridge states to remain correct after visible EOS, so that the Talker receives the hidden-state context required for valid audio generation.
6. As a learner, I want to inspect the post-EOS per-request state transition, so that I can understand why an AR scheduler sometimes must continue after EOS.
7. As a concurrent inference user, I want post-EOS counters and bridge states isolated per request, so that one audio request cannot affect another request.
8. As a learner, I want to see the Talker's MTP codebook active mask, so that I can understand delayed codebook activation and audio-token generation.
9. As a user, I want Mimi codec tokens to be decoded into a playable waveform, so that the output is useful beyond an internal tensor trace.
10. As a learner, I want to run the MiniMind-Omni pipeline with mocked stages, so that I can study and test orchestration without downloading model weights.
11. As a maintainer, I want stage loading and unloading to be explicit, so that the application can keep one model family resident at a time on a 24 GB GPU.
12. As a Hugging Face Spaces user, I want models to load lazily when a Gradio tab is used, so that the hosted demo does not require all model families to fit in VRAM simultaneously.
13. As an AR inference user, I want Qwen2.5-VL-3B available through the same stage contract, so that the runtime demonstrates reuse of vLLM rather than reimplementing an AR engine.
14. As an image-generation user, I want SD3.5-medium available as a single diffusion stage, so that text-to-image is represented without inventing an unnecessary multi-stage topology.
15. As an image-editing user, I want Qwen-Image-Edit available with image conditioning, so that the project covers the common single-stage multimodal diffusion path.
16. As a VLA learner, I want InternVLA-A1-3B represented through an action stage, so that action inference can share the runtime's typed payload and flow-matching concepts.
17. As a portfolio viewer, I want a Gradio application with chat, image, VLA, and audio experiences, so that the architectural scope is demonstrable without writing a separate frontend.
18. As a documentation reader, I want a design mapping between nanovllm-omni and vLLM-Omni, so that every simplification is explicit rather than mistaken for a missing feature.
19. As a documentation reader, I want Jupyter notebooks that execute the stage walkthrough and MiniMind-Omni internals, so that I can learn from runnable examples rather than prose alone.
20. As a maintainer, I want the project to run unit tests without model weights or a GPU, so that changes to orchestration and configuration can be validated in CI.
21. As a maintainer, I want manual real-weight smoke tests through the same public pipeline entrypoint, so that model integration is checked without making every CI run download large checkpoints.
22. As a contributor, I want a clear model-stage registration contract, so that adding another model does not require changing unrelated orchestration code.
23. As a researcher, I want to compare the minimal runtime's topology and responsibilities with vLLM-Omni's Orchestrator, StageRuntime, and connector concepts, so that I can use the repository as a study artifact.
24. As a project author, I want the implementation to remain single-process and readable, so that the project teaches the core design without becoming a distributed systems reproduction.
25. As a project author, I want the repository to document deliberately omitted optimizations, so that users understand the boundary between the educational runtime and production vLLM-Omni.

## Implementation Decisions

- The v1 runtime is single-process and single-device. Logical stages are still explicit even when they share a process and device.
- The pipeline is ordered and supports N stages. MiniMind-Omni is the first complete real pipeline and is the primary multi-stage demo.
- The public runtime contract is a pipeline submission/generation operation that accepts a request and returns a final typed output. Internal orchestration details must not leak into model-specific loaders.
- A stage owns a model component and exposes load, unload, execute, and state/lifecycle behavior. Stage kinds include AR, diffusion, action, and audio decode.
- An `Orchestrator` owns request correlation, current-stage state, stage transitions, accumulated outputs, failure, and cancellation behavior.
- A `StageRuntime` owns materialization, lazy loading, unloading, and basic VRAM-aware lifecycle management.
- A connector boundary transports typed payloads. The minimum payload vocabulary includes token payloads, tensor/hidden-state payloads, Mimi codec-token payloads, image payloads, and audio payloads with sample-rate metadata.
- MiniMind-Omni uses three logical stages: Thinker, Talker, and Code2Wav. Thinker and Talker may share the AR execution abstraction; Code2Wav uses an audio decode abstraction.
- MiniMind-Omni post-EOS behavior is modeled as a per-request state machine: `pending → visible_eos → forced_padding(remaining) → downstream_ready → done`. The exact padding count is configurable and defaults to the reference behavior of 128 steps.
- The post-EOS state machine must execute independently for concurrent requests and must clear state after completion, failure, or cancellation.
- The Talker MTP path preserves an `active_mask` per generation step. Inactive codebook positions receive the audio padding token before the next decode step.
- vLLM is the AR runtime dependency. The project does not reimplement KV-cache management, token sampling, or the full AR executor.
- Diffusers is the image diffusion dependency. SD3.5-medium and Qwen-Image-Edit are single-stage model integrations.
- InternVLA-A1-3B is the initial VLA integration because it is a direct vLLM-Omni reference model and avoids making OpenPI/π0 the first policy integration.
- Existing Mimi/audio tooling is preferred for codec decoding. The project should expose the codec boundary and decode flow without reimplementing a full neural audio codec.
- Configuration is declarative. Pipeline topology and deployment/lifecycle settings are separate concepts, represented by `PipelineConfig` and `DeployConfig`.
- Model weights are downloaded from the Hugging Face Hub at runtime and are not committed to the repository.
- Gradio is the serving UI for local use and Hugging Face Spaces. Model loading is lazy and model-family tabs may release inactive stages when VRAM pressure requires it.
- The implementation should remain within the project's readability budget. Production optimizations are added only when they are required to make the stage contract understandable or the target demo runnable.
- The repository's existing design glossary and decisions are captured in the design grill document; MiniMind-Omni PR #3796 is the reference for post-EOS bridge states, Talker MTP, and the three-stage topology.

## Testing Decisions

- The highest test seam is the public pipeline generation/submission operation. Tests should submit a request through the same entrypoint used by the demo and observe the final typed output and request lifecycle, rather than calling private scheduler helpers.
- Unit tests will inject fake stages through the Stage contract. A fake Thinker, Talker, and Code2Wav will make the pipeline produce a deterministic audio payload without model weights.
- The primary contract test will verify that one request traverses all three stages in order, that each connector receives the expected payload type, and that the final output is a waveform with sample-rate metadata.
- A concurrency contract test will submit at least two fake MiniMind-Omni requests and verify that their post-EOS counters and bridge-state payloads remain isolated.
- A post-EOS behavior test will observe external stage calls and verify that visible EOS is followed by the configured forced-padding count before downstream readiness. The test must not depend on a private counter implementation.
- A connector contract test will verify that hidden-state tensors and codec tokens preserve shape, dtype, and metadata across the transport boundary. It should use the public connector contract and not inspect storage internals.
- Configuration tests will verify accepted stage kinds, unique stage names, valid topology, and separation of pipeline topology from deployment settings.
- Runtime lifecycle tests will verify lazy load, unload, reload, and failure cleanup using fake stage implementations.
- Real-weight smoke tests will use the same public pipeline entrypoint as the fake-stage contract tests. They will be marked as manual/slow, run one request per model family, and check output type/shape rather than benchmark latency or audio/image quality.
- CI will run CPU-safe unit tests without downloading model weights. GPU and Hugging Face Hub access are prerequisites only for manual smoke tests.
- There is no existing application test prior art in the scaffold. The testing style follows the repository's TODO convention and adapts the real-weight/E2E intent of vLLM-Omni's MiniMind-Omni PR #3796.

## Out of Scope

- Reimplementing vLLM's scheduler, KV-cache manager, attention kernels, tokenizer, or multimodal processor.
- Reimplementing diffusers' SD3.5 or Qwen-Image-Edit model internals.
- Reimplementing the full Mimi codec or training any model.
- OpenPI/π0 as the first VLA integration.
- Cross-process, cross-node, or distributed stage placement.
- Mooncake, Mori, Yuanrong, or other production connector transports.
- Full async chunk streaming for MiniMind-Omni.
- Step-level diffusion batching, CUDA graph capture, speculative decoding, quantization, multi-LoRA, and distributed layerwise offload.
- Production-grade fault recovery, autoscaling, multi-tenant isolation, and SLA guarantees.
- Making all model families resident concurrently on the 24 GB target hardware.
- Treating a generic Qwen prompt-rewrite → SD3.5 composition as the canonical vLLM-Omni topology.
- Downloading or committing model weights in the repository.
- A separate custom frontend; Gradio is sufficient for v1.

## Further Notes

- The repository is currently scaffolded and the initial commit is pushed to a private GitHub repository. This spec is the first implementation-level parent issue; child tracer tickets should be created after the spec is accepted.
- The primary demo should be MiniMind-Omni, not the earlier synthetic Qwen → SD3.5 pipeline. The synthetic composition may be documented as an optional educational experiment later, but it must not drive the core architecture.
- The 24 GB hardware target makes lazy stage lifecycle important even though MiniMind-Omni itself is small; the same runtime should be able to switch between AR, image, VLA, and audio model families.
- The implementation should preserve the distinction between model topology, stage runtime policy, and deployment placement. This is the central lesson taken from vLLM-Omni.
- The reference material is [vLLM-Omni PR #3796](https://github.com/vllm-project/vllm-omni/pull/3796), especially its Thinker → Talker → Code2Wav topology, post-EOS bridge-state handling, and Talker MTP masking behavior.
