# ADR 0004: Separate Model, Pipeline, and Stage Registries

- Status: Accepted
- Date: 2026-09-15

## Context

vllm uses a model registry to map Hugging Face architecture names to lazy
model classes. vllm-omni adds a separate pipeline registry because one omni
model can contain several execution stages. Model-specific forward logic stays
in model classes, while the general engine owns scheduling, cache management,
and execution plumbing.

nanovllm-omni has the same conceptual split, but its stages also have local
wrappers for tokenization, decoding, bridge capture, and diffusion/codec
runtime behavior. A single pipeline registry cannot express the distinction
between an independently loadable model class and its stage wrapper without
coupling the two concepts.

## Decision

Use three declaration layers:

1. `OMNI_MODELS` maps stable stage architecture names to lazy
   `(folder, module, class)` registrations.
2. `OMNI_PIPELINES` maps model handles to a pipeline topology or a callable
   resolver. A resolver may select a variant from the loaded HF config.
3. `StageConfig` declares both `model_architecture` and `stage_factory`.
   The model registry supplies the `nn.Module` class; the stage factory builds
   the wrapper that owns model-family runtime behavior.

The generic runner resolves both registrations lazily, injects the model class
into the wrapper, invokes processors declared by the pipeline, and adapts
terminal output by output type. It must not branch on a concrete model name.

Model-specific structure and optimizations remain in the model family. Shared
Attention/Linear/RoPE primitives and backend selection remain reusable engine
capabilities. A model-specific optimization is promoted only after a second
real consumer exists and profiling demonstrates that the shared abstraction is
worth its maintenance cost.

MiniMind-O's `enable_audio_output` config field selects the full or
thinker-only topology. CODEC remains an execution type in the topology and is
not added to the language-model registry.

## Consequences

- A stage model can be resolved and benchmarked without pretending that the
  full omni pipeline is one neural-network class.
- Stage wrappers may keep model-family decode and payload logic without
  contaminating the general runner with model-name branches.
- Existing checkpoints without `enable_audio_output` remain full pipelines by
  default; a local config can select thinker-only mode.
- The old top-level `nanovllm_omni.models` MiniMindBundle exports are removed.
- Full and thinker-only MiniMind paths require both fake-config tests and a
  real-weight WSL RTX 3050 validation before this migration is complete.
