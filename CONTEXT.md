# Omni Inference Domain

This glossary defines the concepts used to describe model declaration, staged inference, and the boundary between reusable engine behavior and model-family behavior.

## Declaration

**Model Architecture**:
The stable name used to identify one independently loadable neural-network stage.
_Avoid_: Model family, pipeline name

**Pipeline**:
An ordered inference topology that connects stages and declares the terminal output.
_Avoid_: Model class, engine

**Stage**:
One executable unit in a pipeline, such as a thinker, talker, diffusion worker, or codec decoder.
_Avoid_: Model architecture

**Variant**:
A pipeline topology selected from the same model handle by a property of the loaded model configuration.
_Avoid_: Runtime mode, request option

## Boundaries

**General Engine**:
The reusable runtime that schedules stage calls, applies resource defaults, invokes declared processors, and adapts terminal outputs.
_Avoid_: Model implementation, model runner

**Model-Specific Optimization**:
A computation or data-layout choice required by one model architecture, including its special forward behavior and tensor shapes.
_Avoid_: Engine optimization

**Stage Processor**:
The model-family-owned transformation that converts one stage's output payload into the next stage's input payload.
_Avoid_: Engine adapter, model registry

**Execution Type**:
The closed semantic category that selects a stage's generic runtime path, such as autoregressive language modeling, diffusion, or codec decoding.
_Avoid_: Model name, architecture name

**Terminal Output**:
The payload produced by the selected final stage and normalized by its declared output type.
_Avoid_: Intermediate payload
