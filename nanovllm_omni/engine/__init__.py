"""Internal pipeline runner for the aligned omni seam.

Owns the per-request ``PipelineRunner`` and its async ``PipelineExecutor``.
Model-specific execution modules (CUDA Graph adapters, KV pools, schedulers,
attention adapters) live next to the model family in
``nanovllm_omni.models.<family>``; this package only contains the
pipeline-level driver that walks ``StageConfig`` instances in order.

No public re-exports are provided here. Import leaf modules directly; public
contract types live in ``nanovllm_omni.config.params`` and the package root.
"""
