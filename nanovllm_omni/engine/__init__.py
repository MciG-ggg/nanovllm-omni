"""Engine package: real modules only (runner, executor, sched, runtime, ...).

No public re-exports here. Internal callers import leaf modules directly
(``nanovllm_omni.engine.runner``, ``nanovllm_omni.engine.executor``), and the
public contract types (``SamplingParams`` / ``OmniEngineArgs``) live in
``nanovllm_omni.config.params``. This mirrors vllm-omni, where engine
transport types live in their own modules rather than being re-exported at
the package root.
"""
