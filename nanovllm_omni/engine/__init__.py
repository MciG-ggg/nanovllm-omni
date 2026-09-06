"""Engine package: real modules only (runner, executor, sched, runtime, ...).

No public re-exports here. Internal callers import leaf modules directly
(the runner and executor modules in this package), and the
public contract types (``SamplingParams`` / ``OmniEngineArgs``) live in
the config package's params module. This mirrors the reference, where
engine transport types live in their own modules rather than being
re-exported at the package root.
"""
