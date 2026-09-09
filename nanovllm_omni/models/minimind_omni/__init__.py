"""MiniMind-Omni model package exports.

MiniMindThinker / MiniMindTalker are lazy: they import nanovllm.layers
which requires triton (Linux only). Access via module-level __getattr__.
"""

from .bundle import (
    DEFAULT_MIMI_MODEL_ID,
    DEFAULT_MINIMIND_MODEL_ID,
    MIMI_SAMPLE_RATE,
    MinimindBundle,
    load_minimind_omni_bundle,
)

_LAZY_IMPORTS = {
    "MiniMindTalker": (".talker", "MiniMindTalker"),
    "MiniMindThinker": (".thinker", "MiniMindThinker"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MINIMIND_MODEL_ID",
    "DEFAULT_MIMI_MODEL_ID",
    "MIMI_SAMPLE_RATE",
    "MinimindBundle",
    "MiniMindThinker",
    "MiniMindTalker",
    "load_minimind_omni_bundle",
]
