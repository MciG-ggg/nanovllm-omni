"""Local config class for the MiniMind-3o checkpoint.

The checkpoint's ``config.json`` carries an ``auto_map`` that points at
``model_omni.OmniConfig`` in the hub repo. ``AutoConfig.from_pretrained``
would then execute that file. This repo implements the thinker and the
talker itself, so the only thing the hub code provides is the config
class — and that class is a plain ``PretrainedConfig`` subclass.

Registering ``minimind-o`` locally makes transformers use the registered
class and skip the hub file: the config loads with
``trust_remote_code=False`` and no remote code runs.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class MiniMindOConfig(PretrainedConfig):
    """Config for ``model_type: minimind-o``.

    Every field the checkpoint needs is in its ``config.json``; the
    only default the vendor class adds is the tied embedding.
    """

    model_type = "minimind-o"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # minimind-3o ties the input embedding to the output head and
        # omits the flag from ``config.json``; the vendor class defaults
        # it to True.
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)


def register_minimind_config() -> None:
    """Register ``MiniMindOConfig`` under ``minimind-o``.

    Idempotent: the second call is a no-op.
    """
    AutoConfig.register("minimind-o", MiniMindOConfig, exist_ok=True)


__all__ = ["MiniMindOConfig", "register_minimind_config"]
