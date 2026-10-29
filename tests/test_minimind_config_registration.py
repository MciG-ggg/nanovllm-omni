"""The local ``minimind-o`` config class replaces the hub's ``auto_map``.

The MiniMind-3o checkpoint's ``config.json`` points ``auto_map.AutoConfig``
at ``model_omni.OmniConfig`` in the hub repo, so ``AutoConfig`` would
execute that file. The repo implements the thinker and the talker itself,
so it registers its own config class and the hub file is never imported.
"""

from __future__ import annotations

import json
import sys

import pytest
from transformers import AutoConfig

from nanovllm_omni.engine.stage_runner import get_stage_config
from nanovllm_omni.models.minimind_omni.hf_config import (
    MiniMindOConfig,
    register_minimind_config,
)

_CONFIG = {
    "model_type": "minimind-o",
    "auto_map": {"AutoConfig": "model_omni.OmniConfig"},
    "hidden_size": 768,
    "vocab_size": 6400,
    "num_hidden_layers": 8,
    "audio_vocab_size": 2112,
    "talker_hidden_size": 768,
    "bridge_layer": 3,
    "dtype": "bfloat16",
}

# The hub file must never run: it raises when it does.
_HUB_FILE = "raise RuntimeError('the hub config file was executed')\n"


def _write_checkpoint(directory) -> None:
    (directory / "config.json").write_text(json.dumps(_CONFIG))
    (directory / "model_omni.py").write_text(_HUB_FILE)


def test_local_config_class_replaces_the_auto_map(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    register_minimind_config()

    config = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=True)

    assert isinstance(config, MiniMindOConfig)
    assert config.hidden_size == 768
    assert config.audio_vocab_size == 2112
    assert config.tie_word_embeddings is True
    assert not [name for name in sys.modules if "model_omni" in name]


def test_fork_config_reads_the_local_config_class(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    register_minimind_config()

    config = get_stage_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config.hf_config, MiniMindOConfig)
    assert config.hf_config.tie_word_embeddings is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
