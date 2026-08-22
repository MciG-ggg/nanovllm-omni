"""MiniMind-O bundle loader.

Wraps the HF trust_remote_code ``MiniMindOmni`` checkpoint, the
``MimiModel`` codec, and the tokenizer into a single ``MinimindBundle``
handle that the per-stage modules (``thinker.py`` / ``talker.py`` /
``code2wav.py``) consume. This module owns the loading path; per-stage
modules own the inference.

Public symbols are the same as the previous ``stages.py`` so existing
importers (``engine/runtime.py``, ``entrypoints/base.py``,
``optim/bench/runner.py``, ``models/__init__.py``) don't need their
import paths rewritten when migrating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanovllm_omni.outputs import AudioPayload

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"
DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"
MIMI_SAMPLE_RATE = 24_000
MIMI_CODE_VOCAB_LIMIT = 2048


@dataclass
class MinimindBundle:
    """Loaded MiniMind-O runtime pieces used by the aligned Omni path."""

    model: Any
    tokenizer: Any
    mimi: Any
    device: str
    model_id: str
    # Aliases kept for create_stages() callers that expect three handles.
    thinker: Any = None
    talker: Any = None
    code2wav: Any = None


def _resolve_snapshot(model_id: str) -> str:
    path = Path(model_id)
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_minimind_omni_bundle(
    model_id: str = DEFAULT_MINIMIND_MODEL_ID,
    device: str | None = None,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
    **kwargs: Any,
) -> MinimindBundle:
    """Load MiniMind-O + tokenizer + Mimi onto ``device``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    device = _pick_device(device)
    snapshot_dir = _resolve_snapshot(model_id)
    mimi_dir = _resolve_snapshot(kwargs.get("mimi_model_id", mimi_model_id))

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, trust_remote_code=True).eval()
    if device != "cpu":
        model = model.half()
    model = model.to(device)

    mimi = MimiModel.from_pretrained(mimi_dir).eval()
    if device != "cpu":
        mimi = mimi.half()
    mimi = mimi.to(device)
    # Official eval_omni attaches mimi on the model for decode convenience.
    model.mimi_model = mimi

    return MinimindBundle(
        model=model,
        tokenizer=tokenizer,
        mimi=mimi,
        device=device,
        model_id=model_id,
        thinker=model,
        talker=getattr(model, "talker", None),
        code2wav=mimi,
    )


def create_bundle(model_id: str, device: str | None = None, **kwargs: Any) -> MinimindBundle:
    return load_minimind_omni_bundle(model_id=model_id, device=device, **kwargs)


def create_stages(model_id: str, device: str | None = None, **kwargs: Any):
    bundle = create_bundle(model_id, device, **kwargs)
    return bundle.thinker, bundle.talker, bundle.code2wav


# Re-exported for convenience; the canonical home for AudioPayload is
# ``nanovllm_omni.outputs`` but the previous ``stages.py`` exposed it here
# too, so keep the surface stable.
__all__ = [
    "AudioPayload",
    "DEFAULT_MINIMIND_MODEL_ID",
    "DEFAULT_MIMI_MODEL_ID",
    "MIMI_SAMPLE_RATE",
    "MIMI_CODE_VOCAB_LIMIT",
    "MinimindBundle",
    "create_bundle",
    "create_stages",
    "load_minimind_omni_bundle",
]
