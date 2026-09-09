"""MiniMind-O bundle loader.

Loads the jingyaogong/minimind-3o checkpoint + Mimi codec + tokenizer
into a ``MinimindBundle`` for the per-stage modules. Public symbols:
``load_minimind_omni_bundle`` (single canonical name; the old ``create_bundle`` / ``create_stages`` aliases were removed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanovllm_omni.outputs import AudioPayload

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"
DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"
MIMI_SAMPLE_RATE = 24_000
MIMI_CODE_VOCAB_LIMIT = 2048

_logger = logging.getLogger(__name__)


@dataclass
class MinimindBundle:
    """Loaded MiniMind-O runtime pieces used by the aligned Omni path."""

    model: Any
    tokenizer: Any
    mimi: Any
    device: str
    model_id: str
    # Aliases removed: create_bundle / create_stages -> load_minimind_omni_bundle.
    thinker: Any = None
    talker: Any = None
    code2wav: Any = None


def _resolve_snapshot(model_id: str) -> str:
    """Resolve a model identifier to a local directory, offline-first.

    Local directory passed through; otherwise lookup in the HF local cache
    via ``snapshot_download(local_files_only=True)``. Unresolvable Hub ids
    are returned unchanged so the caller can surface a clearer error from
    ``from_pretrained`` itself.
    """
    path = Path(model_id)
    if path.is_dir():
        return str(path.resolve())

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return snapshot_download(model_id, local_files_only=True)
    except (OSError, LocalEntryNotFoundError) as exc:
        _logger.warning(
            "[bundle] Could not resolve %s to a local snapshot (%s: %s); "
            "passing through unchanged. Pass --model /path/to/local/dir "
            "for offline use.",
            model_id,
            type(exc).__name__,
            exc,
        )
        return model_id


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cast_model_dtype(model: Any, dtype: str | None, device: str) -> Any:
    """Cast ``model`` to ``dtype``; no-op on cpu regardless of dtype.

    ``dtype=None`` falls back to ``.half()``.
    """
    if device == "cpu":
        return model
    if dtype is None or dtype == "float16":
        return model.half()
    if dtype == "bfloat16":
        return model.bfloat16()
    if dtype == "float32":
        return model.float()
    raise ValueError(f"unsupported dtype {dtype!r}; expected None, float16, bfloat16, or float32")


def load_minimind_omni_bundle(
    model_id: str = DEFAULT_MINIMIND_MODEL_ID,
    device: str | None = None,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
    trust_remote_code: bool = True,
    dtype: str | None = None,
    **kwargs: Any,
) -> MinimindBundle:
    """Load MiniMind-O + tokenizer + Mimi onto ``device``.

    ``trust_remote_code`` and ``dtype`` are plumbed from ``OmniEngineArgs``.
    CUDA Graph is gated independently by the deploy/CLI flags.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    device = _pick_device(device)
    snapshot_dir = _resolve_snapshot(model_id)
    mimi_dir = _resolve_snapshot(kwargs.get("mimi_model_id", mimi_model_id))

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, trust_remote_code=trust_remote_code
    ).eval()
    model = _cast_model_dtype(model, dtype, device)
    model = model.to(device)

    mimi = MimiModel.from_pretrained(mimi_dir).eval()
    mimi = _cast_model_dtype(mimi, dtype, device)
    mimi = mimi.to(device)
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


# AudioPayload re-exported for convenience; canonical home is
# ``nanovllm_omni.outputs``.
__all__ = [
    "AudioPayload",
    "DEFAULT_MINIMIND_MODEL_ID",
    "DEFAULT_MIMI_MODEL_ID",
    "MIMI_SAMPLE_RATE",
    "MIMI_CODE_VOCAB_LIMIT",
    "MinimindBundle",
    "load_minimind_omni_bundle",
]
