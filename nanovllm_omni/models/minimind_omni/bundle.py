"""MiniMind-O bundle loader.

Wraps the jingyaogong/minimind-3o checkpoint, the
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
    # Aliases kept for create_stages() callers that expect three handles.
    thinker: Any = None
    talker: Any = None
    code2wav: Any = None


def _resolve_snapshot(model_id: str) -> str:
    """Resolve a model identifier to a local directory, offline-first.

    Mirrors vllm-omni's ``_resolve_model_to_local_path``
    (``vllm_omni/engine/stage_init_utils.py``): if ``model_id`` is already
    a local directory it is used as-is; otherwise we look it up in the
    HuggingFace local cache via ``snapshot_download(local_files_only=True)``
    and never trigger a network download. Unresolvable Hub ids (no local
    cache, no network) are passed through unchanged so the caller can
    surface a clearer error from ``from_pretrained`` itself.
    """
    path = Path(model_id)
    if path.is_dir():
        return str(path.resolve())

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return snapshot_download(model_id, local_files_only=True)
    except (OSError, LocalEntryNotFoundError) as exc:
        # LocalEntryNotFoundError == "no local cache entry for this id";
        # OSError == broken local cache. Anything else (e.g. a metadata
        # format change) should propagate so the caller sees the real cause.
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

    ``dtype=None`` keeps the legacy ``.half()`` behavior unchanged so
    existing callers are bit-for-bit equivalent. SmolVLA's int8/qint8
    quantization path does NOT flow through here -- SmolVLA's stage
    factory handles its own dtypes.
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

    ``trust_remote_code`` and ``dtype`` are plumbed from
    ``OmniEngineArgs`` (effective-field matrix); both default to the legacy
    behavior so existing callers do not need to change.
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
    from .attention import (
        enable_fused_projections,
        enable_fused_rmsnorm,
        enable_fused_rope,
        enable_sdpa_decode,
    )

    for _patch in (
        enable_sdpa_decode,
        enable_fused_rmsnorm,
        enable_fused_projections,
        enable_fused_rope,
    ):
        _patch(model)

    mimi = MimiModel.from_pretrained(mimi_dir).eval()
    mimi = _cast_model_dtype(mimi, dtype, device)
    mimi = mimi.to(device)
    # Official eval_omni attaches mimi on the model for decode convenience.
    model.mimi_model = mimi

    # Wrap the vendored HF ``TalkerModule`` (option (a) from TICKET-05):
    # consumers reading ``bundle.talker`` get our LLM_AR-shaped class
    # instead of the raw HF module. ``bundle.model.talker`` still exposes
    # the raw module for callers that need it. The wrapper is built from
    # a temporary MinimalBundle so it can read ``bundle.model.config``
    # without the recursive dataclass cycle this would create if we built
    # the bundle first.
    talker_wrapped = None
    if getattr(model, "talker", None) is not None:
        from .talker import wrap_talker

        talker_wrapped = wrap_talker(
            MinimindBundle(
                model=model,
                tokenizer=tokenizer,
                mimi=mimi,
                device=device,
                model_id=model_id,
                thinker=model,
                talker=getattr(model, "talker", None),
                code2wav=mimi,
            )
        )
    return MinimindBundle(
        model=model,
        tokenizer=tokenizer,
        mimi=mimi,
        device=device,
        model_id=model_id,
        thinker=model,
        talker=talker_wrapped,
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
