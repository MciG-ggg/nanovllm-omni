"""``torch.compile`` + bitsandbytes int8 stretch (TK-015).

The plain ``run_generate`` path is unchanged; the bench CLI gates these
optimisations on ``--compile`` / ``--int8``. The compile cost is paid on
the first call (absorbed by the bench warmup), and subsequent calls
reuse the compiled artifact.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

_log = logging.getLogger(__name__)

# Ponytail: torch.compile may raise on unsupported ops or shapes; the
# bench harness needs to fall back to the eager model, not crash. The CLI
# reports the actual path taken via the compile-status log line.

_COMPILE_MARKER = "_nano_vllm_compiled_v1"


def _is_already_compiled(model: Any) -> bool:
    return bool(getattr(model, _COMPILE_MARKER, False))


def _mark_compiled(model: Any) -> None:
    with contextlib.suppress(AttributeError, TypeError):
        setattr(model, _COMPILE_MARKER, True)


def compile_model(
    model: torch.nn.Module,
    *,
    mode: str = "default",
    device: str = "cuda",
) -> torch.nn.Module:
    """Wrap ``model.forward`` with ``torch.compile``.

    On ``device == "cuda"`` we try ``mode="reduce-overhead"`` first
    (CUDA Graphs under the hood; biggest win for repeated forward
    calls with stable shapes). On MPS/CPU we fall back to
    ``mode="default"``. Any ``RuntimeError`` from torch.compile returns
    the original model so the bench can still run.
    """
    if _is_already_compiled(model):
        return model

    effective_mode = mode
    if device == "cuda" and effective_mode == "default":
        effective_mode = "reduce-overhead"
    if device not in ("cuda",) and effective_mode == "reduce-overhead":
        # reduce-overhead requires CUDA Graphs; fall back on CPU/MPS.
        effective_mode = "default"

    try:
        compiled = torch.compile(model, mode=effective_mode)
    except (RuntimeError, ValueError) as exc:
        # ponytail: torch.compile unsupported, returning eager model.
        _log.warning(
            "torch.compile(mode=%r) refused: %s: %s -- returning eager model",
            effective_mode,
            type(exc).__name__,
            exc,
        )
        return model
    except Exception as exc:
        _log.warning(
            "torch.compile(mode=%r) raised unexpected %s: %s -- returning eager model",
            effective_mode,
            type(exc).__name__,
            exc,
        )
        return model

    _mark_compiled(compiled)
    _log.info("compiled model with mode=%r", effective_mode)
    return compiled


def compile_thinker(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Convenience wrapper -- ``reduce-overhead`` on CUDA, ``default`` elsewhere."""
    mode = "reduce-overhead" if device == "cuda" else "default"
    return compile_model(model, mode=mode, device=device)


# Optional bitsandbytes int8 stretch -- only callable when bnb is installed.
try:
    import bitsandbytes as bnb  # type: ignore[import-not-found]

    _HAS_BNB = True
except ImportError:
    bnb = None  # type: ignore[assignment]
    _HAS_BNB = False


def has_bnb() -> bool:
    """True iff bitsandbytes was importable when this module loaded."""
    return _HAS_BNB


def quantize_thinker_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Replace MLP ``nn.Linear`` layers with ``bnb.nn.Linear8bitLt``.

    Only meaningful when :func:`has_bnb` returns True; otherwise raises
    ``RuntimeError``. Heuristic per spec: target layers whose qualified
    name contains ``"mlp"`` or ``"feed_forward"``. VRAM check is the
    caller's responsibility -- on a 4 GB GPU the MiniMind thinker fits
    comfortably in int8 (~250 MiB) so we don't bother to gate here.
    """
    if not _HAS_BNB:
        raise RuntimeError(
            "bitsandbytes not installed; install with "
            "`uv pip install bitsandbytes` to enable int8 quantisation"
        )

    import torch.nn as nn

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        if "mlp" not in lname and "feed_forward" not in lname:
            continue
        weight = module.weight.data
        new_layer = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
        )
        new_layer.weight = bnb.nn.Int8Params(
            weight.clone(), requires_grad=False, has_fp16_weights=False
        )
        if module.bias is not None:
            new_layer.bias = module.bias
        # Replace on parent module (setattr mutates the model in place).
        if "." in name:
            parent_path, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_path)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, new_layer)
        replaced += 1

    _log.info("quantize_thinker_int8: replaced %d Linear layer(s) with int8", replaced)
    return model
