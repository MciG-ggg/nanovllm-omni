"""The MLX entry point for the MiniMind-O thinker decode.

The session itself lives in ``stage_runner``; this module is the one-shot
drain the MLX tools and tests call.

``mlx`` is imported inside the functions, not at module level: the Linux CI
job has no ``mlx``, and the contract tests in ``tests/test_mlx_backend.py``
import this module to stub ``mlx.core`` out. Same convention as
``backend/mlx_runner.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .stage_runner import (
    ThinkerDecode,
    ThinkerDecoded,
    ThinkerDecodeSession,
    _drain_session,
)


def _mx_to_torch(array: Any) -> Any:
    """An MLX array as a float32 torch tensor."""
    import mlx.core as mx

    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.astype(np.float32, copy=False))
    return torch.from_numpy(np.asarray(array.astype(mx.float32)))


def decode_minimind_mlx(
    stage_runner: Any,
    requests: list[ThinkerDecode],
    *,
    eos_id: int | None = None,
    enter_token: int = 0,
    pad_token: int = 0,
    talker_tail: int = 8,
) -> list[ThinkerDecoded]:
    """Decode requests with the fork Scheduler and MLX cache rows.

    One-shot drain of ``ThinkerDecodeSession``. Sampling matches the CUDA
    thinker's temperature, repetition penalty, top-p, and per-request RNG
    keys. The bridge rows come from a full-sequence re-run per request,
    the same way the CUDA track gets them.
    """
    if not requests:
        return []
    session = ThinkerDecodeSession(
        stage_runner,
        backend="mlx",
        eos_id=eos_id,
        enter_token=enter_token,
        pad_token=pad_token,
        talker_tail=talker_tail,
    )
    return _drain_session(session, requests)


__all__ = ["decode_minimind_mlx"]
