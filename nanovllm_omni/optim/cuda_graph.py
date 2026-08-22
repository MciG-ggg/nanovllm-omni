"""CUDA Graph capture for capture-safe sub-paths in nanovllm-omni.

The main ``MiniMindOmni.forward`` was NOT capture-safe on this model --
the blocker was a data-dependent ``if self.thinker.freqs_cos[0, 0] == 0:``
lazy-init check that reads a device tensor scalar (forces a CUDA sync;
rejected by ``torch.cuda.graph``). A ``sum(...)`` MOE aux-loss generator
and the ``out.audio_logits`` list were thought to be additional blockers,
but on the MiniMind-O checkpoint (``use_moe=False``) they capture cleanly
and replay bit-identical to eager.

TK-016 phase 3.b removed the real blocker: ``bundle.py`` now calls the
vendored model's ``materialize_rope()`` after load, and the forward guard
is a CPU-side ``getattr(self, "_rope_materialized", False)`` flag with no
device->host read. Verified on WSL: ``MiniMindOmni.forward`` captures
under ``torch.cuda.graph`` (both prefill and a use_cache decode step) and
replays bit-identical to eager, with audio.wav parity MD5 preserved.

vllm-omni PR #3796 uses ``enforce_eager=True`` for their MiniMind-O
integration; nanovllm-omni's phase 3.c wrapper will attempt capture and
fall back to eager on any failure.

The capture-safe sub-path we can graph is ``mimi.decode(codes)`` --
a single static-shape call invoked once per ``run_one`` with the
collected audio code list. The bench profile-detail on session-1.md
shows the decode stage fires ~1 000 ``cudaLaunchKernel`` calls; the
graph path collapses these into a single launch. Expected saving is
modest (decode is ~3% of total) but the same pattern vllm-omni
applies to the Talker MTP path.

This module exposes:

* :class:`MimiDecodeGraphCache` -- one CUDA graph per input shape
  (keyed by ``codes.shape``), with a persistent input buffer and
  a cloned output buffer.
* :class:`GraphedMimi` -- thin ``nn.Module`` wrapper that routes
  ``decode(codes)`` through the cache and falls back to eager on any
  capture / replay failure. All other attributes delegate to the
  inner mimi model.
* :func:`graph_compile_mimi` -- entry point used by the bench CLI.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

import torch

_log = logging.getLogger(__name__)

_GRAPH_MARKER = "_nano_vllm_mimi_graphed_v1"


@dataclass
class _MimiGraphSlot:
    """Static buffers + captured graph for one mimi input shape."""

    graph: torch.cuda.CUDAGraph
    input_buf: torch.Tensor
    output_buf: torch.Tensor  # cloned audio_values buffer


def _cuda_side_stream() -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    return s


class MimiDecodeGraphCache:
    """Per-input-shape CUDA Graph cache for ``mimi.decode``.

    Captures one graph per distinct ``codes.shape`` on first use. On
    replay, copies the live input into a persistent buffer and runs
    the captured graph, returning a clone of the static output buffer
    (the static buffer is overwritten on the next call).

    Capture failure is memoised per shape: a failed capture writes
    :data:`_FAILED` into ``_slots[key]`` so subsequent calls with the
    same shape short-circuit straight to the eager fallback instead
    of re-running the 3-warmup + sync + capture-attempt cycle. On
    ``cudaErrorStreamCaptureInvalidated`` the stream stays poisoned
    for the rest of the process (PyTorch limitation), so retrying
    has no chance of succeeding.
    """

    # ponytail: class-level sentinel; id() is stable per process, equality
    # is identity. Stored in _slots when capture fails for a shape.
    _FAILED: object = object()

    def __init__(self, mimi: Any) -> None:
        self._mimi = mimi
        # value is _MimiGraphSlot on success, _FAILED on memoised failure
        self._slots: dict[tuple[int, ...], object] = {}
        # how many times _capture wrote _FAILED into _slots (per shape,
        # so repeated calls with the same shape only count once)
        self._capture_failure_count: int = 0

    def replay(self, codes: torch.Tensor) -> torch.Tensor | None:
        """Replay the graph for ``codes.shape`` (capture on first sight).

        Returns the cloned ``audio_values`` tensor on success, or
        ``None`` on capture / replay failure (caller falls back to
        eager).
        """
        key = tuple(codes.shape)
        slot = self._slots.get(key)
        if slot is None:
            slot = self._capture(codes, key)
            # _capture writes _MimiGraphSlot or _FAILED into _slots[key]
        if slot is self._FAILED:
            return None
        slot.input_buf.copy_(codes)  # type: ignore[union-attr]
        try:
            slot.graph.replay()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            _log.warning("graph replay failed for shape=%s: %s", key, exc)
            # Replay failure is potentially transient (different from
            # capture failure which always invalidates the stream).
            # Drop the slot so the next call re-captures.
            self._slots.pop(key, None)
            return None
        return slot.output_buf.clone()  # type: ignore[union-attr]

    def _capture(
        self,
        codes: torch.Tensor,
        key: tuple[int, ...],
    ) -> object:
        # CPU-only hosts have no CUDA; the wrapper falls back to eager.
        if not torch.cuda.is_available():
            _log.info("skipping mimi.decode CUDA graph capture on CPU-only host")
            self._slots[key] = self._FAILED
            self._capture_failure_count += 1
            return self._FAILED

        # Persistent input buffer (allocated once per shape).
        input_buf = torch.empty_like(codes, memory_format=torch.contiguous_format)
        input_buf.copy_(codes)

        # Warmup on a side stream so the first capture's JIT / cuDNN
        # benchmark don't pollute the recorded graph.
        side = _cuda_side_stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self._mimi.decode(input_buf)
        side.synchronize()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=side):
                out = self._mimi.decode(input_buf)
                output_buf = out.audio_values.clone()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "graph capture failed for mimi.decode shape=%s: %s",
                key,
                exc,
            )
            torch.cuda.current_stream().wait_stream(side)
            # ponytail: stream stays poisoned after StreamCaptureInvalidated,
            # so retrying within this process has no chance of succeeding.
            # Memoising the failure avoids repeating the 3-warmup + sync
            # cost on every subsequent call with this shape (~90 ms saved
            # per call on the MiniMind-O bench, see docs/perf/session-9.md).
            self._slots[key] = self._FAILED
            self._capture_failure_count += 1
            return self._FAILED

        # Bind the side stream to the default so the next capture
        # (on a different shape) doesn't race the first one.
        torch.cuda.current_stream().wait_stream(side)
        graph.replay()  # one initial replay to bind all pointers
        torch.cuda.synchronize()

        slot = _MimiGraphSlot(graph=graph, input_buf=input_buf, output_buf=output_buf)
        self._slots[key] = slot
        _log.info(
            "captured mimi.decode CUDA graph for shape=%s (output shape=%s)",
            key,
            tuple(output_buf.shape),
        )
        return slot


class _SimpleOut:
    """Drop-in for the HF output container carrying ``audio_values``."""

    __slots__ = ("audio_values",)

    def __init__(self, *, audio_values: torch.Tensor) -> None:
        self.audio_values = audio_values


class GraphedMimi(torch.nn.Module):
    """Wraps a mimi model so ``decode(codes)`` replays via CUDA Graph.

    The wrapper delegates everything else (other sub-modules, params,
    ``encode``, ``__call__``, etc.) to the inner mimi model via
    ``__getattr__``. Only ``decode`` is intercepted. On any capture
    or replay failure the wrapper falls back to ``self._inner.decode``
    so the bench can still run.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self._inner = inner
        self._cache = MimiDecodeGraphCache(inner)
        # Mark the wrapper so ``graph_compile_mimi`` is idempotent.
        object.__setattr__(self, _GRAPH_MARKER, True)
        self._eager_calls: int = 0
        self._graph_calls: int = 0

    def decode(self, codes: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        cached = self._cache.replay(codes)
        if cached is None:
            self._eager_calls += 1
            return self._inner.decode(codes, *args, **kwargs)
        # Reconstruct the model's output container. The downstream
        # code in ``decode_audio`` only touches ``.audio_values``, so
        # a thin shim is enough; the HF ``BaseModelOutput`` is used
        # when available to stay class-compatible.
        try:
            from transformers.modeling_outputs import BaseModelOutput

            out: Any = BaseModelOutput()
            out.audio_values = cached
        except ImportError:
            out = _SimpleOut(audio_values=cached)
        self._graph_calls += 1
        return out

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the wrapped model. ``super().__getattr__``
        # raises AttributeError for our own missing attrs, so we fall
        # through to the inner model.
        with contextlib.suppress(AttributeError):
            return super().__getattr__(name)
        return getattr(self._inner, name)

    def stats(self) -> dict[str, int]:
        return {
            "eager_calls": self._eager_calls,
            "graph_calls": self._graph_calls,
            "capture_failures": self._cache._capture_failure_count,
            "graphs_captured": len(
                [k for k, v in self._cache._slots.items() if v is not self._cache._FAILED]
            ),
        }


def graph_compile_mimi(mimi: Any) -> Any:
    """Wrap ``mimi`` so ``decode(codes)`` replays via CUDA Graph.

    Idempotent: wrapping an already-graphed mimi is a no-op.
    """
    if getattr(mimi, _GRAPH_MARKER, False):
        return mimi
    return GraphedMimi(mimi)
