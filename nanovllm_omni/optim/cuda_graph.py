"""CUDA Graph capture for capture-safe sub-paths in nanovllm-omni.

The main ``MiniMindOmni.forward`` is NOT capture-safe on this model --
its forward contains:

* a data-dependent ``if self.thinker.freqs_cos[0, 0] == 0:`` check
  (forces a CUDA sync to read the tensor scalar; rejected by
  ``torch.cuda.graph``);
* a ``sum(l.mlp.aux_loss for l in ...)`` MOE aux-loss accumulation
  whose Python generator produces a dynamic allocation pattern;
* a ``out.audio_logits`` list of 8 tensors stored as a dynamic-shape
  attribute on a HF output container.

All three fail with ``cudaErrorStreamCaptureInvalidated``. vllm-omni
PR #3796 confirms the same blocker for their MiniMind-O integration
("Cannot copy between CPU and CUDA tensors during CUDA graph
capture") and uses ``enforce_eager=True`` for the main forward.

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
    """

    def __init__(self, mimi: Any) -> None:
        self._mimi = mimi
        self._slots: dict[tuple[int, ...], _MimiGraphSlot] = {}

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
            if slot is None:
                return None
        slot.input_buf.copy_(codes)
        try:
            slot.graph.replay()
        except Exception as exc:  # noqa: BLE001
            _log.warning("graph replay failed for shape=%s: %s", key, exc)
            return None
        return slot.output_buf.clone()

    def _capture(
        self,
        codes: torch.Tensor,
        key: tuple[int, ...],
    ) -> _MimiGraphSlot | None:
        # CPU-only hosts have no CUDA; the wrapper falls back to eager.
        if not torch.cuda.is_available():
            _log.info("skipping mimi.decode CUDA graph capture on CPU-only host")
            return None

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
            return None

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
        self._capture_failures: int = 0

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
            "capture_failures": self._capture_failures,
            "graphs_captured": len(self._cache._slots),
        }


def graph_compile_mimi(mimi: Any) -> Any:
    """Wrap ``mimi`` so ``decode(codes)`` replays via CUDA Graph.

    Idempotent: wrapping an already-graphed mimi is a no-op.
    """
    if getattr(mimi, _GRAPH_MARKER, False):
        return mimi
    return GraphedMimi(mimi)
