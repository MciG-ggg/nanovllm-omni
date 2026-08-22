"""CUDA Graph capture for MiniMind-O's per-step forward (TK-011 followup).

The bench profile-detail on session-1.md showed ``generate`` fires
~17 000 ``cudaLaunchKernel`` calls per second of wall time, but only
~28 ms of those are real ``cutlass`` matmul work -- the other ~510 ms
is per-kernel launch + dispatch overhead. ``torch.compile`` (TK-015)
cannot reach this: it traces ``forward``, not the Python generator's
``yield`` boundary, so the per-step launch overhead stays in eager
mode.

The real lever is CUDA Graph capture of the per-step forward call.
Each AR step inside ``model.stream_generate`` runs
``self.forward(input_ids[...], past_key_values=past_kvs, ...)`` where
``past_kvs`` grows by one position per step. CUDA Graphs need static
shapes, so we capture one graph **per past_kvs length** (1 graph for
prompt-length 0, 1 for length 1, ..., N-1 for length N-1) and replay
the right one each step.

The first call (the full-prompt forward where ``past_kvs is None``)
always goes through eager -- its shape depends on the user prompt and
capturing it is not worth the complexity. Only the incremental calls
land in a graph.

This module exposes:

* :class:`StepGraphCache` -- the capture/replay engine.
* :class:`GraphedMiniMindOmni` -- a thin wrapper that routes incremental
  forward calls through the cache and falls back to eager when the
  graph for that ``past_len`` is not yet captured.
* :func:`graph_compile_model` -- entry point used by the bench CLI.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

import torch

_log = logging.getLogger(__name__)

# Ponytail: graph capture can fail on dynamic shapes or unsupported ops;
# fall back to eager. The CLI prints the actual path taken via the
# capture/replay log lines.
_GRAPH_MARKER = "_nano_vllm_graphed_v1"
_SIDE_STREAM_NAME = "nano_vllm_capture_stream"


@dataclass
class _GraphSlot:
    """Static buffers + the captured graph for one ``past_len``."""

    graph: torch.cuda.CUDAGraph
    input_ids_buf: torch.Tensor
    past_kvs_buf: list  # list of (k_buf, v_buf) per layer
    output_present_buf: list  # list of (k_buf, v_buf) per layer, length past_len+1
    output_logits_buf: torch.Tensor
    output_audio_logits_buf: list  # list of 8 tensors (one per audio channel)
    output_aux_loss_buf: torch.Tensor


def _cuda_side_stream() -> torch.cuda.Stream:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    return s


class StepGraphCache:
    """Per-step-length CUDA Graph cache for a MiniMind-O forward call.

    Each captured graph uses pre-allocated static input/output buffers
    keyed by ``past_len``. On replay we copy the current
    ``past_key_values`` into the static buffers, replay the graph, and
    return cloned outputs (cloned because the static output buffers are
    overwritten on the next call).
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._slots: dict[int, _GraphSlot] = {}

    def has(self, past_len: int) -> bool:
        return past_len in self._slots

    def capture(
        self,
        past_len: int,
        sample_input_ids: torch.Tensor,
        sample_past_kvs: list,
        **forward_kwargs: Any,
    ) -> None:
        """Capture one graph for ``past_len``.

        ``sample_input_ids`` and ``sample_past_kvs`` are the actual tensors
        passed to ``model.forward`` on the first incremental call --
        their shapes define the captured graph's I/O.
        """
        # Static input buffers (allocated copies).
        input_ids_buf = torch.empty_like(sample_input_ids, memory_format=torch.contiguous_format)
        input_ids_buf.copy_(sample_input_ids)

        past_kvs_buf: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in sample_past_kvs:
            k_buf = torch.empty_like(k, memory_format=torch.contiguous_format)
            v_buf = torch.empty_like(v, memory_format=torch.contiguous_format)
            k_buf.copy_(k)
            v_buf.copy_(v)
            past_kvs_buf.append((k_buf, v_buf))

        # Warmup on a side stream so the first capture's JIT / cuDNN
        # benchmark don't pollute the captured graph.
        side = _cuda_side_stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self._model(
                    input_ids_buf,
                    past_key_values=past_kvs_buf,
                    **forward_kwargs,
                )
        side.synchronize()
        torch.cuda.current_stream().wait_stream(side)

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            out = self._model(
                input_ids_buf,
                past_key_values=past_kvs_buf,
                **forward_kwargs,
            )
            # Materialise output buffers we want to read on replay.
            # Clone the lists so the captured graph's tensors are kept
            # alive (the original `out.past_key_values` Python list will
            # be released when the context manager exits).
            output_present_buf = [(k.clone(), v.clone()) for k, v in out.past_key_values]
            output_logits_buf = out.logits.clone()
            output_audio_logits_buf = [t.clone() for t in out.audio_logits]
            output_aux_loss_buf = (
                out.aux_loss.clone() if out.aux_loss is not None else torch.zeros(())
            )

        # Set the side stream to wait for the default stream so the
        # next graph capture (on a different past_len) doesn't race.
        torch.cuda.current_stream().wait_stream(side)
        graph.replay()  # one initial replay to bind all pointers
        torch.cuda.synchronize()

        self._slots[past_len] = _GraphSlot(
            graph=graph,
            input_ids_buf=input_ids_buf,
            past_kvs_buf=past_kvs_buf,
            output_present_buf=output_present_buf,
            output_logits_buf=output_logits_buf,
            output_audio_logits_buf=output_audio_logits_buf,
            output_aux_loss_buf=output_aux_loss_buf,
        )
        _log.info(
            "captured CUDA graph for past_len=%d (output present shape=%s, %d layers)",
            past_len,
            output_present_buf[0][0].shape,
            len(output_present_buf),
        )

    def replay(
        self,
        past_len: int,
        input_ids: torch.Tensor,
        past_kvs: list,
        **forward_kwargs: Any,
    ) -> tuple[list, torch.Tensor, list, torch.Tensor]:
        """Replay the graph for ``past_len`` and return cloned outputs.

        Returns ``(past_key_values, logits, audio_logits, aux_loss)``.
        """
        slot = self._slots[past_len]
        # Copy inputs into the static buffers.
        slot.input_ids_buf.copy_(input_ids)
        for (k_buf, v_buf), (k, v) in zip(slot.past_kvs_buf, past_kvs, strict=True):
            k_buf.copy_(k)
            v_buf.copy_(v)
        # Replay.
        slot.graph.replay()
        # Clone outputs (the static buffers are overwritten on next call).
        present = [(k.clone(), v.clone()) for k, v in slot.output_present_buf]
        logits = slot.output_logits_buf.clone()
        audio_logits = [t.clone() for t in slot.output_audio_logits_buf]
        aux_loss = slot.output_aux_loss_buf.clone()
        return present, logits, audio_logits, aux_loss


class GraphedMiniMindOmni(torch.nn.Module):
    """Wraps a MiniMindOmni so incremental forward calls use CUDA Graphs.

    The full-prompt call (where ``past_key_values`` is None) always uses
    eager forward -- the prompt shape varies per request. Only the
    incremental calls (one new token + past_kvs of length N) get
    graphed, one graph per past_kvs length.

    The wrapper delegates everything else (attributes, sub-modules,
    ``generate``, ``stream_generate``) to the inner model. Only
    ``forward`` is intercepted.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self._inner = inner
        self._cache = StepGraphCache(inner)
        # Mark the wrapper so graph_compile_model is idempotent.
        object.__setattr__(self, _GRAPH_MARKER, True)
        # Eager fallback counter (informational).
        self._eager_calls: int = 0
        self._graph_calls: int = 0
        self._capture_failures: int = 0

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        past_kvs = kwargs.get("past_key_values")
        # Full-prompt call -- always eager.
        if past_kvs is None or past_kvs[0] is None:
            self._eager_calls += 1
            return self._inner.forward(input_ids, **kwargs)

        # MiniMind-O KV cache layout: [B, seq, n_heads, head_dim] -- the model
        # concatenates past/new along dim=1 (the seq axis). The model's own
        # ``start_pos`` is computed as ``past_key_values[0][0].shape[1]``.
        past_len = past_kvs[0][0].shape[1]
        # Strip past_key_values from kwargs; capture/replay inject it.
        fwd_kwargs = {k: v for k, v in kwargs.items() if k != "past_key_values"}
        if not self._cache.has(past_len):
            try:
                self._cache.capture(past_len, input_ids, past_kvs, **fwd_kwargs)
            except Exception as exc:  # noqa: BLE001
                # Ponytail: graph capture can fail on dynamic shapes;
                # fall back to eager rather than crash the bench.
                self._capture_failures += 1
                _log.warning(
                    "graph capture failed for past_len=%d: %s -- falling back to eager",
                    past_len,
                    exc,
                )
                return self._inner.forward(input_ids, **kwargs)

        try:
            present, logits, audio_logits, aux_loss = self._cache.replay(
                past_len, input_ids, past_kvs, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            # Replay can fail on shape / value drift; fall back to eager.
            self._capture_failures += 1
            _log.warning(
                "graph replay failed for past_len=%d: %s -- falling back to eager",
                past_len,
                exc,
            )
            return self._inner.forward(input_ids, **kwargs)

        # Reconstruct the model's output container. We use the inner
        # model's output class to stay compatible with downstream
        # ``.logits`` / ``.audio_logits`` / ``.past_key_values`` access
        # in ``stream_generate``.
        try:
            from transformers.modeling_outputs import (
                MoeCausalLMOutputWithPast,
            )

            out = MoeCausalLMOutputWithPast(
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=present,
            )
            out.audio_logits = audio_logits
        except ImportError:
            # Fall back to a plain object -- ``stream_generate`` only
            # touches ``.logits`` / ``.past_key_values`` / ``.audio_logits``.
            out = _SimpleOut(logits=logits, past_key_values=present, audio_logits=audio_logits)
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


class _SimpleOut:
    __slots__ = ("logits", "past_key_values", "audio_logits", "aux_loss")

    def __init__(self, *, logits, past_key_values, audio_logits, aux_loss=None) -> None:
        self.logits = logits
        self.past_key_values = past_key_values
        self.audio_logits = audio_logits
        self.aux_loss = aux_loss


def graph_compile_model(model: Any) -> Any:
    """Wrap ``model`` so its incremental forward calls use CUDA Graphs.

    Idempotent: wrapping an already-graphed model is a no-op.
    """
    if getattr(model, _GRAPH_MARKER, False):
        return model
    return GraphedMiniMindOmni(model)
