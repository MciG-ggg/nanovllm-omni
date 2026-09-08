"""Opt-in CUDA Graph execution for the Talker MTP seam.

The Talker wrapper already owns the MTP semantics. This module only owns the
fixed-address input buffers and graph lifecycle around that callable. Sampling
stays outside CUDA Graph capture: stochastic calls use the eager
``talker_mtp`` path with the caller's generator, while deterministic calls can
be captured and replayed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TalkerMtpGraphKey:
    """Fixed-shape identity for one captured Talker MTP graph."""

    batch_size: int
    input_ids_shape: tuple[int, ...]
    input_embeds_shape: tuple[int, ...]
    last_talker_hidden_shape: tuple[int, ...]
    text_step_shape: tuple[int, ...]
    active_mask_shape: tuple[int, ...] | None
    device: str
    input_ids_dtype: torch.dtype
    input_embeds_dtype: torch.dtype
    last_talker_hidden_dtype: torch.dtype
    text_step_dtype: torch.dtype
    active_mask_dtype: torch.dtype | None


@dataclass
class _TalkerMtpGraph:
    graph: Any
    buffers: dict[str, torch.Tensor]
    output: torch.Tensor


class TalkerMtpCudaGraph:
    """Capture/replay deterministic Talker MTP calls per fixed input shape.

    ``decode`` has the same argument contract as ``talker_mtp``. It falls back
    to the original callable for CPU/MPS inputs, unsupported batch sizes,
    stochastic sampling, and graph failures. Each cache entry owns all five
    input buffers, including ``active_mask`` when supplied, so replay never
    changes a caller-owned tensor address.
    """

    _BUFFER_NAMES = (
        "input_ids",
        "input_embeds",
        "last_talker_hidden",
        "text_step",
        "active_mask",
    )

    def __init__(
        self,
        talker: Any,
        *,
        batch_sizes: int | Iterable[int] = (1,),
        temperature: float = 0.2,
        top_k: int = 50,
    ) -> None:
        if not callable(getattr(talker, "talker_mtp", None)):
            raise TypeError("talker must provide a callable talker_mtp method")
        if isinstance(batch_sizes, int):
            batch_sizes = (batch_sizes,)
        self.talker = talker
        self.batch_sizes = frozenset(int(size) for size in batch_sizes)
        if not self.batch_sizes or min(self.batch_sizes) < 1:
            raise ValueError("batch_sizes must contain positive integers")
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self._cache: dict[TalkerMtpGraphKey, _TalkerMtpGraph] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _batch_size(self, input_ids: torch.Tensor) -> int:
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            return int(input_ids.shape[0])
        raise ValueError("input_ids must have shape [batch_size] or [batch_size, 1]")

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> int:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (input_ids, input_embeds, last_talker_hidden, text_step)
        ):
            raise TypeError("Talker MTP inputs must be torch.Tensor values")
        batch_size = self._batch_size(input_ids)
        for name, value in (
            ("input_embeds", input_embeds),
            ("last_talker_hidden", last_talker_hidden),
            ("text_step", text_step),
        ):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"{name} must have first dimension {batch_size}")
        if active_mask is not None and (
            not isinstance(active_mask, torch.Tensor)
            or active_mask.ndim == 0
            or (active_mask.ndim == 1 and batch_size != 1)
            or (active_mask.ndim >= 2 and active_mask.shape[0] != batch_size)
        ):
            raise ValueError("active_mask must have one row per batch item")
        return batch_size

    def _cache_key(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> TalkerMtpGraphKey:
        batch_size = self._validate_inputs(
            input_ids, input_embeds, last_talker_hidden, text_step, active_mask
        )
        return TalkerMtpGraphKey(
            batch_size=batch_size,
            input_ids_shape=tuple(input_ids.shape),
            input_embeds_shape=tuple(input_embeds.shape),
            last_talker_hidden_shape=tuple(last_talker_hidden.shape),
            text_step_shape=tuple(text_step.shape),
            active_mask_shape=None if active_mask is None else tuple(active_mask.shape),
            device=str(input_ids.device),
            input_ids_dtype=input_ids.dtype,
            input_embeds_dtype=input_embeds.dtype,
            last_talker_hidden_dtype=last_talker_hidden.dtype,
            text_step_dtype=text_step.dtype,
            active_mask_dtype=None if active_mask is None else active_mask.dtype,
        )

    def _invoke(
        self,
        buffers: dict[str, torch.Tensor],
        *,
        do_sample: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return self.talker.talker_mtp(
            buffers["input_ids"],
            buffers["input_embeds"],
            buffers["last_talker_hidden"],
            buffers["text_step"],
            active_mask=buffers["active_mask"],
            temperature=self.temperature,
            top_k=self.top_k,
            do_sample=do_sample,
            generator=generator,
        )

    def _graph_safe(
        self,
        key: TalkerMtpGraphKey,
        tensors: tuple[torch.Tensor, ...],
    ) -> bool:
        return (
            torch.cuda.is_available()
            and key.batch_size in self.batch_sizes
            and all(tensor.device.type == "cuda" for tensor in tensors)
            and len({str(tensor.device) for tensor in tensors}) == 1
        )

    def _capture(
        self,
        key: TalkerMtpGraphKey,
        inputs: dict[str, torch.Tensor],
    ) -> _TalkerMtpGraph:
        if inputs["active_mask"] is None:
            raise ValueError("CUDA Graph Talker MTP requires an active_mask tensor")
        buffers = {name: value.clone() for name, value in inputs.items()}
        device = buffers["input_ids"].device
        graph = torch.cuda.CUDAGraph()
        current_stream = torch.cuda.current_stream(device=device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            self._invoke(buffers, do_sample=False, generator=None)
        current_stream.wait_stream(warmup_stream)
        # No ``stream=`` arg: PyTorch allocates an internal capture stream and
        # joins back, which works whether the caller is on the default stream
        # or a side stream. Passing ``stream=current_stream`` would fail when
        # current_stream IS the default stream.
        with torch.cuda.graph(graph), torch.inference_mode():
            output = self._invoke(buffers, do_sample=False, generator=None)
        current_stream.synchronize()
        if not isinstance(output, torch.Tensor):
            raise TypeError("talker_mtp must return a torch.Tensor for CUDA Graph replay")
        return _TalkerMtpGraph(graph=graph, buffers=buffers, output=output)

    def invalidate(self, key: TalkerMtpGraphKey | None = None) -> None:
        """Drop one fixed-shape graph, or all graphs when ``key`` is None."""
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def decode(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run Talker MTP, using a graph only for deterministic CUDA calls."""
        key = self._cache_key(input_ids, input_embeds, last_talker_hidden, text_step, active_mask)
        tensors = (input_ids, input_embeds, last_talker_hidden, text_step)
        if active_mask is not None:
            tensors += (active_mask,)
        if do_sample or active_mask is None or not self._graph_safe(key, tensors):
            return self._invoke(
                {
                    "input_ids": input_ids,
                    "input_embeds": input_embeds,
                    "last_talker_hidden": last_talker_hidden,
                    "text_step": text_step,
                    "active_mask": active_mask,
                },
                do_sample=do_sample,
                generator=generator,
            )

        entry = self._cache.get(key)
        try:
            if entry is None:
                entry = self._capture(
                    key,
                    {
                        "input_ids": input_ids,
                        "input_embeds": input_embeds,
                        "last_talker_hidden": last_talker_hidden,
                        "text_step": text_step,
                        "active_mask": active_mask,
                    },
                )
                self._cache[key] = entry
            for name in self._BUFFER_NAMES:
                if entry.buffers[name] is not None:
                    entry.buffers[name].copy_(
                        {
                            "input_ids": input_ids,
                            "input_embeds": input_embeds,
                            "last_talker_hidden": last_talker_hidden,
                            "text_step": text_step,
                            "active_mask": active_mask,
                        }[name]
                    )
            entry.graph.replay()
            return entry.output.clone()
        except RuntimeError as exc:
            self.invalidate(key)
            _log.warning("Talker MTP CUDA Graph failed; using eager path: %s", exc)
            return self._invoke(
                {
                    "input_ids": input_ids,
                    "input_embeds": input_embeds,
                    "last_talker_hidden": last_talker_hidden,
                    "text_step": text_step,
                    "active_mask": active_mask,
                },
                do_sample=False,
                generator=generator,
            )

    __call__ = decode


CudaGraphTalkerDecoder = TalkerMtpCudaGraph


def enable_talker_mtp_cuda_graph(
    talker: Any,
    *,
    batch_sizes: int | Iterable[int] = (1,),
    temperature: float = 0.2,
    top_k: int = 50,
) -> TalkerMtpCudaGraph | None:
    """Return an opt-in Talker graph decoder, or ``None`` for eager fallback."""
    if not callable(getattr(talker, "talker_mtp", None)):
        _log.warning("enable_talker_mtp_cuda_graph: talker_mtp is unavailable")
        return None
    if not torch.cuda.is_available():
        _log.info("enable_talker_mtp_cuda_graph: no CUDA; keeping eager path")
        return None
    return TalkerMtpCudaGraph(
        talker,
        batch_sizes=batch_sizes,
        temperature=temperature,
        top_k=top_k,
    )


__all__ = [
    "CudaGraphTalkerDecoder",
    "TalkerMtpCudaGraph",
    "TalkerMtpGraphKey",
    "enable_talker_mtp_cuda_graph",
]
