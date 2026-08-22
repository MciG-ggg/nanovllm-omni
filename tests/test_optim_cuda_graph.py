"""Tests for ``nanovllm_omni.optim.cuda_graph`` (mimi-decode graph path).

The previous implementation wrapped the MiniMindOmni thinker; the
main forward was not capture-safe so the wrapper always fell back
to eager. The new scope is much narrower -- only the mimi.decode
call is intercepted -- so the tests focus on that.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# Skip the whole file when torch is missing. See test_optim_bench.py
# for the rationale; this is the same project convention.
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Fake mimi model: a tiny stub that matches the surface
# ``decode(codes)`` needs. The bench only touches ``.audio_values``.
# ---------------------------------------------------------------------------


class _FakeMimi:
    """Stand-in for the HF ``MimiModel`` used by the bench harness."""

    def __init__(self, audio_length: int = 96) -> None:
        # Some attribute the wrapper must proxy through.
        self.some_attr = 42
        self._audio_length = audio_length

    def decode(self, codes: torch.Tensor) -> Any:
        # codes: [B, 8, T]. Produce a fake audio tensor [B, T_audio, 1].
        b = codes.shape[0]
        return SimpleNamespace(
            audio_values=torch.zeros(b, self._audio_length, 1),
        )


def _codes(t: int = 8) -> torch.Tensor:
    return torch.zeros(1, 8, t, dtype=torch.long)


# Skip the graph-capture path on CPU-only hosts; the rest of the
# tests (idempotency, attribute proxying, fallback paths) still
# run and verify the wrapper structure.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graph capture needs a CUDA device",
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_graph_compile_mimi_is_idempotent():
    """Wrapping twice is a no-op (the second call returns the same wrapper)."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_mimi

    inner = _FakeMimi()
    wrapped1 = graph_compile_mimi(inner)
    wrapped2 = graph_compile_mimi(wrapped1)
    assert wrapped1 is wrapped2


@requires_cuda
def test_decode_routes_through_cache_after_successful_capture():
    """After a successful capture, subsequent decodes go through the cache."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_mimi

    inner = _FakeMimi()
    wrapped = graph_compile_mimi(inner)
    out = wrapped.decode(_codes(8))
    # The fake's audio_values is [1, 96, 1]; cloned each call.
    assert out.audio_values.shape == (1, 96, 1)
    s = wrapped.stats()
    assert s["graphs_captured"] >= 1
    assert s["graph_calls"] >= 1
    assert s["eager_calls"] == 0


@requires_cuda
def test_different_input_shapes_capture_separate_graphs():
    """Each ``codes.shape`` key gets its own graph slot."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_mimi

    inner = _FakeMimi()
    wrapped = graph_compile_mimi(inner)
    for t in (4, 6, 8):
        wrapped.decode(_codes(t))
    s = wrapped.stats()
    assert s["graphs_captured"] == 3


@requires_cuda
def test_capture_failure_falls_back_to_eager():
    """If the inner decode raises during capture, fall back to eager."""
    from nanovllm_omni.optim.cuda_graph import GraphedMimi

    inner_calls = 0

    class _FailsOnCaptureAttempt:
        """Succeed during warmup (calls 1-3), raise on capture
        attempt (call 4), succeed on fallback (call 5+)."""

        def decode(self, codes, *args, **kwargs):
            nonlocal inner_calls
            inner_calls += 1
            if inner_calls <= 3:
                return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))
            if inner_calls == 4:
                raise RuntimeError("synthetic decode failure")
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    inner = _FailsOnCaptureAttempt()
    wrapped = GraphedMimi(inner)
    out = wrapped.decode(_codes(8))
    # Fallback returned the eager inner's output, not a captured graph.
    assert out.audio_values.shape == (1, 4, 1)
    s = wrapped.stats()
    assert s["capture_failures"] >= 1
    assert s["graphs_captured"] == 0
    assert s["eager_calls"] == 1
    assert inner_calls == 5  # 3 warmup + 1 capture attempt + 1 eager fallback


@requires_cuda
def test_replay_failure_falls_back_to_eager():
    """If replay raises, the next call goes through the eager inner."""
    from nanovllm_omni.optim.cuda_graph import MimiDecodeGraphCache

    class _BoomReplay:
        """Succeed for calls 1-3 (warmup), succeed again on call 4
        (capture), succeed on call 5+ (replay path).

        (Captures successfully; replay failure is tested by dropping
        the slot and replacing inner so the next capture fails.)"""

        def __init__(self):
            self.calls = 0

        def decode(self, codes, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    inner = _BoomReplay()
    cache = MimiDecodeGraphCache(inner)
    # First call: 3 warmup + 1 capture -> slot stored.
    out = cache.replay(_codes(8))
    assert out is not None

    # Drop the slot to force a re-capture; replace inner with a spy
    # that fails on capture attempt so re-capture is memoised as _FAILED.
    cache._slots.clear()

    class _FailsOnCaptureAttempt:
        def __init__(self):
            self.calls = 0

        def decode(self, codes, *a, **k):
            self.calls += 1
            if self.calls <= 3:
                return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))
            raise RuntimeError("synthetic capture failure")

    cache._mimi = _FailsOnCaptureAttempt()
    out = cache.replay(_codes(8))
    assert out is None
    # capture_failure_count tracks the second capture's failure.
    assert cache._capture_failure_count == 1


def test_replay_output_is_independent_of_static_buffer():
    """Two consecutive replays return tensors that do not alias the buffer."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_mimi

    inner = _FakeMimi()
    wrapped = graph_compile_mimi(inner)
    a = wrapped.decode(_codes(8))
    b = wrapped.decode(_codes(8))
    # The fake's audio_values is zeros; mutating one clone must not
    # affect the other. The clones live in separate memory.
    a.audio_values.fill_(99.0)
    assert b.audio_values.abs().sum() == 0


def test_wrapper_proxies_non_decode_attributes():
    """Non-decode attributes go to the inner mimi via ``__getattr__``."""
    from nanovllm_omni.optim.cuda_graph import graph_compile_mimi

    inner = _FakeMimi()
    inner.extra_attr = "hello"
    wrapped = graph_compile_mimi(inner)
    assert wrapped.some_attr == 42
    assert wrapped.extra_attr == "hello"


def test_decode_proxies_extra_args_and_kwargs():
    """Extra positional / keyword args to ``decode`` are forwarded to the inner."""
    from nanovllm_omni.optim.cuda_graph import GraphedMimi

    last_call: dict[str, Any] = {}

    class _FailsOnCaptureAttempt:
        """Succeed during warmup, raise on capture attempt so the
        wrapper falls back to the inner; succeed on fallback."""

        def __init__(self):
            self.calls = 0

        def decode(self, codes, *args, **kwargs):
            self.calls += 1
            last_call["args"] = args
            last_call["kwargs"] = kwargs
            if self.calls <= 3:
                return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))
            if self.calls == 4:
                raise RuntimeError("synthetic capture failure")
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    inner = _FailsOnCaptureAttempt()
    wrapped = GraphedMimi(inner)
    out = wrapped.decode(_codes(8), "positional1", kw="value")
    # The last call was the eager fallback; it must forward the original
    # extra args/kwargs from the wrapper invocation, NOT the warmup's
    # stripped-down (input_buf,).
    assert out is not None
    assert last_call["args"] == ("positional1",)
    assert last_call["kwargs"] == {"kw": "value"}


@requires_cuda
def test_capture_failure_is_memoised_per_shape():
    """After capture fails for a shape, the wrapper must not retry
    capture on subsequent calls with the same shape.

    On ``cudaErrorStreamCaptureInvalidated`` the stream stays poisoned
    for the rest of the process (PyTorch limitation), so retrying has
    no chance of succeeding. Before this fix, ``_capture`` returned
    ``None`` without marking ``_slots[key]``, so every call re-ran
    the 3-warmup + sync + capture-attempt cycle (~90 ms wasted per
    call on the MiniMind-O bench; see docs/perf/session-9.md).
    """
    from nanovllm_omni.optim.cuda_graph import (
        MimiDecodeGraphCache,
    )

    inner_calls = 0

    class _FailsOnEveryFourth:
        """Succeed for calls 1-3 (warmup), raise on call 4 (capture
        attempt), succeed for 5-7 (next round's warmup), raise on
        call 8 (next round's capture attempt), etc. Lets a single
        spy instance handle multiple distinct shapes cleanly."""

        def decode(self, codes, *args, **kwargs):
            nonlocal inner_calls
            inner_calls += 1
            if inner_calls % 4 == 0:
                raise RuntimeError("synthetic capture failure")
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    cache = MimiDecodeGraphCache(_FailsOnEveryFourth())

    # First call: 3 warmup + 1 capture attempt (caught, _FAILED cached).
    out = cache.replay(_codes(8))
    assert out is None
    assert inner_calls == 4
    assert cache._slots[(1, 8, 8)] is MimiDecodeGraphCache._FAILED

    # Subsequent calls with the same shape must short-circuit --
    # NO additional warmup, NO additional capture attempt.
    for _ in range(10):
        out = cache.replay(_codes(8))
        assert out is None
    assert inner_calls == 4  # unchanged

    # A different shape is unaffected by the memoised failure.
    out = cache.replay(_codes(12))
    assert out is None
    assert inner_calls == 8  # 4 more for the new shape

    # capture_failure_count is exposed (per-shape, so two distinct
    # failed shapes counted twice).
    assert cache._capture_failure_count == 2


def test_replay_short_circuits_on_memoised_failure():
    """Pure unit test (no CUDA): directly pre-populate ``_slots`` with
    ``_FAILED`` and verify ``replay`` returns None without touching
    the inner mimi. Covers the cache logic in isolation from the
    capture machinery.
    """
    from nanovllm_omni.optim.cuda_graph import MimiDecodeGraphCache

    inner_calls = 0

    class _Inner:
        def decode(self, codes, *args, **kwargs):
            nonlocal inner_calls
            inner_calls += 1
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    cache = MimiDecodeGraphCache(_Inner())
    # Simulate a previously-failed capture for shape (1, 8, 8).
    cache._slots[(1, 8, 8)] = MimiDecodeGraphCache._FAILED

    # replay() must NOT call inner.decode.
    for _ in range(5):
        assert cache.replay(_codes(8)) is None
    assert inner_calls == 0
