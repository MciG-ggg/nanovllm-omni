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

    class _Raiser:
        def decode(self, codes, *args, **kwargs):
            raise RuntimeError("synthetic decode failure")

    inner = _Raiser()
    wrapped = GraphedMimi(inner)
    with pytest.raises(RuntimeError, match="synthetic"):
        wrapped.decode(_codes(8))
    s = wrapped.stats()
    assert s["capture_failures"] >= 1
    assert s["graphs_captured"] == 0


@requires_cuda
def test_replay_failure_falls_back_to_eager():
    """If replay raises, the next call goes through the eager inner."""
    from nanovllm_omni.optim.cuda_graph import MimiDecodeGraphCache

    class _BoomReplay:
        def __init__(self):
            self.calls = 0

        def decode(self, codes, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # First call: return a value so capture "succeeds".
                return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))
            raise RuntimeError("synthetic replay failure")

    inner = _BoomReplay()
    cache = MimiDecodeGraphCache(inner)
    # First call: capture + replay works.
    out = cache.replay(_codes(8))
    assert out is not None
    # Now manually drop the slot to force a re-capture path that fails.
    cache._slots.clear()

    # Replace inner so the second capture also fails.
    def _raise(codes, *a, **k):
        raise RuntimeError("synthetic capture failure")

    inner.decode = _raise
    out = cache.replay(_codes(8))
    assert out is None  # capture failure -> None


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

    captured: dict[str, Any] = {}

    class _Spy:
        def decode(self, codes, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(audio_values=torch.zeros(1, 4, 1))

    wrapped = GraphedMimi(_Spy())
    wrapped.decode(_codes(8), "positional1", kw="value")
    # Inner received the extra args. The cache replay path failed
    # (capture failure in the spy is fine -- the wrapper falls back).
    assert captured["args"] == ("positional1",)
    assert captured["kwargs"] == {"kw": "value"}
