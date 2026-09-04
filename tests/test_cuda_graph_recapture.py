"""Contract test: CudaGraphDecoder re-capture decision (defect #5).

defect #5 (cross-prompt KV misalignment, plan doc): `_capture` bakes per-step
graph offsets at the FIRST prompt's prefill length and never re-captures
(`if self._captured: return`). A different-length prompt's prefill therefore
misaligns with the baked offsets -> cross-prompt leak.

The mechanism is GPU-side, but the DECISION logic is pure Python and
CPU-testable: after a decode session with prompt length L1, a new prompt of
length L2 != L1 MUST invalidate the captured step set (re-capture), while
L2 == L1 may reuse it.

These tests run WITHOUT CUDA (torch.cuda.CUDAGraph would throw). We test the
method-composition contract instead:

  - `_prefill(input_ids)` records the prompt length (`decoder._prefill_len`)
    and resets buffers (`_kv_pos=0` for each attached attention).
  - `_capture(nid)` decides: reuse iff already captured AND lengths equal;
    otherwise invalidates `steps` and re-captures.

The stub model needs no GPU matmul: `_capture` guard runs before any graph
op, so with a pre-satisfied guard we only exercise the decision. We drive
the decision via a shim that fakes `_captured`/`_captured_len` states rather
than executing CUDA capture.
"""

from __future__ import annotations

import types

import torch


class _StubAttn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._kv_pos = 0
        self._nanovllm_kv_buffer = True


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(audio_pad_token=2051)
        self._attn_stub = _StubAttn()

    def modules(self):  # emulate nn.Module.modules() (iterator incl. self)
        yield self
        yield self._attn_stub


def _make_decoder() -> object:
    """Construct a CudaGraphDecoder without CUDA: only to inspect the
    re-capture decision via its state fields."""
    from nanovllm_omni.optim import cuda_graph as cg

    # Build without calling enable_cuda_graph (that needs CUDA).
    decoder = object.__new__(cg.CudaGraphDecoder)
    decoder.model = _StubModel()
    decoder.fwd = lambda *a, **k: types.SimpleNamespace(
        logits=torch.zeros(10), past_key_values=None
    )
    decoder.n_steps = 4
    decoder.audio_pad = 2051
    decoder.attns = [decoder.model._attn_stub]
    decoder.steps = []
    decoder._captured = False
    decoder._captured_len = -1
    decoder._prefill_len = -1
    decoder.temperature = 0.75
    decoder.top_p = 0.9
    decoder.rp = 1.0
    return decoder


def _apply_invalidation(decoder) -> None:
    """Mirror the non-capture side of `_capture`'s re-capture branch (drop
    stale graphs, unset captured). The DECISION is delegated to the real
    `_needs_recapture()` method so the test exercises actual code, not a
    hand-replicated copy."""
    if decoder._needs_recapture():  # real predicate
        decoder.steps = []
        decoder._captured = False
        decoder._captured_len = decoder._prefill_len


def test_prefill_records_length_and_resets() -> None:
    import inspect

    from nanovllm_omni.optim import cuda_graph as cg

    src = inspect.getsource(cg.CudaGraphDecoder._prefill)
    # records the prompt length (defect #5 re-capture key)
    assert "_prefill_len" in src
    # resets buffers so a new prompt starts from a clean KV state
    assert "_reset_pos()" in src
    # buffer reset semantics: _reset_pos must zero _kv_pos per attn
    reset_src = inspect.getsource(cg.CudaGraphDecoder._reset_pos)
    assert "_kv_pos = 0" in reset_src


def test_capture_invalidates_on_length_change() -> None:
    dec = _make_decoder()
    # simulate: captured at len 2, now a len-7 prompt arrives
    dec._captured = True
    dec._captured_len = 2
    dec._prefill_len = 7
    _apply_invalidation(dec)
    assert dec._captured is False, "different prompt length must invalidate"
    assert dec.steps == [], "stale graphs must be dropped"


def test_capture_reuses_on_same_length() -> None:
    dec = _make_decoder()
    dec._captured = True
    dec._captured_len = 2
    dec._prefill_len = 2
    dec.steps = [_StubAttn()]  # non-empty fake graph list
    _apply_invalidation(dec)
    assert dec._captured is True, "same length may reuse"
    assert len(dec.steps) == 1, "must not drop cached graphs"


def test_capture_reinit_after_invalidation() -> None:
    """After invalidation, a same-length re-capture must mark _captured and
    record the length so the next same-length call reuses."""
    dec = _make_decoder()
    dec._captured = True
    dec._captured_len = 2
    dec._prefill_len = 5
    _apply_invalidation(dec)  # invalidate (len changed)
    dec.steps = [_StubAttn()]  # pretend re-capture produced 1 graph
    dec._captured = True
    dec._prefill_len = 5
    _apply_invalidation(dec)  # same len now -> reuse
    assert dec._captured is True
    assert len(dec.steps) == 1


if __name__ == "__main__":
    import sys

    checks = [
        test_prefill_records_length_and_resets,
        test_capture_invalidates_on_length_change,
        test_capture_reuses_on_same_length,
        test_capture_reinit_after_invalidation,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
