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

import pytest

torch = pytest.importorskip("torch")


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
    decoder._captured_n_steps = -1
    decoder._prefill_len = -1
    decoder.temperature = 0.75
    decoder.top_p = 0.9
    decoder.rp = 1.0
    # defect B: stop-parity fields consumed by ``_should_stop``.
    decoder.eos_token_id = 999
    decoder.audio_stop_token = 3000
    decoder._text_finished = False
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
    # simulate: captured at len 2 + n_steps 4, now a len-7 prompt arrives
    dec._captured = True
    dec._captured_len = 2
    dec._captured_n_steps = 4
    dec._prefill_len = 7
    _apply_invalidation(dec)
    assert dec._captured is False, "different prompt length must invalidate"
    assert dec.steps == [], "stale graphs must be dropped"


def test_capture_invalidates_on_budget_change() -> None:
    """Per-step graph COUNT is baked at the first-requested ``n_steps``;
    a later request with a different budget must invalidate so the new
    step count replaces the stale one."""
    dec = _make_decoder()
    dec._captured = True
    dec._captured_len = 2
    dec._captured_n_steps = 4
    dec._prefill_len = 2  # same length, OK
    dec.n_steps = 8  # different budget -> must invalidate
    _apply_invalidation(dec)
    assert dec._captured is False, "different budget must invalidate"
    assert dec.steps == [], "stale graphs must be dropped"


def test_capture_reuses_on_same_length() -> None:
    dec = _make_decoder()
    dec._captured = True
    dec._captured_len = 2
    dec._captured_n_steps = 4
    dec._prefill_len = 2
    dec.steps = [_StubAttn()]  # non-empty fake graph list
    _apply_invalidation(dec)
    assert dec._captured is True, "same length + same budget may reuse"
    assert len(dec.steps) == 1, "must not drop cached graphs"


def test_capture_reinit_after_invalidation() -> None:
    """After invalidation, a same-length re-capture must mark _captured and
    record the length so the next same-length call reuses."""
    dec = _make_decoder()
    dec._captured = True
    dec._captured_len = 2
    dec._captured_n_steps = 4
    dec._prefill_len = 5
    _apply_invalidation(dec)  # invalidate (len changed)
    dec.steps = [_StubAttn()]  # pretend re-capture produced 1 graph
    dec._captured = True
    dec._captured_n_steps = 4
    dec._prefill_len = 5
    _apply_invalidation(dec)  # same len now -> reuse
    assert dec._captured is True
    assert len(dec.steps) == 1


# ---------------------------------------------------------------------------
# defect B: stop semantics parity with BatchedThinkerRunner.step_finished
# ---------------------------------------------------------------------------
#
# Bug: CudaGraphDecoder.generate_tokens ran ``for k in range(n_steps - 1)``
# with no EOS / audio-stop check -> emitted exactly ``n_steps`` frames even
# when the model hit text EOS + last-layer audio_stop earlier (the eager
# BatchedThinkerRunner would have halted there). Fix: ``_should_stop``
# predicate mirrors ``step_finished`` (``batched_generation.py:430-435``).
# These tests exercise the REAL predicate via the stub pattern used above
# for ``_needs_recapture`` -- no GPU, no real sampling.


def _pad_audio_codes(num_layers: int = 8, num_steps: int = 3) -> list[list[int]]:
    """Audio codes shaped like the decoder's ``audio_codes``: 8 layers, each
    a per-step sequence initialised to audio_pad (``2051``)."""
    return [[2051] * num_steps for _ in range(num_layers)]


def test_decoder_post_eos_state_machine_matches_eager_tokens() -> None:
    """Graph emits enter, PADs, then internal-stop after text EOS."""
    dec = _make_decoder()
    dec.model.enter_token_id = 201
    dec.model.pad_token_id = 0
    dec._reset_request_state(post_eos_padding_count=2, internal_stop_token_id=17)
    codes = _pad_audio_codes(num_steps=1)

    assert dec._should_stop(dec.eos_token_id, codes) is False
    assert dec._next_post_eos_token() == 201
    assert dec._should_stop(42, codes) is False
    assert dec._next_post_eos_token() == 0
    assert dec._next_post_eos_token() == 0
    assert dec._next_post_eos_token() == 17
    assert dec._should_stop(17, codes) is True


def test_stop_initial_state_does_not_stop() -> None:
    """A token that is neither EOS nor has audio_stop on layer 7 must not
    stop; flag must stay False."""
    dec = _make_decoder()
    codes = _pad_audio_codes()
    assert dec._should_stop(tok=11, audio_codes=codes) is False
    assert dec._text_finished is False


def test_stop_text_eos_only_does_not_stop() -> None:
    """Text EOS flips the flag but stops only when audio_codes[7][-1] ==
    audio_stop. With only pad on layer 7 (no audio_stop yet) the loop
    must keep going -- mirrors eager: ``step_finished`` requires BOTH."""
    dec = _make_decoder()
    codes = _pad_audio_codes()
    assert dec._should_stop(tok=dec.eos_token_id, audio_codes=codes) is False
    # flag flipped on EOS sample
    assert dec._text_finished is True
    # subsequent calls with no audio_stop still do not stop
    codes[7][-1] = 2051  # still pad
    assert dec._should_stop(tok=42, audio_codes=codes) is False


def test_stop_text_eos_and_audio_stop_stops() -> None:
    """The contract that fixes defect B: after text EOS has been observed,
    the FIRST call where audio_codes[7][-1] == audio_stop must return True."""
    dec = _make_decoder()
    codes = _pad_audio_codes()
    # 1st call: EOS observed, flag flips, audio_codes[7][-1] still pad -> False
    assert dec._should_stop(tok=dec.eos_token_id, audio_codes=codes) is False
    # 2nd call: last-layer stop code now present, text_finished is True
    codes[7][-1] = dec.audio_stop_token
    assert dec._should_stop(tok=42, audio_codes=codes) is True


def test_stop_audio_stop_alone_does_not_stop() -> None:
    """Audio stop on the last layer without text EOS must NOT stop --
    matches eager ``step_finished`` gate."""
    dec = _make_decoder()
    codes = _pad_audio_codes()
    codes[7][-1] = dec.audio_stop_token  # audio stop but no text EOS yet
    assert dec._should_stop(tok=42, audio_codes=codes) is False
    assert dec._text_finished is False
    # then text EOS arrives: still no stop because we just appended pad
    # (audio_codes[7][-1] is still pad after this step, not audio_stop).
    codes[7].append(2051)
    assert dec._should_stop(tok=dec.eos_token_id, audio_codes=codes) is False
    # finally audio_codes[7][-1] = audio_stop with text_finished=True
    codes[7][-1] = dec.audio_stop_token
    assert dec._should_stop(tok=11, audio_codes=codes) is True


def test_stop_budget_exhaustion_path_keeps_loop_running() -> None:
    """If neither gate ever fires, ``_should_stop`` stays False across the
    full budget -- the for-loop's natural bound handles budget exhaustion.
    This locks the 'missed stop' case from the spec: budget runs out before
    content naturally ends, loop completes all ``n_steps - 1`` iterations."""
    dec = _make_decoder()
    n = 4  # matches _make_decoder.n_steps
    codes = _pad_audio_codes(num_steps=n + 1)
    for _ in range(n):
        # never observe EOS, never append audio_stop on layer 7
        assert dec._should_stop(tok=11, audio_codes=codes) is False
    assert dec._text_finished is False


def test_generate_tokens_break_is_wired_into_loop() -> None:
    """Source-code contract: ``generate_tokens`` must call ``_should_stop``
    inside the decode loop and break on True. Locks the wiring so a future
    refactor that drops the break (regression to defect B) fails here."""
    import inspect

    from nanovllm_omni.optim import cuda_graph as cg

    src = inspect.getsource(cg.CudaGraphDecoder.generate_tokens)
    assert "_should_stop" in src, "defect B regression: break predicate missing"
    # break must be inside the per-step loop (not just at seed0)
    # rough check: the loop body contains both the predicate call and break
    loop_body = src.split("for k in range(self.n_steps - 1):", 1)[1]
    assert "_should_stop" in loop_body
    assert "break" in loop_body


def test_enable_cuda_graph_threads_stop_kwargs() -> None:
    """``enable_cuda_graph`` must accept and forward ``eos_token_id`` /
    ``audio_stop_token`` into the decoder (no implicit keyword crash)."""
    import inspect

    from nanovllm_omni.optim import cuda_graph as cg

    sig = inspect.signature(cg.enable_cuda_graph)
    assert "eos_token_id" in sig.parameters
    assert "audio_stop_token" in sig.parameters


if __name__ == "__main__":
    import sys

    checks = [
        test_prefill_records_length_and_resets,
        test_capture_invalidates_on_length_change,
        test_capture_reuses_on_same_length,
        test_capture_reinit_after_invalidation,
        test_stop_initial_state_does_not_stop,
        test_stop_text_eos_only_does_not_stop,
        test_stop_text_eos_and_audio_stop_stops,
        test_stop_audio_stop_alone_does_not_stop,
        test_stop_budget_exhaustion_path_keeps_loop_running,
        test_generate_tokens_break_is_wired_into_loop,
        test_enable_cuda_graph_threads_stop_kwargs,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
