"""Tests for ``nanovllm_omni.optim.compile`` (TK-015)."""

from __future__ import annotations

from typing import Any

import pytest

# Skip the whole file when torch is missing. See test_optim_bench.py
# for the rationale; this is the same project convention.
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_compile_model_returns_callable():
    """``compile_model`` returns an object on which forward still works."""
    from nanovllm_omni.optim.compile import compile_model

    class M(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    m = M()
    out = compile_model(m, mode="default", device="cpu")
    assert out is not None
    x = torch.tensor([1.0, 2.0])
    result = out(x)
    assert torch.allclose(result, x * 2)


def test_compile_unsupported_op_returns_eager(caplog):
    """When torch.compile raises, return the original model and log a warning."""
    import logging

    from nanovllm_omni.optim.compile import compile_model

    class Broken(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    m = Broken()
    real_compile = torch.compile

    def boom(model: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic compile failure")

    torch.compile = boom
    try:
        with caplog.at_level(logging.WARNING, logger="nanovllm_omni.optim.compile"):
            out = compile_model(m, mode="default", device="cpu")
    finally:
        torch.compile = real_compile

    assert out is m, "compile_model must return the original model on RuntimeError"
    # At least one warning was emitted.
    assert any("torch.compile" in rec.message for rec in caplog.records)


def test_run_generate_with_compile_runs_and_returns_wav():
    """``run_one(..., compile=True)`` returns a RunResult; stub bundle suffices."""
    from nanovllm_omni.optim.bench import run_one

    # Reuse the stub bundle from test_optim_bench without importing private names.
    from tests.test_optim_bench import _bundle, _short_prompt

    bundle = _bundle()
    prompt = _short_prompt()

    r = run_one(bundle, prompt, max_tokens=4, compile=True)
    assert r.audio_bytes[:4] == b"RIFF"
    # Compile path is a no-op on CPU stub (no-op kernel); output still valid.


# ---------------------------------------------------------------------------
# bitsandbytes gate
# ---------------------------------------------------------------------------


def test_quantize_skipped_if_no_bnb():
    """When bitsandbytes is absent, ``quantize_thinker_int8`` raises RuntimeError."""
    from nanovllm_omni.optim.compile import (
        has_bnb,
        quantize_thinker_int8,
    )

    if has_bnb():
        pytest.skip("bitsandbytes is installed in this env; skip-path not exercised")

    m = torch.nn.Module()
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        quantize_thinker_int8(m)
