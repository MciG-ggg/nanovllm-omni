"""CPU-only contract tests for ``optim/paged_cuda_graph``.

These lock the *defining property* of this decoder: it captures ONE
graph and replays it for every AR step, in contrast to
``optim/cuda_graph`` which captures ``n_steps - 1`` graphs (one per
decode position).

The assertions are deliberately structural (source-level + attribute
shape) so they run without CUDA and still fail loudly if someone
reintroduces a per-position capture loop.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest

# torch must load BEFORE the stub registration below; stubbing numpy
# first makes torch's C extension abort at import time.
import torch  # noqa: E402  (import order is load-bearing)


def _ensure_stub(name: str, **attrs: Any) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_ensure_stub("triton", jit=(lambda *a, **kw: (lambda fn: fn)))
_ensure_stub("triton.language", constexpr=type("constexpr", (), {}))
_ensure_stub(
    "flash_attn",
    flash_attn_varlen_func=(lambda *a, **kw: None),
    flash_attn_with_kvcache=(lambda *a, **kw: None),
)
# numpy is a real torch dependency -- never stub it.
_ensure_stub("xxhash")

from nanovllm_omni.models.minimind_omni import cuda_graph as cg  # noqa: E402
from nanovllm_omni.models.minimind_omni import paged_attention as pa  # noqa: E402
from nanovllm_omni.models.minimind_omni import paged_cuda_graph as pcg  # noqa: E402


def test_capture_builds_exactly_one_graph() -> None:
    """``_capture`` must not loop over decode positions.

    The per-position decoder builds ``n_steps - 1`` graphs; this one
    builds a single graph whose replay serves every position.
    """
    src = inspect.getsource(pcg.PagedCudaGraphDecoder._capture)
    assert src.count("torch.cuda.CUDAGraph()") == 1, "exactly one graph must be constructed"
    assert (
        "for " not in src.split("side = torch.cuda.Stream()")[1]
    ), "capture must not iterate over decode positions"
    body = src.split('"""')[-1]
    assert "n_steps" not in body, "n_steps must not influence capture"


def test_per_position_decoder_still_captures_n_steps_minus_one() -> None:
    """Guard the contrast: the old decoder is genuinely per-position.

    If this ever stops being true, the comparison this work is based on
    (and the numbers reported from it) would be describing something
    else.
    """
    src = inspect.getsource(cg.CudaGraphDecoder._capture)
    assert "num_decode_graphs = max(self.n_steps - 1, 0)" in src
    assert "for _ in range(num_decode_graphs):" in src


def test_decode_loop_replays_the_same_graph_object() -> None:
    """The decode loop must replay ``self.graph`` -- not index a list."""
    src = inspect.getsource(pcg.PagedCudaGraphDecoder.generate_tokens)
    assert "self.graph.replay()" in src, "must replay the single captured graph"
    assert "self.steps[" not in src, "must not index a per-position graph list"
    # Metadata is refreshed before every replay -- that is what lets one
    # graph serve every position.
    replay_at = src.index("self.graph.replay()")
    sync_at = src.index("self._sync_decode_ctx()")
    assert sync_at < replay_at, "decode context must be synced before replay"


def test_recapture_is_independent_of_n_steps_and_prompt_len() -> None:
    """Only the window width may invalidate the capture.

    ``n_steps`` and prompt length are capture shapes for the
    per-position decoder but must not be for this one.
    """
    src = inspect.getsource(pcg.PagedCudaGraphDecoder._needs_recapture)
    # Strip the docstring: it *describes* n_steps, the logic must not use it.
    body = src.split('"""')[-1]
    assert "_captured_window" in body
    assert "n_steps" not in body
    assert "_prefill_len" not in body


def test_paged_decoder_inherits_stop_machine() -> None:
    """Stop parity with the per-position decoder is by inheritance.

    Re-implementing the post-EOS state machine would silently change
    frame counts and make the two graph paths incomparable.
    """
    assert issubclass(pcg.PagedCudaGraphDecoder, cg.CudaGraphDecoder)
    for name in ("_should_stop", "_next_post_eos_token", "_reset_request_state", "_result"):
        assert (
            name not in pcg.PagedCudaGraphDecoder.__dict__
        ), f"{name} must be inherited, not overridden"


@pytest.mark.parametrize(
    ("prompt_len", "n_steps", "block_size", "expected"),
    [
        (2, 16, 16, 2),  # 2 + 16 + 1 = 19 -> 2 blocks
        (6, 16, 16, 2),  # 6 + 16 + 1 = 23 -> 2 blocks
        (30, 16, 16, 3),  # 30 + 16 + 1 = 47 -> 3 blocks
        (2, 16, 8, 3),  # 19 -> 3 blocks at block_size 8
    ],
)
def test_required_blocks_covers_prompt_plus_budget(
    prompt_len: int, n_steps: int, block_size: int, expected: int
) -> None:
    """Window must cover prompt + generation budget (+1 for the seed token).

    Undersizing here would corrupt decode once the sequence grows past
    the gathered window.
    """
    dec = object.__new__(pcg.PagedCudaGraphDecoder)
    dec.n_steps = n_steps
    dec.block_size = block_size
    assert dec._required_blocks(prompt_len) == expected
    assert dec._required_blocks(prompt_len) * block_size >= prompt_len + n_steps


def test_prefill_context_sets_flashattention_boundaries() -> None:
    """The B=1 paged decoder must supply valid varlen prefill metadata."""
    dec = object.__new__(pcg.PagedCudaGraphDecoder)
    dec.block_size = 4
    dec._seq = types.SimpleNamespace(block_table=[2, 3])
    ctx = pa.PagedKVContext(
        slot_mapping=torch.full((8,), -1, dtype=torch.int32),
        context_lens=torch.zeros(1, dtype=torch.int32),
        block_tables=torch.zeros((1, 2), dtype=torch.int32),
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32),
        cu_seqlens_k=torch.zeros(2, dtype=torch.int32),
    )
    dec.cache = types.SimpleNamespace(_ctx=ctx)

    dec._sync_prefill_ctx(prompt_len=6)

    assert ctx.is_prefill
    assert ctx.cu_seqlens_q.tolist() == [0, 6]
    assert ctx.cu_seqlens_k.tolist() == [0, 6]
    assert ctx.max_seqlen_q == ctx.max_seqlen_k == 6


def test_enable_paged_cuda_graph_returns_none_without_cuda() -> None:
    """No CUDA -> no decoder, and no exception."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the no-CUDA guard cannot be exercised here")
    assert pcg.enable_paged_cuda_graph(object()) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
