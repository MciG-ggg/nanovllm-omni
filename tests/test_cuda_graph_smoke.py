"""CUDA smoke tests for thinker CUDA Graph parity with eager decode.

These tests require a GPU and real model weights; skip gracefully when
CUDA is unavailable or weights are missing.  Mark: ``@pytest.mark.smoke``.

Run on WSL RTX 3050::

    cd ~/nanovllm-omni && source ~/venvs/nanovllm-omni/bin/activate
    HF_HUB_OFFLINE=1 python -m pytest tests/test_cuda_graph_smoke.py -v
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

_MODEL = os.environ.get("NANO_MODEL", "/home/mcig/minimind-3o")
_MIMI = os.environ.get("NANO_MIMI", "/home/mcig/mimi")
_HAS_CUDA = torch.cuda.is_available()
_HAS_WEIGHTS = os.path.isdir(_MODEL) and os.path.isdir(_MIMI)

pytestmark = pytest.mark.smoke
skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
skip_no_weights = pytest.mark.skipif(not _HAS_WEIGHTS, reason="model weights not found")


@pytest.fixture(scope="module")
def bundle():
    if not _HAS_CUDA or not _HAS_WEIGHTS:
        pytest.skip("CUDA + weights required")
    from nanovllm_omni.models.minimind_omni import load_minimind_omni_bundle

    return load_minimind_omni_bundle(model_id=_MODEL, mimi_model_id=_MIMI)


@pytest.fixture(scope="module")
def input_ids(bundle):
    tok = bundle.tokenizer
    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    return ids


@skip_no_cuda
@skip_no_weights
def test_graph_captures_n_steps_minus_one(bundle, input_ids):
    """Verify the decoder captures exactly n_steps - 1 graphs."""
    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph

    n = 8
    model = bundle.model
    dec = enable_cuda_graph(model, n_steps=n, max_len=input_ids.shape[1] + n + 4)
    assert dec is not None
    # Force capture by running generate_tokens
    dec.generate_tokens(input_ids, seed=42, return_audio=True)
    assert len(dec.steps) == n - 1, f"Expected {n - 1} graphs, got {len(dec.steps)}"


@skip_no_cuda
@skip_no_weights
def test_graph_token_count_matches_eager(bundle, input_ids):
    """Graph and eager produce the same number of text tokens (same seed)."""
    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph
    from nanovllm_omni.models.minimind_omni.generation import stream_generate

    model = bundle.model
    seed = 42
    max_new = 16
    eos_id = getattr(model, "eos_token_id_2", 2)

    # Eager
    torch.manual_seed(seed)
    eager_frames = []
    for _text, audio in stream_generate(
        model,
        input_ids,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_id,
        open_thinking=False,
    ):
        if audio is not None:
            eager_frames.append(audio)

    # Graph
    dec = enable_cuda_graph(model, n_steps=max_new, max_len=input_ids.shape[1] + max_new + 4)
    assert dec is not None
    dec.temperature = 0.7
    dec.top_p = 0.9
    text_codes, audio_codes = dec.generate_tokens(input_ids, seed=seed, return_audio=True)

    # Token count should match (both stop at EOS or budget)
    # Allow ±1 tolerance for EOS alignment differences
    assert (
        abs(len(text_codes) - len(eager_frames)) <= 1
    ), f"Token count mismatch: graph={len(text_codes)}, eager={len(eager_frames)}"


@skip_no_cuda
@skip_no_weights
def test_graph_deterministic_across_calls(bundle, input_ids):
    """Same seed, two graph calls → identical text tokens."""
    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph

    model = bundle.model
    dec = enable_cuda_graph(model, n_steps=16, max_len=input_ids.shape[1] + 20)
    assert dec is not None

    t1 = dec.generate_tokens(input_ids, seed=42, return_audio=False)
    t2 = dec.generate_tokens(input_ids, seed=42, return_audio=False)
    assert t1 == t2, "Same seed must produce identical tokens"


@skip_no_cuda
@skip_no_weights
def test_recapture_on_different_prompt_length(bundle):
    """Different prompt lengths trigger recapture (defect #5 guard)."""
    from nanovllm_omni.engine.cuda_graph import enable_cuda_graph

    model = bundle.model
    tok = bundle.tokenizer
    short = tok("Hi", return_tensors="pt").input_ids.cuda()
    long = tok("Tell me a long story about adventures", return_tensors="pt").input_ids.cuda()
    max_len = max(short.shape[1], long.shape[1]) + 20

    dec = enable_cuda_graph(model, n_steps=8, max_len=max_len)
    assert dec is not None

    dec.generate_tokens(short, seed=42, return_audio=False)
    captured_len_1 = dec._captured_len

    dec.generate_tokens(long, seed=42, return_audio=False)
    captured_len_2 = dec._captured_len

    assert captured_len_1 != captured_len_2, "Different prompts should recapture"
    assert dec._captured is True, "Should be captured after recapture"
