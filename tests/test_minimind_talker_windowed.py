"""Windowed KV vs dense: the talker's two paths must agree.

ADR 0017 wants the talker to cache the settled prefix and recompute only the
newest eight columns, so every step's query width is constant. That is the
precondition for varlen batching and for graph capture.

The window changes the attention GEMM's shape, so the two paths cannot be
bit-identical (ADR 0022 records the same for the lm_head). What must hold is
that the last position's logits agree within tolerance -- otherwise the
window dropped or kept the wrong columns, and the sampled audio diverges.

Two gates:

- ``test_windowed_matches_dense_logits`` compares the model's last-position
  logits on CPU in fp32. It needs no GPU, no triton and no fork attention, so
  it runs anywhere.
- ``test_windowed_stage_matches_dense_codes`` compares the audio codes the
  stage produces under both paths on CPU with a fixed bridge and checkpoint.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

MODEL = os.environ.get("MINIMIND_PARITY_MODEL_DIR", os.path.expanduser("~/models/minimind-3o"))
PROMPT_LEN = 12

# fp32 on CPU: the window and dense arms differ only in the attention GEMM's
# shape, so the drift is GEMM reduction noise, not a semantic difference.
# The reference value leaves room for fp32 noise while still catching a wrong
# RoPE offset or a dropped column.
MAX_ABS_DIFF = 1e-3
MAX_MEAN_ABS_DIFF = 1e-4

NUM_CODEBOOKS = 8


def _require_checkpoint() -> None:
    if not os.path.isfile(os.path.join(MODEL, "pytorch_model.bin")):
        pytest.skip(f"checkpoint not found at {MODEL}")


def _stage(kv: str):
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage

    return TalkerStage(
        deploy=None,
        args=SimpleNamespace(
            model=MODEL,
            device="cpu",
            dtype="float32",
            extra={"talker_impl": "rewritten", "talker_kv": kv},
        ),
    )


def _payload(bridge: torch.Tensor, entry: dict):
    from nanovllm_omni.models.minimind_omni.stage_processors import TalkerInputPayload

    length = bridge.shape[0]
    num_steps = entry["num_steps"]
    return TalkerInputPayload(
        input_ids=torch.zeros(length, dtype=torch.long),
        bridge_states=bridge,
        text_token_ids=tuple(range(length)),
        prompt_token_ids=tuple(range(PROMPT_LEN)),
        output_token_ids=tuple(range(PROMPT_LEN, length)),
        metadata={"start_pos": PROMPT_LEN, "num_steps": num_steps},
    )


def test_windowed_matches_dense_logits() -> None:
    """The windowed path must land on the same logits as the dense path."""
    _require_checkpoint()

    from nanovllm_omni.models.minimind_omni.talker import (
        TalkerConfig,
        TalkerKVCache,
        load_talker,
    )

    config = TalkerConfig.from_checkpoint(MODEL)
    model = load_talker(MODEL, device="cpu", dtype=torch.float32)

    start_pos = NUM_CODEBOOKS
    num_steps = 16

    torch.manual_seed(0)
    total = start_pos + num_steps
    bridge = torch.randn(1, total, config.talker_hidden_size)
    codes = torch.randint(0, config.audio_vocab_size, (1, NUM_CODEBOOKS, total))

    # The stage recomputes the growing prefix every step, so the dense arm
    # does the same. Comparing one-shot dense against a stepped window
    # compares different columns.
    with torch.no_grad():
        cache = TalkerKVCache(config.num_layers)
        for step in range(num_steps):
            seq_len = start_pos + step
            dense = model(bridge[:, :seq_len], codes[:, :, :seq_len], None)

            lo = cache.length
            windowed = model(bridge[:, lo:seq_len], codes[:, :, lo:seq_len], None, cache)

            # The model returns one logit tensor per codebook, each [B, T, V].
            dense_last = torch.stack([row[:, -1, :].float() for row in dense])
            windowed_last = torch.stack([row[:, -1, :].float() for row in windowed])
            max_diff = (dense_last - windowed_last).abs().max().item()
            mean_diff = (dense_last - windowed_last).abs().mean().item()

            assert (
                max_diff <= MAX_ABS_DIFF
            ), f"step {step}: max abs diff {max_diff} > {MAX_ABS_DIFF}"
            assert (
                mean_diff <= MAX_MEAN_ABS_DIFF
            ), f"step {step}: mean abs diff {mean_diff} > {MAX_MEAN_ABS_DIFF}"


def test_windowed_cache_keeps_the_settled_prefix() -> None:
    """The cache holds the settled prefix; its length is the next RoPE offset."""
    _require_checkpoint()

    from nanovllm_omni.models.minimind_omni.talker import (
        TalkerConfig,
        TalkerKVCache,
        load_talker,
    )

    config = TalkerConfig.from_checkpoint(MODEL)
    model = load_talker(MODEL, device="cpu", dtype=torch.float32)

    start_pos = NUM_CODEBOOKS
    num_steps = 16
    total = start_pos + num_steps

    torch.manual_seed(0)
    bridge = torch.randn(1, total, config.talker_hidden_size)
    codes = torch.randint(0, config.audio_vocab_size, (1, NUM_CODEBOOKS, total))

    cache = TalkerKVCache(config.num_layers)
    with torch.no_grad():
        for step in range(num_steps):
            seq_len = start_pos + step
            lo = cache.length
            model(bridge[:, lo:seq_len], codes[:, :, lo:seq_len], None, cache)
            # The settled prefix grows by one column per step once the first
            # window has been trimmed; it never exceeds the sequence so far.
            assert cache.length <= seq_len
            assert cache.length >= max(0, seq_len - NUM_CODEBOOKS)


def test_windowed_stage_matches_dense_codes() -> None:
    """The stage must emit the same audio codes under both paths."""
    _require_checkpoint()

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "talker_golden_cpu.json")
    with open(fixture, encoding="utf-8") as handle:
        entry = json.load(handle)["dense_seed42_steps12"]
    generator = torch.Generator().manual_seed(entry["bridge_seed"])
    bridge = torch.randn(
        PROMPT_LEN + entry["num_steps"], 768, generator=generator, dtype=torch.float32
    )
    dense_payload = _payload(bridge, entry)
    windowed_payload = _payload(bridge, entry)
    sampling = SimpleNamespace(extra={"audio_temperature": 0.2}, seed=entry["seed"])
    dense_stage = _stage("dense")
    windowed_stage = _stage("windowed")

    dense_codes = dense_stage(dense_payload, sampling).audio_codes
    windowed_codes = windowed_stage(windowed_payload, sampling).audio_codes

    assert torch.equal(dense_codes, windowed_codes)
