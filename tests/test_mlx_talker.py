"""MLX talker gates: torch parity, the runner's frames, the converted weights.

``mlx`` is a Mac-track dependency and is not in the project venv, so the whole
module skips without it. The parity gate additionally needs ``torch`` and the
MiniMind-O checkpoint (``MINIMIND_PARITY_MODEL_DIR``).

Measured on the M2 Air, mlx 0.32.4 / torch 2.11.0 CPU fp16, on a random bridge
and random audio codes: the eight codebook logits differ by at most **0.0625**
and 0.0033-0.0118 on average, over sequence lengths 1, 2, 8, 12, 33 and 64.
The logits reach |47|, where the fp16 grid step is 0.03125, so the gap is two
fp16 steps of the last GEMM and not a modelling difference. A 2e-2 tolerance
would sit below that grid step and cannot be met by two fp16 implementations;
the gate uses 0.125, the measured 0.0625 plus the same margin the CUDA talker
gate uses (ADR 0022).

The frames of the two runners are **not** comparable: ``torch.multinomial``
(the exponential race) and ``mx.random.categorical`` (Gumbel-max) consume
different RNG streams, so the same seed aligns the sampling parameters and not
the draws (ADR 0023). What the gate locks instead is that the MLX runner is
deterministic for a fixed seed and follows the vendor's emission schedule.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402

torch = pytest.importorskip("torch")

from nanovllm_omni.backend.base import TalkerRequest  # noqa: E402
from nanovllm_omni.backend.mlx_talker import (  # noqa: E402
    MlxTalkerKVCache,
    MlxTalkerRunner,
    load_mlx_talker,
)
from nanovllm_omni.backend.talker_recipe import (  # noqa: E402
    AUDIO_PAD_TOKEN,
    AUDIO_STOP_TOKEN,
    NUM_CODEBOOKS,
)

SOURCE = os.environ.get("MINIMIND_PARITY_MODEL_DIR", os.path.expanduser("~/models/minimind-3o"))

MAX_ABS_DIFF = 0.125
SEQUENCE_LEN = 12
HIDDEN_SIZE = 768


def _converter():
    """The conversion script, loaded from ``scripts/`` because it is not a package."""
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "convert_minimind_talker_mlx.py"
    )
    spec = importlib.util.spec_from_file_location("convert_minimind_talker_mlx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def talker_models(tmp_path_factory):
    """The torch talker and the MLX talker converted from the same checkpoint."""
    from nanovllm_omni.models.minimind_omni.talker import load_talker

    if not os.path.isfile(os.path.join(SOURCE, "pytorch_model.bin")):
        pytest.skip(f"talker checkpoint not found in {SOURCE}")
    output = str(tmp_path_factory.mktemp("mlx-talker"))
    _converter().convert(SOURCE, output)
    torch_model = load_talker(SOURCE, device="cpu", dtype=torch.float16)
    mlx_model = load_mlx_talker(output, dtype="float16", device="gpu")
    return torch_model, mlx_model, output


def _inputs(sequence_len: int, seed: int = 0):
    """One random bridge and one random audio-code buffer."""
    generator = torch.Generator().manual_seed(seed)
    bridge = torch.randn(1, sequence_len, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    codes = torch.randint(0, 2048, (1, NUM_CODEBOOKS, sequence_len), generator=generator)
    return bridge.half(), codes


def test_mlx_talker_logits_match_the_torch_talker(talker_models) -> None:
    torch_model, mlx_model, _ = talker_models

    bridge, codes = _inputs(SEQUENCE_LEN)
    with torch.no_grad():
        torch_logits = torch_model(bridge, codes, None, None)
    mlx_logits = mlx_model(mx.array(bridge.numpy()), mx.array(codes.numpy()), None, None)

    assert len(mlx_logits) == NUM_CODEBOOKS
    for index, (left, right) in enumerate(zip(torch_logits, mlx_logits, strict=True)):
        difference = float(mx.max(mx.abs(mx.array(left.numpy()) - right[0, -1, :])).item())
        assert difference <= MAX_ABS_DIFF, f"codebook {index} differs by {difference}"


def test_mlx_windowed_matches_the_dense_path(talker_models) -> None:
    """The windowed path recomputes the newest columns and reads the rest (ADR 0022)."""
    _, mlx_model, _ = talker_models

    bridge, codes = _inputs(SEQUENCE_LEN, seed=1)
    mlx_bridge = mx.array(bridge.numpy())
    mlx_codes = mx.array(codes.numpy())
    cache = MlxTalkerKVCache(len(mlx_model.layers))

    for step in range(1, SEQUENCE_LEN + 1):
        lo = cache.length
        window = mlx_model(mlx_bridge[:, lo:step, :], mlx_codes[:, :, lo:step], None, cache)
        dense = mlx_model(mlx_bridge[:, :step, :], mlx_codes[:, :, :step], None, None)
        for left, right in zip(dense, window, strict=True):
            difference = float(mx.max(mx.abs(left[0, -1, :] - right[0, -1, :])).item())
            assert difference <= MAX_ABS_DIFF


def test_mlx_runner_frames_follow_the_vendor_schedule(talker_models) -> None:
    """The frames are deterministic for a seed and start at step 8 (ADR 0022's gate)."""
    _, _, output = talker_models

    runner = MlxTalkerRunner(output, dtype="float16", device="gpu")
    generator = torch.Generator().manual_seed(3)
    bridge = torch.randn(40, HIDDEN_SIZE, generator=generator, dtype=torch.float32).numpy()
    request = TalkerRequest(bridge=bridge, num_steps=24, start_pos=16, temperature=0.2, seed=42)

    frames = runner.generate(request)

    assert frames == runner.generate(request)
    assert len(frames) == 24 - NUM_CODEBOOKS
    assert all(len(frame) == NUM_CODEBOOKS for frame in frames)


def test_converted_weights_load_and_pad_codes_stay_inside_the_vocabulary(talker_models) -> None:
    """The pad and stop codes are ordinary vocabulary entries of the audio head."""
    _, mlx_model, output = talker_models

    weights = mx.load(os.path.join(output, "model.safetensors"))

    assert len(weights) == 92
    assert weights["text_scale"].shape == ()
    assert float(weights["text_scale"].item()) == pytest.approx(2.140625, abs=1e-3)
    assert mlx_model.config.audio_vocab_size > AUDIO_PAD_TOKEN
    assert mlx_model.config.audio_vocab_size > AUDIO_STOP_TOKEN

    codes = mx.full((1, NUM_CODEBOOKS, 4), AUDIO_PAD_TOKEN, dtype=mx.int32)
    embeds = mlx_model.embed_tokens(codes)

    assert embeds.shape == (1, 4, HIDDEN_SIZE)
    assert bool(mx.all(mx.isfinite(embeds)).item())
    assert float(mx.max(mx.abs(embeds)).item()) > 0.0
