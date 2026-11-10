"""Code2Wav's codec must load and decode on the devices the Mac track uses.

The Mimi loader used to build the model on the meta device and materialise
it afterwards; on MPS that leaves placeholders without storage and every
``decode`` call died with "Placeholder storage has not been allocated on
MPS device". The stage was therefore CPU-only on a Mac, quietly.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

MIMI_DIR = os.path.expanduser("~/models/mimi")
FRAMES = 20
CODEBOOKS = 8


def _require_checkpoint() -> str:
    if not os.path.isdir(MIMI_DIR):
        pytest.skip(f"mimi codec not found at {MIMI_DIR}")
    return MIMI_DIR


def _frames(device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    flat = torch.randint(0, 2048, (FRAMES, CODEBOOKS), generator=generator, dtype=torch.long)
    return flat.to(device)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_the_codec_loads_and_decodes_on_this_device(device: str) -> None:
    """BUILD / OPERATE / CHECK: load on ``device``, decode, get audio back."""
    from nanovllm_omni.models.minimind_omni.bundle import MIMI_SAMPLE_RATE
    from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav

    _require_checkpoint()
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps is not available on this host")

    from nanovllm_omni.models.minimind_omni.code2wav import load_mimi_codec

    mimi = load_mimi_codec(device=device)
    assert next(mimi.parameters()).device.type == device

    codes = _frames("cpu").tolist()
    samples = decode_audio(mimi, codes, device)

    assert samples.size > 0
    assert len(encode_wav(samples, sample_rate=MIMI_SAMPLE_RATE)) > 44


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mps_and_cpu_agree_at_the_same_precision(dtype: str) -> None:
    """The two devices must agree when they run the same precision.

    They do not at mixed precision: CPU silently promotes fp16 weights to
    fp32 (measured: cpu fp16 output == cpu fp32 output exactly), while MPS
    really runs fp16. Measured at equal precision: fp32 2.5e-06, fp16
    2.8e-03, both against a waveform whose peak is 0.88.
    """
    import numpy as np

    from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, load_mimi_codec

    _require_checkpoint()
    if not torch.backends.mps.is_available():
        pytest.skip("mps is not available on this host")

    codes = _frames("cpu").tolist()
    on_cpu = decode_audio(load_mimi_codec(device="cpu", dtype=dtype), codes, "cpu")
    on_mps = decode_audio(load_mimi_codec(device="mps", dtype=dtype), codes, "mps")

    assert on_cpu.shape == on_mps.shape
    diff = float(np.abs(on_cpu - on_mps).max())
    tolerance = 1e-5 if dtype == "float32" else 1e-2
    assert diff < tolerance, f"{dtype}: cpu vs mps max abs diff {diff}"
