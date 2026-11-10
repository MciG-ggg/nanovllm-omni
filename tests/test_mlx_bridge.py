"""The MLX thinker's bridge rows must match the vendor's.

The vendor takes the hidden states after layer ``bridge_layer`` and before
the final norm; the fork does the same. A runner that returns the model's
output instead feeds the talker a different tensor, and the divergence
only shows up as wrong audio much later -- so it is pinned here against
the vendor's own code.
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("mlx.core")

MODEL_DIR = os.path.expanduser("~/models/minimind-3o")
MLX_THINKER = os.path.join(MODEL_DIR, "mlx-thinker")
PROMPT = "今天天气怎么样？"


def _require_checkpoint() -> str:
    if not os.path.isdir(MODEL_DIR) or not os.path.isdir(MLX_THINKER):
        pytest.skip(f"minimind-3o checkpoint or mlx-thinker not found under {MODEL_DIR}")
    return MODEL_DIR


def _token_ids() -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return tokenizer.encode(PROMPT, add_special_tokens=False)[:16]


def _vendor_bridge(token_ids: list[int]) -> object:
    """The vendor's own bridge rows, straight from ``model_omni.py``."""
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_cls = get_class_from_dynamic_module("model_omni.OmniConfig", MODEL_DIR)
    model_cls = get_class_from_dynamic_module("model_omni.MiniMindOmni", MODEL_DIR)
    config = config_cls.from_pretrained(MODEL_DIR)
    model = model_cls(config)
    state = torch.load(
        os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=False)
    model = model.eval().half()

    captured: dict[str, object] = {}
    layer = model.thinker.layers[config.bridge_layer]
    original = layer.forward

    def spy(hidden, *args, **kwargs):
        out = original(hidden, *args, **kwargs)
        captured["rows"] = out[0] if isinstance(out, tuple) else out
        return out

    layer.forward = spy
    with torch.no_grad():
        model.thinker(torch.tensor([token_ids], dtype=torch.long))
    return captured["rows"]


def _mlx_bridge(token_ids: list[int]) -> object:
    """The MLX runner's recorded bridge rows."""
    from nanovllm_omni.backend import get_backend

    backend = get_backend("mlx", device="gpu", dtype="float16")
    config = backend.make_config(model_path=MLX_THINKER)
    runner = backend.make_stage_runner(model_class=None, config=config)
    ids = mx.array([token_ids], dtype=mx.int32)
    runner.set_context(is_prefill=True)
    runner.forward(ids, mx.arange(len(token_ids))[None, :], True)
    return runner.bridge_hidden()


def test_bridge_rows_match_the_vendor() -> None:
    """BUILD / OPERATE / CHECK: the recorded rows are the vendor's rows."""
    import numpy as np

    _require_checkpoint()
    token_ids = _token_ids()
    vendor = _vendor_bridge(token_ids)
    rows = _mlx_bridge(token_ids)

    assert rows is not None
    assert tuple(rows.shape) == tuple(vendor.shape)

    diff = np.abs(vendor.float().numpy() - np.asarray(rows[0].astype(mx.float32)))
    # One fp16 step at the bridge's magnitude: the vendor runs in fp16 and
    # so does the MLX runner, so they cannot agree any closer than that.
    magnitude = float(vendor.float().abs().max())
    assert diff.max() <= 0.0625, f"bridge max abs diff {diff.max()} at magnitude {magnitude}"


def test_bridge_is_not_the_model_output() -> None:
    """The bridge is taken before the final norm, so it is not the output."""
    _require_checkpoint()
    token_ids = _token_ids()
    rows = _mlx_bridge(token_ids)
    assert rows is not None
    # The final norm scales every row to roughly unit RMS; the bridge rows
    # are raw block outputs and stay much larger.
    scale = float(mx.max(mx.abs(rows[0].astype(mx.float32))))
    assert scale > 5.0, f"bridge rows look normalised (max {scale})"
