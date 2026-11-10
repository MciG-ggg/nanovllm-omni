"""The MLX thinker stage: decode a prompt and hand the talker its bridge rows.

The MLX track runs the same vendor decode dance as the CUDA one, so the
stage has to produce the same two things: the text, and the bridge rows
the talker cross-attends to. The bridge is the hidden state after layer
``bridge_layer`` and before the final norm -- the rows the vendor's own
thinker produces, which ``tests/test_mlx_bridge.py`` pins for the
backend's recorder; here the rows travel the whole way through the
stage, decode included.

``torch`` / ``transformers`` / ``numpy`` are imported inside the test
bodies: they are heavy, and the file's own ``mlx`` guard already skips
the whole module when the Mac track is not installed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("mlx")

MODEL_DIR = os.path.expanduser("~/models/minimind-3o")
MLX_THINKER = os.path.join(MODEL_DIR, "mlx-thinker")
PROMPT = "今天天气怎么样？"
MAX_TOKENS = 16
SEED = 42


def _require_checkpoint() -> None:
    if not os.path.isdir(MODEL_DIR) or not os.path.isdir(MLX_THINKER):
        pytest.skip(f"minimind-3o checkpoint or mlx-thinker not found under {MODEL_DIR}")


@pytest.fixture(scope="module")
def decoded() -> Any:
    """One MLX thinker decode of a short Chinese prompt."""
    _require_checkpoint()
    from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
    from nanovllm_omni.models.minimind_omni.stage import ThinkerStage

    args = OmniEngineArgs(model=MODEL_DIR, extra={"backend": "mlx"})
    stage = ThinkerStage(deploy=None, args=args)
    sampling = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=MAX_TOKENS, seed=SEED)
    return stage(PROMPT, sampling)


def _vendor_bridge(token_ids: list[int]) -> Any:
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

    captured: dict[str, Any] = {}
    layer = model.thinker.layers[config.bridge_layer]
    original = layer.forward

    def spy(hidden, *args, **kwargs):
        out = original(hidden, *args, **kwargs)
        captured["rows"] = out[0] if isinstance(out, tuple) else out
        return out

    layer.forward = spy
    with torch.no_grad():
        model.thinker(torch.tensor([token_ids], dtype=torch.long))
    return captured["rows"][0].float().numpy()


def test_mlx_thinker_decodes_text(decoded: Any) -> None:
    """CHECK: the stage returns the thinker's text and its token ids."""
    assert decoded.text
    assert len(decoded.output_token_ids) == MAX_TOKENS
    assert len(decoded.prompt_token_ids) > 0


def test_mlx_bridge_has_one_row_per_thinker_position(decoded: Any) -> None:
    """CHECK: one bridge row per thinker position, ``[T, 768]``."""
    positions = len(decoded.prompt_token_ids) + len(decoded.output_token_ids)

    assert tuple(decoded.bridge_states.shape) == (positions, 768)


def test_mlx_bridge_matches_the_vendor(decoded: Any) -> None:
    """CHECK: the rows are the vendor's rows."""
    import numpy as np

    token_ids = list(decoded.prompt_token_ids) + list(decoded.output_token_ids)
    vendor = _vendor_bridge(token_ids)
    diff = np.abs(vendor - decoded.bridge_states.numpy())

    # One fp16 step at the bridge's magnitude: the vendor runs in fp16 and
    # so does the MLX runner, so they cannot agree any closer than that.
    magnitude = float(np.abs(vendor).max())
    assert diff.max() <= magnitude * 5e-4, f"bridge max abs diff {diff.max()} at {magnitude}"
