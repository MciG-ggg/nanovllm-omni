"""The MLX-VLM SmolVLM track: the forward width, the merge shape, greedy text.

There is no cross-framework text comparison here on purpose. The torch SmolVLM
arm rides the fork's triton attention kernel and triton has no macOS wheel, so
the second arm of ``tools/parity_check.py`` cannot run on this Mac; a verdict
(EXACT / TOP_K_MATCH / FAIL) built from one arm would claim a comparison that
never happened. What is pinned instead is everything the MLX arm alone must
hold: the logits width, one merged-embedding row per prompt token at the
config's hidden width, and coherent greedy text for a solid-colour image.

``mlx`` / ``mlx-vlm`` are Mac-track extras the project venv does not have, so
they are guarded per test rather than at module level: a module-level
``importorskip`` would also skip the backend-capability tests, which are exactly
the part that has to hold on a machine without mlx.
"""

from __future__ import annotations

import os

import pytest

from nanovllm_omni.backend import get_backend
from nanovllm_omni.backend.mlx_vlm import MlxVlmBackend
from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams

MODEL_DIR = os.environ.get(
    "SMOLVLM_MLX_MODEL_DIR", os.path.expanduser("~/models/smolvlm-500m-instruct")
)
# Measured on the 500M checkpoint: ``config.vocab_size`` and the tokenizer both
# report 49280, and ``config.text_config.hidden_size`` is 960.
VOCAB_SIZE = 49280
HIDDEN_SIZE = 960
# Measured: the processor expands one 512x512 image into 1088 image-token slots
# out of 1139 prompt tokens.
IMAGE_TOKENS = 1088


def _solid(colour: tuple[int, int, int]) -> object:
    """A 512x512 solid-colour image; skips the test when PIL is absent."""
    pil_image = pytest.importorskip("PIL.Image")
    return pil_image.new("RGB", (512, 512), colour)


@pytest.fixture(scope="module")
def stage():
    """The SmolVLM stage on the MLX-VLM track; skips without mlx or the weights."""
    pytest.importorskip("mlx")
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"MLX SmolVLM weights not found at {MODEL_DIR}")
    from nanovllm_omni.models.smolvlm.stage import SmolVLMStage

    args = OmniEngineArgs(
        model=MODEL_DIR, dtype="float16", device="cpu", extra={"backend": "mlx-vlm"}
    )
    return SmolVLMStage(deploy=None, args=args)


def test_capabilities_are_the_mlx_vlm_ones() -> None:
    capabilities = get_backend("mlx-vlm").capabilities

    assert capabilities.varlen is False
    assert capabilities.graph is True
    assert capabilities.max_batch == 1


def test_a_cuda_device_request_lands_on_the_mlx_gpu() -> None:
    # MLX has no CUDA device, so the substitution is explicit instead of a
    # surprise on the first forward.
    assert MlxVlmBackend(device="cuda").device == "gpu"
    assert MlxVlmBackend(device="cpu").device == "cpu"


def test_the_vlm_backend_has_no_talker_runner() -> None:
    # SmolVLM is one stage; a talker request is a wiring mistake to fail loudly.
    with pytest.raises(NotImplementedError):
        get_backend("mlx-vlm").make_talker_runner(MODEL_DIR)


def test_prefill_returns_one_full_vocabulary_logit_row(stage) -> None:
    mx = pytest.importorskip("mlx.core")

    input_ids, pixel_values = stage.vlm.encode("Say hello.", [])

    logits = stage.stage_runner.forward(input_ids, None, True)

    assert pixel_values is None
    assert logits.shape == (1, VOCAB_SIZE)
    assert stage.stage_runner.model.config.vocab_size == VOCAB_SIZE
    assert len(stage.processor.tokenizer) == VOCAB_SIZE
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_a_solid_image_merges_into_one_row_per_prompt_token(stage) -> None:
    image = _solid((200, 30, 30))

    input_ids, pixel_values = stage.vlm.encode("What colour is this image?", [image])
    merged = stage.vlm.merged_embeddings(input_ids, pixel_values)

    assert int((input_ids == stage.image_token_id).sum().item()) == IMAGE_TOKENS
    assert merged.shape[0] == input_ids.shape[1]
    assert merged.shape[1] == HIDDEN_SIZE
    assert stage.model.config.text_config.hidden_size == HIDDEN_SIZE


def test_greedy_decode_describes_a_solid_red_image(stage) -> None:
    image = _solid((200, 30, 30))
    sampling = SamplingParams(temperature=0.0, max_tokens=24, seed=0, extra={"images": [image]})

    text = stage("What colour is this image?", sampling)

    assert text.strip()
    assert "red" in text.lower()
