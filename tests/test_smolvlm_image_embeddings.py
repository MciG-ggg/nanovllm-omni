"""SmolVLM image-embedding assembly: tile folding and row placement.

The processor hands the stage ``pixel_values`` shaped
``[num_images, num_tiles, 3, H, W]``; the SigLIP tower takes a 4-D batch, so
one 512x512 image is 17 tile forwards. Before the fix the stage passed the
5-D tensor straight through and then used only ``image_embeds[0]``, so the
image-token positions all received the first tile.
"""

from types import SimpleNamespace

import pytest
import torch

from nanovllm_omni.models.smolvlm.stage import SmolVLMStage

_IMAGE_TOKEN_ID = 3


class _FakeVision:
    def __init__(self) -> None:
        self.seen_shape: tuple[int, ...] | None = None

    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.seen_shape = tuple(pixel_values.shape)
        if pixel_values.dim() != 4:
            raise AssertionError(f"vision tower takes a 4-D batch, got {self.seen_shape}")
        marker = pixel_values[:, 0, 0, 0].reshape(-1, 1, 1).expand(-1, 1, 2)
        return SimpleNamespace(last_hidden_state=marker)


class _FakeEmbedTokens:
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.full((1, input_ids.shape[1], 2), -1.0)


def _stage() -> tuple[SmolVLMStage, _FakeVision]:
    vision = _FakeVision()
    shell = SimpleNamespace(
        model=SimpleNamespace(
            vision_model=vision,
            connector=lambda features: features,
            text_model=SimpleNamespace(embed_tokens=_FakeEmbedTokens()),
        )
    )
    stage = object.__new__(SmolVLMStage)
    stage.model = shell
    stage.image_token_id = _IMAGE_TOKEN_ID
    return stage, vision


def test_tiles_are_folded_into_the_vision_batch_axis() -> None:
    stage, vision = _stage()
    pixel_values = torch.zeros(1, 3, 3, 2, 2)
    input_ids = torch.tensor([[7, _IMAGE_TOKEN_ID, 8, _IMAGE_TOKEN_ID, 9, _IMAGE_TOKEN_ID]])

    with torch.no_grad():
        stage._build_merged_embeddings(input_ids, pixel_values)

    assert vision.seen_shape == (3, 3, 2, 2)


def test_image_rows_land_in_token_order() -> None:
    stage, _ = _stage()
    pixel_values = torch.zeros(1, 3, 3, 2, 2)
    for tile in range(3):
        pixel_values[0, tile, 0, 0, 0] = float(tile + 1)
    input_ids = torch.tensor([[7, _IMAGE_TOKEN_ID, 8, _IMAGE_TOKEN_ID, 9, _IMAGE_TOKEN_ID]])

    with torch.no_grad():
        merged = stage._build_merged_embeddings(input_ids, pixel_values)

    assert merged.shape == (6, 2)
    assert merged[0].tolist() == [-1.0, -1.0]
    assert merged[1].tolist() == [1.0, 1.0]
    assert merged[2].tolist() == [-1.0, -1.0]
    assert merged[3].tolist() == [2.0, 2.0]
    assert merged[4].tolist() == [-1.0, -1.0]
    assert merged[5].tolist() == [3.0, 3.0]


def test_image_token_count_must_match_the_tiles() -> None:
    stage, _ = _stage()
    pixel_values = torch.zeros(1, 3, 3, 2, 2)
    input_ids = torch.tensor([[7, _IMAGE_TOKEN_ID, 8]])

    with pytest.raises(RuntimeError, match="image-token positions"):
        stage._build_merged_embeddings(input_ids, pixel_values)
