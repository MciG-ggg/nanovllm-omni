"""SmolVLA action parity vs lerobot under fixed inputs (docs/adr/0011-parity-standard.md P2)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

MODEL = os.environ.get("SMOLVLA_MODEL_DIR", "/home/mcig/models/smolvla_libero")
PROMPT = "pick up the red block"
ATOL = 1e-4
NUM_STEPS = 10


def _skip_unless_ready() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not os.path.isdir(MODEL):
        pytest.skip(f"SmolVLA checkpoint not found: {MODEL}")


def _image(seed: int) -> object:
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def _args() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        model=MODEL,
        device="cuda",
        extra={"allow_hf_download": False},
    )


@pytest.mark.smoke
def test_smolvla_action_parity() -> None:
    _skip_unless_ready()

    import numpy as np
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from nanovllm_omni import SamplingParams
    from nanovllm_omni.models.smolvla.action_stage import _action_stage
    from nanovllm_omni.models.smolvla.stage_processors import vlm2action
    from nanovllm_omni.models.smolvla.vlm_stage import _vlm_stage

    image = _image(0)
    wrist = _image(1)
    state = np.zeros(8, dtype=np.float32)
    args = _args()

    policy = SmolVLAPolicy.from_pretrained(MODEL, local_files_only=True, strict=True).to("cuda")
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL)
    batch = preprocessor(
        {
            "observation.images.image": torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0),
            "observation.images.image2": torch.from_numpy(wrist).permute(2, 0, 1).unsqueeze(0),
            "observation.state": torch.from_numpy(state).reshape(1, -1),
            "task": [PROMPT],
        }
    )
    torch.manual_seed(0)
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch).cpu().numpy()[0]

    vlm = _vlm_stage(None, args)
    action = _action_stage(None, args)
    sampling = SamplingParams(
        extra={
            "image": image,
            "wrist_image": wrist,
            "state": state,
            "num_inference_steps": NUM_STEPS,
        }
    )
    torch.manual_seed(0)
    payload = vlm2action(vlm(PROMPT, sampling), PROMPT)
    step_state = action.prepare(payload)
    steps = step_state.metadata["num_steps"]
    for step in range(steps):
        action.step_scheduler(
            step_state, action.denoise_step(step_state, step=step, num_steps=steps)
        )
    actual = action.post_decode(step_state).images[0].array

    diff = np.abs(expected.astype("float64") - actual.astype("float64"))

    assert actual.shape == expected.shape
    assert diff.max() <= ATOL, f"max action diff {diff.max():.3e} > {ATOL:.0e}"
