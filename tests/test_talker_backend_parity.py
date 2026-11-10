"""The talker runner must reproduce the frames the stage used to emit.

The fixture was captured from the stage's own decode loop before the
loop moved into the backend. Bitwise equality here is the evidence that
the move changed nothing -- the CUDA gates cannot run on a Mac, so this
is the local half of the same question.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

MODEL_DIR = os.path.expanduser("~/models/minimind-3o")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "talker_golden_cpu.json")
PROMPT_LEN = 12
HIDDEN = 768

CASES = ["dense_seed42_steps12", "dense_seed7_steps20", "windowed_seed42_steps12"]


def _require_checkpoint() -> str:
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"minimind-3o checkpoint not found at {MODEL_DIR}")
    return MODEL_DIR


def _fixture() -> dict:
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _bridge(seed: int, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(length, HIDDEN, generator=generator, dtype=torch.float32)


@pytest.mark.parametrize("case", CASES)
def test_runner_reproduces_the_pre_backend_frames(case: str) -> None:
    """BUILD / OPERATE / CHECK: the runner's frames equal the captured ones."""
    from nanovllm_omni.backend import get_backend
    from nanovllm_omni.backend.base import TalkerRequest

    entry = _fixture()[case]
    model_dir = _require_checkpoint()
    backend = get_backend("torch", device="cpu", dtype="float16")
    runner = backend.make_talker_runner(model_dir, impl="rewritten", kv=entry["kv"])

    bridge = _bridge(entry["bridge_seed"], PROMPT_LEN + entry["num_steps"])
    frames = runner.generate(
        TalkerRequest(
            bridge=bridge,
            num_steps=entry["num_steps"],
            start_pos=PROMPT_LEN,
            temperature=0.2,
            seed=entry["seed"],
            repetition_penalty=1.05,
        )
    )
    assert frames == entry["frames"]


def test_the_stage_and_the_runner_agree() -> None:
    """The stage is a thin shell: it must produce the runner's frames."""
    from nanovllm_omni.backend import get_backend
    from nanovllm_omni.backend.base import TalkerRequest
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage
    from nanovllm_omni.models.minimind_omni.stage_processors import TalkerInputPayload

    entry = _fixture()["dense_seed42_steps12"]
    model_dir = _require_checkpoint()
    length = PROMPT_LEN + entry["num_steps"]
    bridge = _bridge(entry["bridge_seed"], length)

    class _Args:
        model = model_dir
        device = "cpu"
        dtype = "float16"
        extra = {"talker_impl": "rewritten", "talker_kv": "dense"}

    class _Sampling:
        extra = {"audio_temperature": 0.2}
        seed = entry["seed"]

    payload = TalkerInputPayload(
        input_ids=torch.zeros(length, dtype=torch.long),
        bridge_states=bridge,
        text_token_ids=tuple(range(length)),
        prompt_token_ids=tuple(range(PROMPT_LEN)),
        output_token_ids=tuple(range(PROMPT_LEN, length)),
        metadata={"start_pos": PROMPT_LEN, "num_steps": entry["num_steps"]},
    )
    stage = TalkerStage(deploy=None, args=_Args())
    stage_output = stage(payload, _Sampling())

    backend = get_backend("torch", device="cpu", dtype="float16")
    runner = backend.make_talker_runner(model_dir, impl="rewritten", kv="dense")
    frames = runner.generate(
        TalkerRequest(
            bridge=bridge,
            num_steps=entry["num_steps"],
            start_pos=PROMPT_LEN,
            temperature=0.2,
            seed=entry["seed"],
            repetition_penalty=1.05,
        )
    )
    assert stage_output.audio_codes.tolist() == frames
