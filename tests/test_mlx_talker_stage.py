"""The MLX talker reached through ``TalkerStage``, and what survives the swap.

The sampled codes are **not** comparable across the two runners: torch draws with
``torch.multinomial`` (an exponential race) and MLX with ``mx.random.categorical``
(Gumbel-max), so the same seed aligns the parameters and not the draws (ADR 0023,
measured: one bridge, one seed, 16 frames each, 0 identical frames). What *is*
comparable is the frame count -- the schedule and the emission gate are shared --
and that is what the cross-runner test asserts.

``mlx`` is a Mac-track extra the project venv does not have, so the module-level
guard skips both the MLX and the torch arms on a machine without it. One test is
a strict xfail: the stage forwards an ``impl=`` kwarg that
``MlxTalkerRunner.__init__`` does not accept, so ``extra={"backend": "mlx"}``
cannot build a talker runner today.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx")
torch = pytest.importorskip("torch")

from nanovllm_omni.backend.base import TalkerRequest  # noqa: E402
from nanovllm_omni.backend.mlx_backend import MlxBackend  # noqa: E402
from nanovllm_omni.backend.mlx_talker import MlxTalkerRunner  # noqa: E402
from nanovllm_omni.backend.talker_recipe import NUM_CODEBOOKS  # noqa: E402

MODEL_DIR = os.path.expanduser("~/models/minimind-3o")
MLX_TALKER = os.path.join(MODEL_DIR, "mlx-talker")
HIDDEN = 768
# Measured on both runners at start_pos 16 with 24 steps: 16 frames, 8 codes wide.
START_POS = 16
NUM_STEPS = 24
EXPECTED_FRAMES = NUM_STEPS - NUM_CODEBOOKS
SEED = 42
BRIDGE_SEED = 3
LENGTH = START_POS + NUM_STEPS


def _require_checkpoint() -> None:
    if not os.path.isfile(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        pytest.skip(f"talker checkpoint not found in {MODEL_DIR}")
    if not os.path.isdir(MLX_TALKER):
        pytest.skip(f"converted MLX talker not found at {MLX_TALKER}")


def _bridge() -> torch.Tensor:
    """The one deterministic bridge both runners read, ``[start_pos + steps, 768]``."""
    generator = torch.Generator().manual_seed(BRIDGE_SEED)
    return torch.randn(LENGTH, HIDDEN, generator=generator, dtype=torch.float32)


def _request() -> TalkerRequest:
    return TalkerRequest(
        bridge=_bridge(),
        num_steps=NUM_STEPS,
        start_pos=START_POS,
        temperature=0.2,
        seed=SEED,
    )


class _Args:
    """``OmniEngineArgs`` stand-in: the three fields the stage reads."""

    model = MODEL_DIR
    device = "cpu"
    dtype = "float16"
    extra = {"talker_impl": "rewritten", "talker_kv": "dense", "backend": "mlx"}


class _Sampling:
    """``SamplingParams`` stand-in carrying the vendor audio temperature and seed."""

    extra = {"audio_temperature": 0.2}
    seed = SEED


def _payload():
    """What ``thinker2talker`` hands the talker: bridge rows plus the precomputed split."""
    from nanovllm_omni.models.minimind_omni.stage_processors import TalkerInputPayload

    return TalkerInputPayload(
        input_ids=torch.zeros(LENGTH, dtype=torch.long),
        bridge_states=_bridge(),
        text_token_ids=tuple(range(LENGTH)),
        prompt_token_ids=tuple(range(START_POS)),
        output_token_ids=tuple(range(START_POS, LENGTH)),
        metadata={"start_pos": START_POS, "num_steps": NUM_STEPS},
    )


@pytest.fixture(scope="module")
def mlx_runner() -> MlxTalkerRunner:
    """One converted MLX talker, dense KV; the runner the stage is supposed to build."""
    _require_checkpoint()
    return MlxBackend().make_talker_runner(MLX_TALKER, kv="dense")


def test_the_mlx_backend_builds_an_mlx_talker_runner(mlx_runner) -> None:
    assert isinstance(mlx_runner, MlxTalkerRunner)


def test_the_frames_are_eight_codes_and_reproduce_for_a_seed(mlx_runner) -> None:
    frames = mlx_runner.generate(_request())

    assert len(frames) == EXPECTED_FRAMES
    assert all(len(frame) == NUM_CODEBOOKS for frame in frames)
    assert frames == mlx_runner.generate(_request())


def test_the_frame_count_matches_the_torch_runner(mlx_runner) -> None:
    """Same schedule on both sides: the counts agree though the draws do not."""
    from nanovllm_omni.backend.torch_backend import TorchBackend

    _require_checkpoint()
    torch_runner = TorchBackend(device="cpu", dtype="float16").make_talker_runner(
        MODEL_DIR, impl="rewritten", kv="dense"
    )

    torch_frames = torch_runner.generate(_request())
    mlx_frames = mlx_runner.generate(_request())

    assert len(mlx_frames) == len(torch_frames) == EXPECTED_FRAMES


def test_the_stage_routes_through_the_mlx_backend(monkeypatch) -> None:
    """``extra['backend']='mlx'`` reaches ``get_backend('mlx')`` with the MLX directory."""
    from nanovllm_omni.models.minimind_omni import stage as stage_module
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage

    _require_checkpoint()
    consulted: dict[str, object] = {}

    class _FakeBackend:
        def make_talker_runner(self, model_path, **kwargs):
            consulted["running"] = model_path
            return object()

    def _get_backend(name=None, **kwargs):
        consulted["name"] = name
        return _FakeBackend()

    monkeypatch.setattr(stage_module, "get_backend", _get_backend)
    TalkerStage(deploy=None, args=_Args())._ensure_runner()

    assert consulted["name"] == "mlx"
    assert consulted["running"] == MLX_TALKER


def test_the_stage_produces_frames_once_the_impl_kwarg_is_dropped(monkeypatch) -> None:
    """The stage path holds end to end apart from the ``impl=`` kwarg; see the xfail."""
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage

    _require_checkpoint()
    make_runner = MlxBackend.make_talker_runner

    def _without_impl(self, model_path, **kwargs):
        kwargs.pop("impl", None)
        return make_runner(self, model_path, **kwargs)

    monkeypatch.setattr(MlxBackend, "make_talker_runner", _without_impl)
    stage = TalkerStage(deploy=None, args=_Args())

    output = stage(_payload(), _Sampling())

    assert isinstance(stage._runner, MlxTalkerRunner)
    assert output.audio_codes.shape == (EXPECTED_FRAMES, NUM_CODEBOOKS)
    assert output.audio_codes.tolist() == stage._runner.generate(_request())


def test_the_stage_can_build_its_mlx_talker_runner_unpatched() -> None:
    """``extra['backend']='mlx'`` must build ``MlxTalkerRunner`` with no local shim."""
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage

    _require_checkpoint()

    stage = TalkerStage(deploy=None, args=_Args())

    assert isinstance(stage._ensure_runner(), MlxTalkerRunner)
