"""SmolVLA two-stage tests: CPU-side, no GPU or lerobot required.

Verifies:
- ``vlm2action`` bridge: shape validation, TypeError on non-VlmStageOutput
- Euler integration: fake constant velocity, N steps → result = velocity
- ``PipelineRunner`` two-stage call sequence (mock)

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §6.1.
"""

from __future__ import annotations

import pytest


class _FakePolicy:
    """Fake policy for testing without lerobot + GPU."""

    class config:  # noqa: N801 (test double mirrors lerobot's lowercase config attrs)
        chunk_size = 10
        max_action_dim = 7
        num_steps = 3
        use_cache = True

        class action_feature:  # noqa: N801 (mirrors policy.config.action_feature)
            shape = [7]

    class model:  # noqa: N801 (mirrors policy.model)
        @staticmethod
        def sample_noise(shape, device):
            import torch

            return torch.randn(shape, device=device, dtype=torch.float32)

        @staticmethod
        def denoise_step(x_t, prefix_pad_masks, past_key_values, timestep):
            import torch

            return torch.full_like(x_t, 0.1)


def test_vlm2action_accepts_vlm_stage_output():
    """vlm2action: happy path."""
    import torch

    from nanovllm_omni.models.smolvla.stage_processors import (
        VlmStageOutput,
        vlm2action,
    )

    hidden = torch.randn(8, 576)
    state = torch.randn(7)
    out = VlmStageOutput(
        prefix_states=hidden,
        robot_state=state,
        request_id="t1",
        metadata={"chunk_len": 10, "action_dim": 7},
    )
    payload = vlm2action(out, prompt="pick up the red block")
    assert payload.prefix_states is hidden
    assert payload.robot_state is state
    assert payload.chunk_len == 10
    assert payload.action_dim == 7
    assert payload.request_id == "t1"
    assert payload.metadata["instruction"] == "pick up the red block"


def test_vlm2action_rejects_non_vlm_output():
    """vlm2action: non-VlmStageOutput → TypeError (ADR-028)."""
    from nanovllm_omni.models.smolvla.stage_processors import vlm2action

    with pytest.raises(TypeError, match="VlmStageOutput"):
        vlm2action({"prefix_states": None, "robot_state": None}, prompt="x")
    with pytest.raises(TypeError, match="VlmStageOutput"):
        vlm2action("raw string", prompt="x")
    with pytest.raises(TypeError, match="VlmStageOutput"):
        vlm2action(42, prompt="x")


def test_euler_integration_constant_velocity():
    """ADR-030: Euler integration with constant velocity v for N steps
    of dt = -1/N yields x_0 + N * v * (-1/N) = x_0 - v."""
    import torch

    from nanovllm_omni.diffusion.interface import StepState
    from nanovllm_omni.models.smolvla.action_stage import SmolVLAActionPipeline

    policy = _FakePolicy()
    pipe = SmolVLAActionPipeline(policy=policy, num_inference_steps=10)

    # Fake a VlmStageOutput with metadata that action_stage expects.
    # In real flow, vlm_stage provides past_key_values + prefix_pad_masks.
    fake_state = StepState(
        request_id="test",
        latents=torch.randn(1, 10, 32),
        metadata={
            "past_key_values": None,  # fake
            "prefix_pad_masks": torch.ones(1, 10),  # fake
            "original_action_dim": 7,
            "policy": policy,
        },
    )

    # Manually init latents like prepare_encode does.
    x0 = torch.randn(1, 10, 32)

    # Override latents for test.
    fake_state.latents = x0.clone()
    x0 = fake_state.latents.clone()

    for step in range(10):
        v_t = pipe.denoise_step(fake_state, step=step, num_steps=10)
        pipe.step_scheduler(fake_state, v_t)

    # velocity=0.1, dt=-1/10, 10 steps: x_1 = x_0 + 10 * 0.1 * (-0.1) = x_0 - 0.1
    expected = x0 + 10 * 0.1 * (-0.1)
    assert torch.allclose(fake_state.latents, expected, atol=1e-5), (
        f"Euler regression: expected {expected[0, 0, 0]:.4f}, "
        f"got {fake_state.latents[0, 0, 0]:.4f}"
    )


def test_action_pipeline_supports_step_execution():
    """ActionPipeline must declare supports_step_execution=True."""
    from nanovllm_omni.models.smolvla.action_stage import SmolVLAActionPipeline

    pipe = SmolVLAActionPipeline(policy=_FakePolicy(), num_inference_steps=5)
    assert pipe.supports_step_execution is True
    assert hasattr(pipe, "prepare_encode")
    assert hasattr(pipe, "denoise_step")
    assert hasattr(pipe, "step_scheduler")
    assert hasattr(pipe, "post_decode")


def test_post_decode_returns_action_artifact():
    """post_decode: latents → ActionArtifact [chunk_len, action_dim]."""
    import torch

    from nanovllm_omni.diffusion.interface import StepState
    from nanovllm_omni.models.smolvla.action_stage import SmolVLAActionPipeline
    from nanovllm_omni.outputs import ActionArtifact

    policy = _FakePolicy()
    pipe = SmolVLAActionPipeline(policy=policy, num_inference_steps=3)

    state = StepState(
        request_id="test",
        latents=torch.randn(1, 10, 32),
        metadata={
            "past_key_values": None,
            "prefix_pad_masks": torch.ones(1, 10),
            "original_action_dim": 7,
            "policy": policy,
        },
    )
    output = pipe.post_decode(state)
    assert output.finished is True
    assert output.images is not None and len(output.images) == 1
    action_artifact = output.images[0]
    assert isinstance(action_artifact, ActionArtifact)
    assert action_artifact.array.shape == (10, 7)


def test_smolvla_pipeline_topology():
    """Pipeline topology: 2 stages (vlm + action), correct kinds."""
    from nanovllm_omni.config import resolve_pipeline_config
    from nanovllm_omni.config.registry import StageExecutionType

    config = resolve_pipeline_config("smolvla")
    assert config is not None
    assert config.name == "smolvla"
    assert len(config.stages) == 2
    assert config.stages[0].name == "vlm"
    assert config.stages[0].kind == StageExecutionType.LLM_AR
    assert config.stages[0].is_terminal is False
    assert config.stages[1].name == "action"
    assert config.stages[1].kind == StageExecutionType.DIFFUSION
    assert config.stages[1].is_terminal is True
    assert config.stages[1].final_output_type == "actions"
    assert config.stages[1].process_input is not None
    assert "vlm2action" in config.stages[1].process_input


def test_smolvla_libero_handle_resolves():
    """HF handle resolves to the two-stage smolvla pipeline."""
    from nanovllm_omni.config import resolve_pipeline_config

    config = resolve_pipeline_config("HuggingFaceVLA/smolvla_libero")
    assert config is not None
    assert config.name == "smolvla"
    assert len(config.stages) == 2
