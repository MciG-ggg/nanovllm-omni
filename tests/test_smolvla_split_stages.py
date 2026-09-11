"""SmolVLA split-stages tests: CPU-side, no GPU or lerobot required.

Verifies:
- ``vlm2action`` bridge: shape validation, TypeError on non-VlmStageOutput
- Euler integration: fake constant velocity, N steps → result = velocity
- ``PipelineRunner`` two-stage call sequence (mock)

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §6.1.
"""

from __future__ import annotations

import pytest


def test_vlm2action_accepts_vlm_stage_output():
    """vlm2action: happy path."""
    import torch

    from nanovllm_omni.models.smolvla.stage_processors import (
        VlmStageOutput,
        vlm2action,
    )

    hidden = torch.randn(8, 576)  # [T, H]
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
    of dt = 1/N yields x_1 = x_0 + N * v * (1/N) = x_0 + v."""
    import torch

    from nanovllm_omni.models.smolvla.action_stage import (
        SmolVLAActionPipeline,
        _FakeActionExpert,
    )
    from nanovllm_omni.models.smolvla.stage_processors import ActionInputPayload

    action_expert = _FakeActionExpert(action_dim=7, chunk_len=10, device="cpu")
    pipe = SmolVLAActionPipeline(action_expert, num_inference_steps=10)

    # Prepare.
    request = ActionInputPayload(
        prefix_states=torch.zeros(8, 576),
        robot_state=torch.zeros(7),
        chunk_len=10,
        action_dim=7,
        request_id="test",
    )
    state = pipe.prepare_encode(request)
    x0 = state.latents.clone()

    # Run 10 steps.
    for step in range(10):
        velocity = pipe.denoise_step(state, step=step, num_steps=10)
        pipe.step_scheduler(state, velocity)

    # x_1 should equal x_0 + v (= x_0 + 0.1)
    expected = x0 + 0.1
    assert torch.allclose(state.latents, expected, atol=1e-5), (
        f"Euler regression: expected {expected[0, 0]:.4f}, " f"got {state.latents[0, 0]:.4f}"
    )


def test_action_pipeline_supports_step_execution():
    """ActionPipeline must declare supports_step_execution=True."""
    from nanovllm_omni.models.smolvla.action_stage import SmolVLAActionPipeline

    # Use any callable as fake action expert.
    pipe = SmolVLAActionPipeline(action_expert=lambda *a, **kw: None, num_inference_steps=5)
    assert pipe.supports_step_execution is True
    # Structural check: implements the protocol (duck-typed).
    assert hasattr(pipe, "prepare_encode")
    assert hasattr(pipe, "denoise_step")
    assert hasattr(pipe, "step_scheduler")
    assert hasattr(pipe, "post_decode")


def test_post_decode_returns_action_artifact():
    """post_decode: latents → ActionArtifact [chunk_len, action_dim]."""
    import torch

    from nanovllm_omni.models.smolvla.action_stage import SmolVLAActionPipeline
    from nanovllm_omni.models.smolvla.stage_processors import ActionInputPayload
    from nanovllm_omni.outputs import ActionArtifact

    pipe = SmolVLAActionPipeline(
        action_expert=lambda *a, **kw: torch.zeros(10, 7),
        num_inference_steps=3,
    )
    request = ActionInputPayload(
        prefix_states=torch.zeros(8, 576),
        robot_state=torch.zeros(7),
        chunk_len=10,
        action_dim=7,
    )
    state = pipe.prepare_encode(request)
    output = pipe.post_decode(state)
    assert output.finished is True
    assert output.images is not None
    assert len(output.images) == 1
    action_artifact = output.images[0]
    assert isinstance(action_artifact, ActionArtifact)
    assert action_artifact.array.shape == (10, 7)


def test_smolvla_split_pipeline_topology():
    """Pipeline topology: 2 stages (vlm + action), correct kinds."""
    from nanovllm_omni.config import resolve_pipeline_config
    from nanovllm_omni.config.registry import StageExecutionType

    config = resolve_pipeline_config("smolvla_split")
    assert config is not None
    assert config.name == "smolvla_split"
    assert len(config.stages) == 2
    assert config.stages[0].name == "vlm"
    assert config.stages[0].kind == StageExecutionType.LLM_AR
    assert config.stages[0].is_terminal is False
    assert config.stages[1].name == "action"
    assert config.stages[1].kind == StageExecutionType.DIFFUSION
    assert config.stages[1].is_terminal is True
    assert config.stages[1].final_output_type == "actions"
    # Bridge is wired.
    assert config.stages[1].process_input is not None
    assert "vlm2action" in config.stages[1].process_input


def test_smolvla_legacy_pipeline_unchanged():
    """Legacy single-stage pipeline still works (ADR-031 backward compat)."""
    from nanovllm_omni.config import resolve_pipeline_config
    from nanovllm_omni.config.registry import StageExecutionType

    config = resolve_pipeline_config("smolvla")
    assert config is not None
    assert len(config.stages) == 1
    assert config.stages[0].name == "vla"
    assert config.stages[0].kind == StageExecutionType.LLM_GENERATION
    assert config.stages[0].is_terminal is True
    assert config.stages[0].final_output_type == "actions"
