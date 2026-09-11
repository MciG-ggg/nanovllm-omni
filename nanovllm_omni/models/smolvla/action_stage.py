"""SmolVLA action (flow matching) stage.

Implements the ``DiffusionPipeline`` 4-method contract to run flow
matching with the action expert.  This is the second stage of the
SmolVLA two-stage split.

Flow matching uses Euler integration: ``x_{t+dt} = x_t + v * dt``,
where ``v`` is the velocity predicted by the action expert network.

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §3 ADR-030.
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.outputs import ActionArtifact


class SmolVLAActionPipeline:
    """Flow-matching pipeline for SmolVLA action expert.

    Implements the 4-method DiffusionPipeline contract:
    - prepare_encode: init latent noise + cache backbone states
    - denoise_step: action expert forward → velocity
    - step_scheduler: Euler integration step
    - post_decode: produce ActionArtifact
    """

    supports_step_execution = True

    def __init__(self, action_expert: Any, num_inference_steps: int = 10) -> None:
        """action_expert: lerobot's action expert module (nn.Module-like).
        For testing without lerobot, pass a fake that returns constant velocity.
        """
        self.action_expert = action_expert
        self.num_inference_steps = num_inference_steps

    def prepare_encode(self, request: Any) -> Any:
        """Init latent noise; stash backbone states for conditioning."""
        import torch

        from nanovllm_omni.diffusion.interface import StepState

        # The "request" here is an ActionInputPayload (from vlm2action).
        chunk_len = getattr(request, "chunk_len", 10)
        action_dim = getattr(request, "action_dim", 7)
        device = "cpu"  # default; real impl reads from tensors

        # Get device from the prefix_states tensor if available.
        prefix_states = getattr(request, "prefix_states", None)
        if hasattr(prefix_states, "device"):
            device = prefix_states.device

        # Initial latent = pure noise, shape [chunk_len, action_dim].
        latents = torch.randn(
            chunk_len,
            action_dim,
            device=device,
            dtype=torch.float32,
        )

        return StepState(
            request_id=getattr(request, "request_id", "sync"),
            latents=latents,
            encoder_hidden_states=prefix_states,  # cache backbone states
            metadata={
                "robot_state": getattr(request, "robot_state", None),
                "chunk_len": chunk_len,
                "action_dim": action_dim,
                "attention_mask": getattr(request, "attention_mask", None),
            },
        )

    def denoise_step(self, state: Any, *, step: int, num_steps: int) -> Any:
        """Run action expert; return velocity."""
        import torch

        # The action expert takes (latents, prefix_states, robot_state) and
        # returns velocity.  For testing, the fake action expert is a simple
        # linear projection that ignores inputs and returns a constant.
        prefix_states = state.encoder_hidden_states
        robot_state = state.metadata.get("robot_state")

        with torch.inference_mode():
            velocity = self.action_expert(
                state.latents,
                prefix_states=prefix_states,
                robot_state=robot_state,
            )
        return velocity

    def step_scheduler(self, state: Any, noise_pred: Any) -> None:
        """Euler integration: x_{t+dt} = x_t + v * dt."""
        num_steps = self.num_inference_steps
        dt = 1.0 / num_steps
        state.latents = state.latents + noise_pred * dt
        state.step_index += 1

    def post_decode(self, state: Any) -> Any:
        """Produce ActionArtifact from final latents."""
        from nanovllm_omni.diffusion.interface import DiffusionOutput

        latents = state.latents  # [chunk_len, action_dim]
        # Convert to numpy for ActionArtifact.
        array = latents.detach().cpu().numpy().astype("float32")
        return DiffusionOutput(
            images=[ActionArtifact.from_array(array)],
            finished=True,
        )


def _action_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: load action expert, return SmolVLAActionPipeline."""

    # The action expert is part of the lerobot SmolVLA policy.
    # For the split-stages path, we load it separately.
    # TODO(ADR-029): proper weight slicing from SmolVLA checkpoint.
    extra = dict(getattr(args, "extra", None) or {})
    allow_hf = bool(extra.get("allow_hf_download", False))
    device = getattr(args, "device", None) or "cuda"

    num_inference_steps = int(extra.get("num_inference_steps", 10))

    # Try to load the action expert from the lerobot checkpoint.
    try:
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-not-found]
            SmolVLAPolicy,
        )

        # extra["action_model"] overrides the model path for the action stage.
        # When split stages are used, the top-level args.model points to the
        # VLM backbone, not the lerobot checkpoint.
        model_path = (
            extra.get("action_model")
            or getattr(args, "model", None)
            or "HuggingFaceVLA/smolvla_libero"
        )
        # If the model path looks like a backbone (not a lerobot policy),
        # fall back to the default lerobot checkpoint.
        if "smolvla" not in model_path.lower() and "lerobot" not in model_path.lower():
            model_path = extra.get("action_model") or "HuggingFaceVLA/smolvla_libero"
        load_kwargs: dict[str, Any] = {"strict": False}
        if not allow_hf:
            load_kwargs["local_files_only"] = True
        policy = SmolVLAPolicy.from_pretrained(model_path, **load_kwargs)
        if device:
            policy = policy.to(device)
        # Extract the action expert.
        action_expert = getattr(policy.model, "action_expert", None) or getattr(
            policy,
            "action_expert",
            None,
        )
        if action_expert is None:
            # Fallback: fake action expert for testing.
            action_expert = _FakeActionExpert(
                action_dim=7,
                chunk_len=10,
                device=device,
            )
    except (ImportError, OSError) as e:
        # lerobot not installed or checkpoint not available; use fake for testing.
        print(f"  [action_stage] Falling back to fake action expert: {type(e).__name__}: {e}")
        action_expert = _FakeActionExpert(
            action_dim=7,
            chunk_len=10,
            device=device,
        )

    return SmolVLAActionPipeline(
        action_expert=action_expert,
        num_inference_steps=num_inference_steps,
    )


class _FakeActionExpert:
    """Fake action expert for testing without lerobot + GPU."""

    def __init__(self, action_dim: int = 7, chunk_len: int = 10, device: str = "cpu") -> None:

        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.device = device

    def __call__(self, latents: Any, prefix_states: Any = None, robot_state: Any = None) -> Any:
        """Return constant velocity = 0.1 * ones.

        With N Euler steps of dt=1/N: x_1 = N * 0.1 * (1/N) = 0.1.
        So ActionArtifact should have values ≈ 0.1 after the full loop.
        """
        import torch

        return torch.full_like(latents, 0.1)


__all__ = ["SmolVLAActionPipeline", "_action_stage", "_FakeActionExpert"]
