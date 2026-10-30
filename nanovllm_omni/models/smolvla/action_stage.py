"""SmolVLA action (flow matching) stage.

Implements the ``DiffusionPipeline`` 4-method contract to run flow
matching with the action expert. This is the second stage of the
SmolVLA two-stage pipeline.

Uses lerobot's internal ``denoise_step`` + ``euler_integrate`` to
maintain bit-exact alignment with ``policy.predict_action_chunk``.

See ``docs/dev/nanovllm-omni-smolvla-arflow-migration.md`` §3 ADR-030.
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.diffusion.runner import DiffusionRunner
from nanovllm_omni.outputs import ActionArtifact


class SmolVLAActionPipeline:
    """Flow-matching pipeline for SmolVLA action expert.

    Implements the 4-method DiffusionPipeline contract:
    - prepare_encode: init latent noise from KV cache state
    - denoise_step: action expert forward → velocity
    - step_scheduler: Euler integration step
    - post_decode: produce ActionArtifact
    """

    supports_step_execution = True

    def __init__(
        self,
        policy: Any,
        num_inference_steps: int | None = None,
        compile_flow: bool = False,
    ) -> None:
        """policy: lerobot SmolVLAPolicy (provides denoise_step + sample_noise)."""
        if policy is None:
            raise ValueError("SmolVLAActionPipeline requires a policy (got None)")
        self.policy = policy
        self.config = policy.config
        self.num_inference_steps = num_inference_steps or self.config.num_steps
        self.compile_flow = compile_flow
        self._compiled_step: Any = None

    def prepare_encode(self, request: Any, sampling: Any = None) -> Any:
        """Init latent noise; stash VLM KV cache state for conditioning.

        The ``request`` is an ``ActionInputPayload`` from ``vlm2action``.
        Prefer the policy from the payload metadata (vlm_stage's policy)
        over self.policy to ensure bit-exact alignment.
        """
        from nanovllm_omni.diffusion.interface import StepState

        metadata = dict(request.metadata) if hasattr(request, "metadata") else {}
        past_key_values = metadata.get("past_key_values")
        prefix_pad_masks = metadata.get("prefix_pad_masks")
        if past_key_values is None or prefix_pad_masks is None:
            raise ValueError("ActionInputPayload missing past_key_values or prefix_pad_masks")

        # Policy identity was fixed at construction (see policy.py): the
        # vlm_stage's instance arrives in metadata, self.policy is the
        # same object in production (same cache key) and the test double
        # in unit tests. No third option — a missing policy is a bug.
        policy = metadata.get("policy") or metadata.get("policy_ref") or self.policy
        if policy is None:
            raise ValueError("ActionInputPayload carries no policy and stage has none")
        config = policy.config

        # Sample initial noise (same as lerobot's sample_noise).
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        actions_shape = (bsize, config.chunk_size, config.max_action_dim)
        x_t = policy.model.sample_noise(actions_shape, device)

        return StepState(
            request_id=getattr(request, "request_id", "sync"),
            latents=x_t,
            encoder_hidden_states=None,
            metadata={
                "past_key_values": past_key_values,
                "prefix_pad_masks": prefix_pad_masks,
                "original_action_dim": metadata.get(
                    "original_action_dim", config.action_feature.shape[0]
                ),
                "num_steps": metadata.get("num_steps", config.num_steps),
                "policy": policy,
            },
        )

    def _step_callable(self, policy: Any) -> Any:
        """The one place the per-step callable is chosen.

        One flow step costs ~125 ms while the GPU is busy ~9 ms of it, and a
        single step issues ~2400 kernel launches (ADR-0009). ``torch.compile``
        is the only lever that does not need the dependency changed: CUDA
        Graph capture is refused by lerobot's unpinned CPU->CUDA copy.
        """
        if not self.compile_flow:
            return policy.model.denoise_step
        if self._compiled_step is None:
            import torch

            self._compiled_step = torch.compile(policy.model.denoise_step)
        return self._compiled_step

    def denoise_step(self, state: Any, *, step: int, num_steps: int) -> Any:
        """Run action expert; return velocity.

        Matches lerobot's ``denoise_step(x_t, prefix_pad_masks, past_kv, timestep)``
        with timestep = 1.0 + step * (-1/num_steps) (euler_integrate convention).
        """
        import torch

        dt = -1.0 / num_steps
        time = 1.0 + step * dt
        bsize = state.latents.shape[0]
        device = state.latents.device
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        policy = state.metadata["policy"]
        v_t = self._step_callable(policy)(
            x_t=state.latents,
            prefix_pad_masks=state.metadata["prefix_pad_masks"],
            past_key_values=state.metadata["past_key_values"],
            timestep=time_tensor,
        )
        state.metadata["dt"] = dt
        return v_t

    def step_scheduler(self, state: Any, noise_pred: Any) -> None:
        """Euler integration: x_t = x_t + dt * v_t."""
        dt = state.metadata["dt"]
        state.latents = state.latents + noise_pred * dt
        state.step_index += 1

    def post_decode(self, state: Any) -> Any:
        """Truncate to original_action_dim → ActionArtifact."""
        from nanovllm_omni.diffusion.interface import DiffusionOutput

        original_action_dim = state.metadata["original_action_dim"]
        actions = state.latents[:, :, :original_action_dim]
        # Squeeze batch dim for ActionArtifact (expects 2-D [chunk, dim]).
        if actions.dim() == 3 and actions.shape[0] == 1:
            actions = actions[0]
        array = actions.detach().cpu().numpy().astype("float32")
        return DiffusionOutput(
            images=[ActionArtifact.from_array(array)],
            finished=True,
        )


def _action_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory: return SmolVLAActionPipeline.

    Same shared policy instance as vlm_stage (same cache key → same
    object), so no second ~2GB load on the 4GB card and identical
    sample_noise RNG state. A missing lerobot install or an unreadable
    checkpoint raises here; substituting a constant-velocity fake would
    return plausible-looking garbage actions instead.
    """
    from .policy import get_policy

    extra = dict(getattr(args, "extra", None) or {})
    num_inference_steps = int(extra.get("num_inference_steps", 0)) or None

    return DiffusionRunner(
        SmolVLAActionPipeline(
            policy=get_policy(args, extra),
            num_inference_steps=num_inference_steps,
            compile_flow=bool(extra.get("compile_flow", False)),
        )
    )


__all__ = ["SmolVLAActionPipeline", "_action_stage"]
