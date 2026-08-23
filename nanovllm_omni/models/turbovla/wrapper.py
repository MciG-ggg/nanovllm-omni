"""Thin wrapper around H-EmbodVis/TurboVLA's ``TurboVLAPolicy``.

Upstream package: github.com/H-EmbodVis/TurboVLA (Apache-2.0).
The upstream ``TurboVLAPolicy`` class already owns:

- 256x256 RGB image preprocessing (DINOv3 manual normalizer)
- State normalization (PROPRIO_MEAN / PROPRIO_STD baked in)
- Action denormalization (ACTION_MIN / ACTION_MAX baked in)
- 180-degree image rotation for LIBERO camera convention

This wrapper only adapts its ``predict_*`` calls to nanovllm-omni's
``OmniRequestOutput`` contract with a single ``ActionArtifact`` entry
(see TK-015-rev / ``docs/contracts/multimodal_output.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from nanovllm_omni.outputs import OmniRequestOutput


@dataclass(frozen=True)
class TurboVLAConfig:
    """Inputs needed to construct a TurboVLAPolicy.

    ``ckpt_path`` is the .pth checkpoint under
    ``H-EmbodVis/TurboVLA/pretrained/TurboVLA/checkpoints/libero/``.
    ``dinov3_path`` and ``bert_path`` are local directories of those
    encoders (download separately via ``huggingface-cli download``).
    """

    ckpt_path: str
    dinov3_path: str
    bert_path: str
    device: str | None = None
    precision: str = "bf16"
    allow_hf_download: bool = False


class TurboVLAOmni:
    """Nanovllm-omni adapter for the upstream TurboVLAPolicy.

    Usage::

        from nanovllm_omni.models.turbovla import TurboVLAOmni, TurboVLAConfig
        model = TurboVLAOmni(TurboVLAConfig(ckpt_path=..., dinov3_path=..., bert_path=...))
        out = model.predict(primary_img, wrist_img, instruction, state, execute_steps=12)
        action = out.multimodal_output["action"]   # ActionArtifact
    """

    config_cls = TurboVLAConfig

    def __init__(self, config: TurboVLAConfig) -> None:
        self.config = config
        self._policy: Any = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from turbovla.evaluation.policy import TurboVLAPolicy  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "turbovla package not installed. Run scripts/setup_turbovla.sh "
                "or: pip install -e '.[turbovla]' && "
                "pip install git+https://github.com/H-EmbodVis/TurboVLA.git"
            ) from e

        for label, path in (
            ("checkpoint", self.config.ckpt_path),
            ("DINOv3 directory", self.config.dinov3_path),
            ("BERT directory", self.config.bert_path),
        ):
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"TurboVLA {label} not found: {path}. "
                    f"Run scripts/setup_turbovla.sh to download."
                )

        self._policy = TurboVLAPolicy(
            ckpt_path=self.config.ckpt_path,
            dinov3_path=self.config.dinov3_path,
            bert_path=self.config.bert_path,
            device=self.config.device,
            precision=self.config.precision,
            allow_hf_download=self.config.allow_hf_download,
            verbose=False,
        )

    @property
    def chunk_size(self) -> int:
        return int(self._policy.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self._policy.action_dim)

    @property
    def precision(self) -> str:
        return str(self._policy.precision)

    def predict(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> OmniRequestOutput:
        """Run one inference; return ``OmniRequestOutput`` with ``ActionArtifact``.

        ``primary_image`` / ``wrist_image``: HxWx3 uint8 RGB at 256x256.
        ``state``: 8-D float32 (3 eef_pos + 3 axisangle + 2 gripper) **or**
        a LIBERO obs dict (``robot0_eef_pos``, ``robot0_eef_quat``,
        ``robot0_gripper_qpos``), which the wrapper converts via
        ``state_from_libero_obs``.
        """
        # Lazy imports keep this module importable without numpy / outputs.
        import numpy as np

        from nanovllm_omni.outputs import ActionArtifact, MultimodalPayload, OmniRequestOutput

        env_actions = self._policy.predict_env_action_chunk(
            primary_image=primary_image,
            wrist_image=wrist_image,
            instruction=instruction,
            state_or_obs=state,
            execute_steps=execute_steps,
        )
        # env_actions: [execute_steps, 7] (or [1, 7] if no chunk)
        if env_actions.ndim == 1:
            env_actions = env_actions[None, :]
        array = np.asarray(env_actions, dtype=np.float32)

        artifact = ActionArtifact.from_array(array)
        payload = MultimodalPayload.from_dict({"action": artifact})
        return OmniRequestOutput(
            multimodal_output=payload,
            metrics={
                "chunk_size": float(artifact.chunk_size),
                "action_dim": float(artifact.action_dim),
            },
        )


__all__ = ["TurboVLAConfig", "TurboVLAOmni"]
