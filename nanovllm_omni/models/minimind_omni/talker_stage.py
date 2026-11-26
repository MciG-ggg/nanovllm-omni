"""The MiniMind talker stage: audio codes out of the thinker's bridge rows.

Split out of ``stage.py`` so both files stay under the size gate. The
thinker stage lives in ``stage.py``; the talker stage lives here, with the
family's talker contract in ``talker_recipe`` and the per-device runners in
``talker_torch`` / ``talker_mlx_runner``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nanovllm_omni.engine.stage_engine import StageEngine

from .stage import _resolve_mlx_model_path, _stage_backend


@dataclass
class TalkerOutput:
    """Talker output: per-frame audio codes consumed by ``talker2code2wav``."""

    audio_codes: Any  # torch.Tensor [frames, 8]


class TalkerStage(StageEngine):
    """Stage 1 — MiniMind talker decode; dense rewrite by default, vendor arm selectable."""

    def __init__(self, deploy: Any, args: Any) -> None:
        super().__init__()
        self.deploy = deploy
        self.args = args
        self._runner: Any = None
        # ADR 0020: the checkpoint stays as weights + parity reference. The
        # vendor class is the P1/P2 reference arm, not the default.
        extra = getattr(args, "extra", None) or {}
        self._impl = str(extra.get("talker_impl", "rewritten"))
        if self._impl not in ("rewritten", "vendor"):
            raise ValueError(
                f"unknown talker_impl {self._impl!r}; expected 'rewritten' or 'vendor'"
            )
        # ADR 0017 windowed KV stays opt-in until its parity gate is green on
        # CUDA (ADR 0022); the dense path is the one with a measured baseline.
        self._kv = str(extra.get("talker_kv", "dense"))
        if self._kv not in ("dense", "windowed"):
            raise ValueError(f"unknown talker_kv {self._kv!r}; expected 'dense' or 'windowed'")
        if self._kv == "windowed" and self._impl != "rewritten":
            raise ValueError("talker_kv='windowed' requires talker_impl='rewritten'")

    def _ensure_runner(self) -> Any:
        """Build the talker runner for the selected track."""
        if self._runner is not None:
            return self._runner

        from .bundle import _resolve_snapshot

        extra = getattr(self.args, "extra", None) or {}
        backend_name = _stage_backend(self.args, extra)
        dtype = getattr(self.args, "dtype", None) or "float16"
        if backend_name == "mlx":
            from .talker_mlx_runner import MlxTalkerRunner

            # The MLX runner loads the converted directory and runs on the
            # unified-memory GPU, so the torch device does not apply.
            model_path = _resolve_mlx_model_path(self.args, "mlx-talker")
            self._runner = MlxTalkerRunner(
                model_path,
                device="gpu",
                dtype=dtype,
                impl=self._impl,
                kv=self._kv,
            )
        elif backend_name == "torch":
            from ...backend.device import default_torch_device
            from .talker_torch import TorchTalkerRunner

            device = getattr(self.args, "device", None) or default_torch_device()
            model_path = _resolve_snapshot(getattr(self.args, "model", None))
            self._runner = TorchTalkerRunner(
                model_path,
                device=device,
                dtype=dtype,
                impl=self._impl,
                kv=self._kv,
            )
        else:
            raise ValueError(f"no MiniMind talker runner for backend {backend_name!r}")
        return self._runner

    def __call__(self, payload: Any, sampling: Any) -> Any:
        """Run the Talker: AR decode audio codes from bridge hidden states.

        Mirrors ``MiniMindOmni.stream_generate`` (vendor model_omni.py):
        at each step we run one forward (full sequence), sample one
        code per codebook from the 8 returned logit tensors using the
        vendor's staggered schedule (``audio_codes[i]`` lags codebook
        ``i`` by one position), and stop once codebook 7 has emitted
        ``audio_stop_token``.

        Args:
            payload: TalkerInputPayload (from thinker2talker processor).
            sampling: SamplingParams with temperature, max_tokens.

        Returns:
            TalkerOutput with audio_codes [frames, 8] tensor.
        """
        import torch

        from .stage_processors import TalkerInputPayload
        from .talker_recipe import (
            AUDIO_REPETITION_PENALTY,
            AUDIO_VENDOR_TEMPERATURE,
            NUM_CODEBOOKS,
            TalkerRequest,
        )

        if not isinstance(payload, TalkerInputPayload):
            raise TypeError(
                f"TalkerStage expected TalkerInputPayload, got {type(payload).__name__}"
            )
        bridge = payload.bridge_states
        text_codes = list(payload.text_token_ids)
        if bridge is None or (isinstance(bridge, torch.Tensor) and bridge.numel() == 0):
            raise ValueError("TalkerStage received empty bridge hidden states")
        if not text_codes:
            raise ValueError("TalkerStage received empty text_token_ids")

        extra = getattr(sampling, "extra", None) or {}
        seed = getattr(sampling, "seed", None)
        # Audio temperature is taken from the vendor default (0.2); the
        # text ``sampling.temperature`` is for the thinker stage.
        temperature = float(extra.get("audio_temperature", AUDIO_VENDOR_TEMPERATURE))
        if temperature <= 1e-10:
            temperature = 1e-5
        # ``watchdog_limit`` caps the number of talker decode steps the
        # way upstream's ``talker_max_steps_after_last_thinker_token``
        # does; unset means "replay every generated thinker row".
        watchdog_limit = int(extra["watchdog_limit"]) if extra.get("watchdog_limit") else None

        # ``start_pos``/``num_steps`` arrive precomputed from thinker2talker
        # (prompt length / generated-row count); no second split here.
        meta = dict(payload.metadata) if isinstance(payload.metadata, Mapping) else {}
        bridge_len = bridge.shape[-2]
        start_pos = int(meta.get("start_pos", len(payload.prompt_token_ids)))
        num_steps = int(meta.get("num_steps", bridge_len - start_pos))
        if watchdog_limit is not None:
            num_steps = min(num_steps, watchdog_limit)

        runner = self._ensure_runner()
        frames = runner.generate(
            TalkerRequest(
                bridge=bridge,
                num_steps=num_steps,
                start_pos=start_pos,
                temperature=temperature,
                seed=seed,
                speaker_embedding=payload.speaker_embedding,
                repetition_penalty=AUDIO_REPETITION_PENALTY,
            )
        )

        # ``frames`` mirrors the vendor: one row per emitted frame, each row
        # is the diagonal ``audio_codes[i][step - 7 + i]``. Truncation is
        # not needed because only full 8-code rows are ever appended.
        if not frames:
            audio_codes_tensor = torch.zeros(0, NUM_CODEBOOKS, dtype=torch.long)
        else:
            audio_codes_tensor = torch.tensor(frames, dtype=torch.long)
        return TalkerOutput(audio_codes=audio_codes_tensor.cpu())


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory — the talker selected by ``extra["talker_impl"]``."""
    return TalkerStage(deploy, args)


__all__ = [
    "TalkerOutput",
    "TalkerStage",
    "_talker_stage",
]
