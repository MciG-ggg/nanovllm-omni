"""L1 demo: load local TurboVLA, run inference on synthetic observation.

Run after ``scripts/setup_turbovla.sh``:

    python examples/turbovla.py --ckpt pretrained/TurboVLA/checkpoints/libero/libero_object.pth \\
                               --dinov3  pretrained/dinov3-vitb \\
                               --bert    pretrained/bert-base-uncased

What this does:

1. Constructs ``TurboVLAOmni`` (loads TurboVLA + DINOv3 + BERT onto CUDA).
2. Builds a synthetic observation (two random 256x256 RGB images, 8-D
   zero state, arbitrary instruction string). Synthetic so no real
   LIBERO install is needed to verify the wrapper contract.
3. Calls ``model.predict(...)``; verifies the returned
   ``OmniRequestOutput`` has ``multimodal_output["action"]`` as an
   ``ActionArtifact`` with the expected shape.
4. Prints the head of the action chunk.

This is the L1 bar from TK-008/TK-009/TK-010/TK-011: the script runs,
produces a valid artifact, and exits 0. No LIBERO integration here;
that's ``examples/turbovla_libero.py`` (TK-011 Step 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from nanovllm_omni.models.turbovla import TurboVLAConfig, TurboVLAOmni
from nanovllm_omni.outputs import ActionArtifact, OmniRequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboVLA L1 demo (synthetic obs)")
    parser.add_argument("--ckpt", required=True, type=Path, help="path to libero_*.pth checkpoint")
    parser.add_argument("--dinov3", required=True, type=Path, help="local DINOv3-vitb directory")
    parser.add_argument(
        "--bert", required=True, type=Path, help="local BERT-base-uncased directory"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--instruction",
        default="pick up the red mug",
        help="language instruction (any string; L1 demo does not require a real LIBERO task)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[turbovla] loading model from {args.ckpt}")
    config = TurboVLAConfig(
        ckpt_path=str(args.ckpt),
        dinov3_path=str(args.dinov3),
        bert_path=str(args.bert),
        device=args.device,
        precision=args.precision,
    )
    model = TurboVLAOmni(config)
    print(
        f"[turbovla] loaded: chunk_size={model.chunk_size}, "
        f"action_dim={model.action_dim}, precision={model.precision}"
    )

    # Synthetic observation: two 256x256 RGB uint8 images + zero state.
    primary = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)

    print(f"[turbovla] running inference for instruction: {args.instruction!r}")
    output: OmniRequestOutput = model.predict(
        primary_image=primary,
        wrist_image=wrist,
        instruction=args.instruction,
        state=state,
        execute_steps=model.chunk_size,
    )

    if output.error:
        print(f"[turbovla] FAIL: {output.error}", file=sys.stderr)
        return 1

    if output.multimodal_output is None or "action" not in output.multimodal_output:
        print("[turbovla] FAIL: multimodal_output['action'] missing", file=sys.stderr)
        return 1

    action = output.multimodal_output["action"]
    if not isinstance(action, ActionArtifact):
        print(
            f"[turbovla] FAIL: expected ActionArtifact, got {type(action).__name__}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[turbovla] OK: action shape={action.array.shape}, dtype={action.dtype}, "
        f"action_dim={action.action_dim}, chunk_size={action.chunk_size}"
    )
    print("[turbovla] first action (env-space):")
    print(action.array[0])
    print("[turbovla] metrics:", output.metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
