"""L1 demo: Omni.generate → ActionArtifact for SmolVLA (synthetic obs).

Same seam as ``examples/offline_inference/minimind_o/end2end.py``:
construct Omni, call generate. Images / state go in
``SamplingParams.extra``; the prompt is the instruction.

    pip install -e '.[smolvla]'
    # Mac (proxy) then rsync to WSL — do not run inference on the Mac:
    HTTPS_PROXY=http://127.0.0.1:29659 hf download HuggingFaceVLA/smolvla_libero \
        --local-dir pretrained/smolvla_libero
    ssh mcigs-wsl '.venv/bin/python examples/smolvla.py \
        --model pretrained/smolvla_libero --dtype int8 --device cuda'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from nanovllm_omni import Omni, SamplingParams
from nanovllm_omni.outputs import ActionArtifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA L1 demo via Omni.generate")
    parser.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument(
        "--dtype",
        default="int8",
        help="int8 (3050 4GB) / fp16 / None. CUDA int8 needs bitsandbytes; else fp16.",
    )
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="allow Hub fetches (WSL cannot reach huggingface.co; keep off there)",
    )
    parser.add_argument("--instruction", default="pick up the red mug")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra: dict[str, object] = {}
    if args.deploy_config:
        extra["deploy_config_path"] = args.deploy_config
    if args.allow_hf_download:
        extra["allow_hf_download"] = True
    if Path(args.model).is_dir():
        extra.setdefault("pipeline", "smolvla")

    omni = Omni(args.model, device=args.device, dtype=args.dtype, **extra)

    rng = np.random.default_rng(args.seed)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    sampling = SamplingParams(
        extra={"image": image, "wrist_image": wrist, "state": state},
    )
    out = omni.generate([args.instruction], sampling)[0]
    if out.error:
        print(f"[smolvla] FAIL: {out.error}", file=sys.stderr)
        return 1
    payload = out.multimodal_output
    if payload is None or "actions" not in payload:
        print("[smolvla] FAIL: multimodal_output['actions'] missing", file=sys.stderr)
        return 1
    action = payload["actions"]
    if not isinstance(action, ActionArtifact):
        print(
            f"[smolvla] FAIL: expected ActionArtifact, got {type(action).__name__}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[smolvla] OK: action shape={action.array.shape}, dtype={action.dtype}, "
        f"action_dim={action.action_dim}, chunk_size={action.chunk_size}"
    )
    print("[smolvla] first action:")
    print(action.array[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
