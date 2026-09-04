#!/usr/bin/env python
"""SmolVLM-500M-Instruct offline demo (TK-011).

Single-stage VLM (image + text -> text). Heavy deps (transformers, torch)
stay inside the factory; only ``PIL`` is needed at the example layer to
open the image.

Usage:
    # First-time setup (weights cache, ~1 GB):
    hf download HuggingFaceTB/SmolVLM-500M-Instruct --local-dir pretrained/SmolVLM-500M-Instruct
    # Then:
    python run.py --image /path/to/cat.jpg --prompt "What is in this image?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

import nanovllm_omni
from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

# Deploy YAMLs ship inside the installed package at
# ``nanovllm_omni/deploy/``; resolve via the package's __file__ so the path
# works whether the script runs from a wheel install or a source clone.
_DEPLOY_YAML = Path(nanovllm_omni.__file__).resolve().parent / "deploy" / "smolvlm.yaml"
# Default to the HF repo id so ``HF_HUB_OFFLINE=1`` resolves the snapshot
# from ``~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/``.
_DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def _make_synthetic_image(path: Path) -> Path:
    """Write a 224x224 RGB gradient as a smoke image when the caller
    didn't pass ``--image``. Saves to ``path`` and returns it."""
    import math

    w = h = 224
    pixels = [
        (
            int(128 + 127 * math.sin(i / 16.0)),
            int(128 + 127 * math.sin(j / 16.0)),
            int(128 + 127 * math.cos((i + j) / 22.0)),
        )
        for j in range(h)
        for i in range(w)
    ]
    img = Image.new("RGB", (w, h))
    img.putdata(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLM-500M-Instruct demo")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--image",
        default=None,
        help="path to a PIL-readable image. Default: synthesise a 224x224 gradient.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            "local snapshot dir OR the HF repo id (downloads on the fly). "
            f"Default: {_DEFAULT_MODEL}"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-download", action="store_true", help="allow Hub download (default: local-only)"
    )
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = _make_synthetic_image(Path("/tmp/smolvlm_synthetic.png"))

    image = Image.open(image_path).convert("RGB")

    extra: dict[str, Any] = {
        "deploy_config_path": str(_DEPLOY_YAML),
        "allow_hf_download": bool(args.allow_download),
    }
    if Path(args.model).is_dir():
        extra["checkpoint_path"] = args.model
    omni = Omni(args.model, device=args.device, extra=extra)
    sp_extra: dict[str, Any] = {
        "images": [image],
        "max_new_tokens": args.max_new_tokens,
    }
    out = omni.generate(
        [args.prompt],
        SamplingParams(extra=sp_extra),
    )[0]
    text = out.multimodal_output["text"]
    print(f"=== prompt ===\n{args.prompt}\n=== output ===\n{text}\n(image={image_path})")


if __name__ == "__main__":
    main()
