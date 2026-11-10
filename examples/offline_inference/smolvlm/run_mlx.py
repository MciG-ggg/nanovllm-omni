#!/usr/bin/env python
"""Smoke: Omni.generate -> text, on the MLX-VLM backend.

The torch track rides the fork's triton attention kernel, which has no
macOS wheel, so on an Apple Silicon host this is the only way to run
SmolVLM. It reads an mlx-vlm checkpoint directory, not the transformers
one: ``mlx-vlm convert --hf-path <hf-id> -o <dir>`` writes it.

Usage:
    python run_mlx.py --model ~/models/smolvlm-500m-instruct --image cat.jpg
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image

from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

_DEFAULT_MODEL = os.path.expanduser("~/models/smolvlm-500m-instruct")


def _synthetic_image(path: Path) -> Path:
    """A 512x512 gradient, so the demo needs no external asset."""
    width = height = 512
    pixels = [
        (
            int(128 + 127 * math.sin(i / 32.0)),
            int(128 + 127 * math.sin(j / 32.0)),
            int(128 + 127 * math.cos((i + j) / 44.0)),
        )
        for j in range(height)
        for i in range(width)
    ]
    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLM on MLX (Apple Silicon)")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--image", default=None, help="default: synthesise a 512x512 gradient")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"mlx-vlm checkpoint directory. Default: {_DEFAULT_MODEL}",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        raise SystemExit(
            f"no mlx-vlm checkpoint at {args.model}\n"
            f"convert one with: mlx-vlm convert --hf-path "
            f"HuggingFaceTB/SmolVLM-500M-Instruct -o {args.model}"
        )

    image_path = Path(args.image) if args.image else _synthetic_image(Path("smolvlm_mlx.png"))
    image = Image.open(image_path).convert("RGB")

    extra: dict[str, Any] = {"allow_hf_download": False}
    omni = Omni(args.model, backend="mlx-vlm", device="cpu", extra=extra)
    out = omni.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            extra={"images": [image], "max_new_tokens": args.max_new_tokens},
        ),
    )[0]
    if out.error:
        raise SystemExit(out.error)

    print(f"=== prompt ===\n{args.prompt}")
    print(f"=== output ===\n{out.multimodal_output['text']}")
    print(f"(image={image_path}, backend=mlx-vlm)")


if __name__ == "__main__":
    main()
