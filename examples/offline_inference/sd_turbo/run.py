#!/usr/bin/env python
"""SD-Turbo offline demo (TK-009 rev2).

Runs one text prompt through the ``sd_turbo`` pipeline and writes the image
to ``output.png``. SD-Turbo is an adversarially-distilled SD 2.1 variant:
1 inference step, ``guidance_scale=0.0`` (locked). Recognizable output at
512x512 in well under a second on any GPU with >= 2.5 GB VRAM.

Usage:
    python run.py --model /path/to/sd_turbo_snapshot --prompt "a cute cat, studio photo"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

# Deploy YAMLs ship inside the installed package at
# ``nanovllm_omni/deploy/``; resolve via the package's __file__ so the path
# works whether the script runs from a wheel install or a source clone.
_DEPLOY_YAML = Path(nanovllm_omni.__file__).resolve().parent / "deploy" / "sd_turbo.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="SD-Turbo text-to-image demo")
    parser.add_argument("--prompt", default="a cute cat, studio photo")
    parser.add_argument(
        "--model",
        default="stabilityai/sd-turbo",
        help="diffusers repo id or a local snapshot dir",
    )
    parser.add_argument("--steps", type=int, default=1, help="num_inference_steps (1-4)")
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="guidance_scale (default 0.0; SD-Turbo is distilled, CFG >0 hurts)",
    )
    parser.add_argument("--height", type=int, default=None, help="image height (default 512)")
    parser.add_argument("--width", type=int, default=None, help="image width (default 512)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="output.png")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow Hub download (default: local-only)"
    )
    args = parser.parse_args()

    extra: dict[str, Any] = {
        "deploy_config_path": str(_DEPLOY_YAML),
        "allow_hf_download": bool(args.allow_download),
    }
    omni = Omni(args.model, device=args.device, extra=extra)
    sp_extra: dict[str, Any] = {"num_inference_steps": args.steps}
    if args.guidance is not None:
        sp_extra["guidance_scale"] = args.guidance
    if args.height is not None:
        sp_extra["height"] = args.height
    if args.width is not None:
        sp_extra["width"] = args.width
    out = omni.generate(
        [args.prompt],
        SamplingParams(extra=sp_extra),
    )[0]
    image = out.multimodal_output["image"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"saved {args.output}: size={image.size} mode={image.mode} prompt={args.prompt!r}")


if __name__ == "__main__":
    main()
