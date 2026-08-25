#!/usr/bin/env python
"""Sana-0.6B offline demo (TK-009).

Runs one text prompt through the ``sana_06b`` pipeline and writes the image
to ``output.png``. The checkpoint is read local-only by default (matches the
download-locally-then-rsync workflow); pass ``--allow-download`` to fetch
from the Hub instead.

Usage:
    python run.py --model /path/to/sana_snapshot --prompt "a cyberpunk cat"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

# Repo root = parents[3] from examples/offline_inference/sana_06b/run.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY_YAML = _REPO_ROOT / "deploy" / "sana_06b.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sana-0.6B text-to-image demo")
    parser.add_argument("--prompt", default="a cyberpunk cat")
    parser.add_argument(
        "--model",
        default="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        help="diffusers repo id or a local snapshot dir",
    )
    parser.add_argument("--steps", type=int, default=20, help="num_inference_steps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="output.png")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow Hub download (default: local-only)"
    )
    args = parser.parse_args()

    extra = {"deploy_config_path": str(_DEPLOY_YAML)}
    if not args.allow_download:
        extra["allow_hf_download"] = False
    omni = Omni(args.model, device=args.device, extra=extra)
    out = omni.generate(
        [args.prompt],
        SamplingParams(extra={"num_inference_steps": args.steps}),
    )[0]
    image = out.multimodal_output["image"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"saved {args.output}: size={image.size} mode={image.mode} prompt={args.prompt!r}")


if __name__ == "__main__":
    main()
