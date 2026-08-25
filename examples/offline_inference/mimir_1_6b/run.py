#!/usr/bin/env python
"""Mimir-1.6B-Instruct offline demo (TK-021).

Three-stage SONAR embedding-space diffusion: SaT + SONAR encoder ->
Mimir LCM (40 steps) -> SONAR decoder. Output is plain text (terminal
``final_output_type="text"``).

Layout follows the repo convention ``pretrained/<family_name>``: pre-
provision weights once, then ``--model pretrained/Mimir-1.6B-Instruct``.

Usage:
    # First-time setup (one-off, brings in sonar-space, wtpsplit,
    # fairseq2 from Meta's index, and clones the upstream `lcm` repo):
    ./scripts/setup_mimir.sh
    # Then pre-provision weights (one-off, ~10 GB total):
    mkdir -p pretrained
    hf download mimir-lcm/Mimir-1.6B-Instruct       --local-dir pretrained/Mimir-1.6B-Instruct
    curl -fL -o pretrained/text_sonar_basic_encoder/sonar_text_encoder.pt  https://dl.fbaipublicfiles.com/SONAR/sonar_text_encoder.pt
    curl -fL -o pretrained/text_sonar_basic_decoder/sonar_text_decoder.pt  https://dl.fbaipublicfiles.com/SONAR/sonar_text_decoder.pt
    curl -fL -o pretrained/sentencepiece.source.256000.model               https://dl.fbaipublicfiles.com/SONAR/sentencepiece.source.256000.model
    # Optional: SaT for multilingual / threshold-based sentence split:
    hf download segment-any-text/sat-3l --include model.onnx config.json --local-dir pretrained/sat-3l
    hf download FacebookAI/xlm-roberta-base --include tokenizer.json config.json tokenizer_config.json sentencepiece.bpe.model --local-dir pretrained/xlm-roberta-base
    # Then:
    python run.py --model pretrained/Mimir-1.6B-Instruct --prompt "hi"

Note: if SaT is not cached, ``prompt_encoder`` silently falls back to a
regex sentence split (English-bias, fine for short prompts).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

# Repo root = parents[3] from examples/offline_inference/mimir_1_6b/run.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY_YAML = _REPO_ROOT / "deploy" / "mimir_1_6b.yaml"
_DEFAULT_MODEL = str(_REPO_ROOT / "pretrained" / "Mimir-1.6B-Instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mimir-1.6B-Instruct demo")
    parser.add_argument(
        "--prompt", default="User turn.\n\nDefine the word 'house'.\n\nAssistant turn."
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            "local snapshot dir OR the HF repo id (downloads model.pt on the fly). "
            f"Default: {_DEFAULT_MODEL}"
        ),
    )
    parser.add_argument("--steps", type=int, default=40, help="LCM inference_timesteps")
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="output.txt")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow Hub download (default: local-only)"
    )
    args = parser.parse_args()

    extra: dict[str, Any] = {
        "deploy_config_path": str(_DEPLOY_YAML),
        "allow_hf_download": bool(args.allow_download),
    }
    # If the caller passed a local snapshot dir, point the LCM at it.
    if Path(args.model).is_dir():
        extra["checkpoint_path"] = args.model
    omni = Omni(args.model, device=args.device, extra=extra)
    sp_extra: dict[str, Any] = {
        "inference_timesteps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
    }
    out = omni.generate(
        [args.prompt],
        SamplingParams(extra=sp_extra),
    )[0]
    text = out.multimodal_output["text"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f"saved {args.output}: {len(text)} chars prompt={args.prompt!r}")


if __name__ == "__main__":
    main()
