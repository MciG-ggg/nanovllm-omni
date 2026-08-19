"""Smoke: Omni.generate → audio.wav for MiniMind-O.

Reads sampling defaults from deploy/minimind_omni.yaml via the
PipelineRunner; the caller-supplied SamplingParams overrides at request
time. No temperature / max_tokens are hardcoded in this example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanovllm_omni import Omni, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--out", default="audio.wav")
    args = parser.parse_args()

    extra: dict[str, str] = {}
    if args.mimi:
        extra["mimi_model_id"] = args.mimi
    if args.deploy_config:
        extra["deploy_config_path"] = args.deploy_config

    omni = Omni(args.model, device=args.device, **extra)

    kwargs: dict[str, object] = {}
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    sampling = SamplingParams(**kwargs) if kwargs else None

    outs = omni.generate([args.prompt], sampling)
    audio = outs[0].multimodal_output["audio"]
    data = audio.wav_bytes()
    Path(args.out).write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes, sr={audio.sample_rate})")


if __name__ == "__main__":
    main()
