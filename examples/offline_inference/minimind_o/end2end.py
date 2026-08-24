"""Smoke: Omni.generate -> audio.wav (MiniMind-O).

Single-prompt offline inference against the aligned Python seam
(``nanovllm_omni.Omni``). Sampling defaults are read from
``deploy/minimind_omni.yaml`` via the PipelineRunner; caller-supplied
``--max-tokens`` overrides at request time.

Run with the local weights pre-provisioned by the project README::

    cd examples/offline_inference/minimind_o
    HF_HUB_OFFLINE=1 bash run_end2end.sh --model /home/mcig/minimind-3o \
        --mimi /home/mcig/mimi --out audio.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanovllm_omni import Omni, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
