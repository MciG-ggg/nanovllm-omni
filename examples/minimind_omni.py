"""Smoke: Omni.generate → audio.wav for MiniMind-O."""

import argparse
from pathlib import Path

from nanovllm_omni import Omni, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/mcig/minimind-3o")
    parser.add_argument("--mimi", default="/home/mcig/mimi")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="audio.wav")
    args = parser.parse_args()

    omni = Omni(args.model, device=args.device, mimi_model_id=args.mimi)
    outs = omni.generate(
        [args.prompt],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.7, top_p=0.9),
    )
    audio = outs[0].multimodal_output["audio"]
    data = audio.wav_bytes()
    Path(args.out).write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes, sr={audio.sample_rate})")


if __name__ == "__main__":
    main()
