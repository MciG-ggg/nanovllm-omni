#!/usr/bin/env python
"""Smoke: Omni.generate -> WAV, on the MLX track (Apple Silicon).

The three stages split across frameworks by measurement (ADR 0024): the
thinker and the talker run on MLX, code2wav stays on torch and runs on
MPS. The thinker and talker read converted directories written by
``scripts/convert_minimind_{thinker,talker}_mlx.py``.

Usage:
    python end2end_mlx.py --prompt "你好" --output out.wav
"""

from __future__ import annotations

import argparse
import os
import time

from nanovllm_omni import Omni
from nanovllm_omni.config.params import SamplingParams

_DEFAULT_MODEL = os.path.expanduser("~/models/minimind-3o")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind-O on MLX (Apple Silicon)")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="checkpoint directory")
    parser.add_argument("--output", default="minimind_mlx.wav")
    # The talker's emission gate opens at step 8, so a very small budget
    # emits zero frames and code2wav rejects the empty code matrix.
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = os.path.abspath(os.path.expanduser(args.model))
    for sub in ("mlx-thinker", "mlx-talker"):
        if not os.path.isdir(os.path.join(model, sub)):
            raise SystemExit(
                f"missing {sub} under {model}\n"
                f"convert it with: python scripts/convert_minimind_{sub.split('-')[-1]}_mlx.py"
            )

    omni = Omni(model, backend="mlx", extra={"allow_hf_download": False})
    started = time.perf_counter()
    out = omni.generate(
        [args.prompt],
        SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed),
    )[0]
    elapsed = time.perf_counter() - started
    if out.error:
        raise SystemExit(out.error)

    audio = (out.multimodal_output or {}).get("audio")
    if audio is None or not getattr(audio, "data", None):
        raise SystemExit("no audio: the talker emitted no frames (raise --max-tokens)")

    with open(args.output, "wb") as handle:
        handle.write(audio.data)

    print(f"=== prompt ===\n{args.prompt}")
    print(f"=== wav ===\n{args.output} ({len(audio.data)} bytes, {elapsed * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
