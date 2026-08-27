"""Smoke: Omni.generate audio-in -> ASR transcript + audio.wav (MiniMind-O).

Half-duplex voice turn: read a user wav, feed it to the thinker via the
engine-native audio-input path (<|audio_pad|> prefill injection, Q4), ASR it
in parallel (double-track, Q2) and emit both the transcript and the model's
spoken reply as audio.wav.

Run with the local weights pre-provisioned by the project README::

    cd examples/offline_inference/minimind_o
    HF_HUB_OFFLINE=1 bash run_audio_in.sh --model /home/mcig/minimind-3o \
        --mimi /home/mcig/mimi \
        --audio-encoder /home/mcig/SenseVoiceSmall \
        --audio /path/to/user.wav --out reply.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanovllm_omni import Omni, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pretrained/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument(
        "--audio-encoder",
        default="pretrained/SenseVoiceSmall",
        help="SenseVoice checkpoint dir (funasr); required when --audio is set",
    )
    parser.add_argument("--audio", required=True, help="user speech wav / numpy path")
    parser.add_argument(
        "--prompt", default="", help="optional text; audio markers carry the speech"
    )
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
    extra["audio_encoder_path"] = args.audio_encoder

    omni = Omni(args.model, device=args.device, **extra)

    kwargs: dict[str, object] = {}
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    sampling = SamplingParams(**kwargs) if kwargs else None

    prompt = {"prompt": args.prompt, "audio": args.audio}
    out = omni.generate([prompt], sampling)[0]

    transcript = out.custom_output.get("transcript", "")
    if transcript:
        print(f"ASR transcript: {transcript}")

    audio = out.multimodal_output["audio"]
    data = audio.wav_bytes()
    Path(args.out).write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes, sr={audio.sample_rate})")


if __name__ == "__main__":
    main()
