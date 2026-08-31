#!/usr/bin/env python
"""Profile any omni family's ``Omni.generate`` under torch.profiler.

One recording surface for the library-backed families (smolvlm / sd_turbo /
smolvla) so wall-time / VRAM peak / top-kernel comparisons across families
are apples-to-apples. minimind_omni has its own stage-level ``bench`` CLI;
this script is the generic ``Omni.generate`` cross-section.

Usage::

    python tools/profile_family.py smolvlm --model <path|id> --out <prefix> [--max-new-tokens N]
    python tools/profile_family.py sd_turbo --model <path|id> --out <prefix> [--steps N]
    python tools/profile_family.py smolvla --model <path|id> --out <prefix>

Flags:
    --out <prefix>   write <prefix>.summary.md (always) and, with --trace,
                     <prefix>.trace.json (Kineto chrome trace).
    --trace          export the chrome trace (can be hundreds of MB).
    --prompt TEXT    override default prompt/instruction.
    --device DEV     torch device (default: cuda when available).
    --deploy PATH    deploy yaml; default <package>/deploy/<family>.yaml.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_deploy(family: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    import nanovllm_omni

    path = Path(nanovllm_omni.__file__).resolve().parent / "deploy" / f"{family}.yaml"
    return str(path) if path.exists() else None


def _build_omni(family: str, args: argparse.Namespace) -> Any:
    """Replicates each family's example payload construction."""
    from nanovllm_omni import Omni

    deploy = _resolve_deploy(family, args.deploy)
    extra: dict[str, Any] = {}
    if deploy:
        extra["deploy_config_path"] = deploy
    if family == "smolvla":
        extra.setdefault("pipeline", "smolvla")
    model = (
        args.model
        or {
            "smolvlm": str(_repo_root() / "pretrained" / "SmolVLM-500M-Instruct"),
            "sd_turbo": "stabilityai/sd-turbo",
            "smolvla": "HuggingFaceVLA/smolvla_libero",
        }[family]
    )
    return Omni(model, device=args.device, **extra)


def _build_sampling(family: str, args: argparse.Namespace) -> Any:
    from nanovllm_omni import SamplingParams

    extra: dict[str, Any] = {}
    if family == "smolvlm":
        from PIL import Image

        w = h = 224
        px = [
            (
                int(128 + 127 * math.sin(i / 16.0)),
                int(128 + 127 * math.sin(j / 16.0)),
                int(128 + 127 * math.cos((i + j) / 22.0)),
            )
            for j in range(h)
            for i in range(w)
        ]
        img = Image.new("RGB", (w, h))
        img.putdata(px)
        extra["images"] = [img]
        extra["max_new_tokens"] = args.max_new_tokens
    elif family == "sd_turbo":
        extra["num_inference_steps"] = args.steps
    elif family == "smolvla":
        rng = np.random.default_rng(0)
        extra["image"] = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        extra["wrist_image"] = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        extra["state"] = np.zeros(8, dtype=np.float32)
    return SamplingParams(extra=extra)


def _prompt_for(family: str) -> str:
    return {
        "smolvlm": "Describe this image in one sentence.",
        "sd_turbo": "a cute cat, studio photo",
        "smolvla": "pick up the block",
    }[family]


def _top_kernels(prof: Any, n: int = 10) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            rows.append((ev.key, ev.self_device_time_total / 1e3, ev.count))
    rows.sort(key=lambda r: -r[1])
    return rows[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=["smolvlm", "sd_turbo", "smolvla"])
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="profile")
    p.add_argument("--prompt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--deploy", default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--trace", action="store_true")
    p.add_argument(
        "--wall-runs",
        type=int,
        default=3,
        help="bare (un-profiled) generate iterations; median shown alongside the probe",
    )
    args = p.parse_args()

    omni = _build_omni(args.family, args)
    sampling = _build_sampling(args.family, args)
    prompt = [args.prompt or _prompt_for(args.family)]

    wall_median_ms = 0.0
    if args.wall_runs > 0:
        times: list[float] = []
        with torch.inference_mode():
            for _ in range(args.wall_runs + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                omni.generate(prompt, sampling)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
        wall_median_ms = statistics.median(times[1:])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof,
        record_function("omni.generate"),
    ):
        out = omni.generate(prompt, sampling)[0]
    wall_ms = (time.perf_counter() - t0) * 1e3
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1e6

    text = out.multimodal_output
    if isinstance(text, dict):
        text = {
            k: (str(v)[:60] if not isinstance(v, (list, tuple)) else f"<{len(v)} items>")
            for k, v in text.items()
        }
    else:
        text = str(text)[:120]

    lines = [
        f"# {args.family} Omni.generate profile",
        "",
        f"- model: {args.model or 'default'}",
        f"- device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}",
        f"- wall_ms (probe): **{wall_ms:.1f}**",
        f"- wall_median_ms (bare): **{wall_median_ms:.1f}**",
        f"- vram_peak_mb: **{vram_peak_mb:.0f}**",
        f"- output: {text}",
        "",
        "| kernel | total_ms | count |",
        "| --- | ---: | ---: |",
    ]
    for name, total_ms, count in _top_kernels(prof):
        lines.append(f"| {name} | {total_ms:.2f} | {count} |")

    out_path = Path(args.out)
    if args.trace:
        prof.export_chrome_trace(str(out_path) + ".trace.json")
    (out_path.parent).mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    (out_path.with_suffix(".summary.md")).write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
