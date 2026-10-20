"""Flow-step bench for the SmolVLA action expert.

ncu target for the flow-matching kernels. The two-stage bench mixes
the VLM prefix kernels with the flow kernels, and the ncu NVTX filter
does not match the push/pop ranges that the repository emits, so an
ncu capture lands on the VLM GEMMs instead of the flow kernels (see
ADR-0003, "smolvla" note).

This bench builds the prefix once and then runs the flow loop
``--repeats`` times, so a launch-count window of an ncu capture lands
inside the flow kernels.

Usage:

    python -m nanovllm_omni.bench.bench_smolvla_flow --repeats 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from .bench_smolvla import _load_pipeline, _sampling
from .smolvla_prompts import SMOLVLA_INPUTS


def flow_loop(
    action: Any,
    payload: Any,
    sampling: Any,
    num_steps: int,
) -> float:
    """Run the flow-matching loop once; return the wall time in ms.

    The action stage owns the loop: ``prepare`` samples the initial
    noise, ``denoise_step`` runs the action expert, and
    ``step_scheduler`` integrates the velocity.
    """
    import torch

    state = action.prepare(payload, sampling)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(num_steps):
        noise_pred = action.denoise_step(state, step=step, num_steps=num_steps)
        action.step_scheduler(state, noise_pred)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Flow-matching steps per repeat (the checkpoint default).",
    )
    args = parser.parse_args(argv)

    from nanovllm_omni.models.smolvla.stage_processors import vlm2action

    vlm, action = _load_pipeline(args.device, args.model)
    sm_input = SMOLVLA_INPUTS[0]

    sampling = _sampling(sm_input)
    vlm_out = vlm(sm_input.instruction, sampling)
    payload = vlm2action(vlm_out, sm_input.instruction)
    print("prefix done", flush=True)

    walls: list[float] = []
    for repeat in range(args.repeats):
        wall = flow_loop(action, payload, sampling, args.steps)
        walls.append(wall)
        print(
            f"repeat={repeat} flow={wall:.1f} ms per_step={wall / args.steps:.1f} ms",
            flush=True,
        )

    import torch

    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    median = statistics.median(walls)
    print(
        f"flow median={median:.1f} ms per_step={median / args.steps:.1f} ms vram={peak:.0f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
