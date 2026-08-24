"""SmolVLA policy in LIBERO, driven through the Omni seam.

Inference goes through ``Omni(...).generate(...)`` -- the same public
entry point as MiniMind-O. The SmolVLA pipeline stage
(``nanovllm_omni.models.smolvla.stage``) runs the exact official eval
chain when the caller passes a raw lerobot LIBERO obs dict:

    preprocess_observation -> env_preprocessor (flip + quat->axis-angle)
    -> policy.preprocessor -> policy.select_action (50-step chunk queue)
    -> policy.postprocessor

That chain is what delivers ~100% SR on libero_object task 5 with the
``HuggingFaceVLA/smolvla_libero`` checkpoint (matches lerobot/scripts/
lerobot_eval.py). This script only builds the env, collects raw obs,
and feeds each one through ``Omni.generate``; the stage owns all the
preprocessing that a from-scratch loop tends to get subtly wrong
(which is why hand-rolled obs led to 0% SR).

Requires a GPU host with ``lerobot`` (see ``[smolvla]`` extra), and
the ``pretrained/smolvla_libero`` + ``pretrained/smolvlm2-500m``
directories (see ``synthetic_obs.py`` docstring; WSL has no HF access,
so point the patched config at a local SmolVLM2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from nanovllm_omni import Omni, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA in LIBERO via Omni.generate")
    parser.add_argument("--model", default="pretrained/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--out-dir", default="episodes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark as _bench

    ts = _bench.get_benchmark_dict()[args.task_suite]()
    env = LiberoEnv(
        task_suite_name=args.task_suite,
        task_suite=ts,
        task_id=args.task_id,
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        obs_type="pixels_agent_pos",
        num_steps_wait=10,
        control_mode="relative",
        episode_length=args.max_steps,
    )

    task_description = args.instruction or env.task_description
    print(f"[smolvla-libero] task: {args.task_suite}.{args.task_id} -- {task_description!r}")
    print(
        f"[smolvla-libero] plan: {args.num_episodes} episodes x max {args.max_steps} steps (Omni seam)"
    )

    omni = Omni(args.model, device=args.device, dtype=args.dtype, pipeline="smolvla")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    successes: list[bool] = []

    try:
        for ep_idx in range(args.num_episodes):
            obs, _info = env.reset(seed=args.seed + ep_idx)
            frames: list[np.ndarray] = []
            episode_success = False
            step = 0

            while step < args.max_steps:
                if step % args.record_every == 0:
                    raw_img = obs["pixels"]["image"]
                    gp = raw_img[::-1, ::-1]
                    if gp.ndim == 4:
                        gp = gp[0]
                    frames.append(np.asarray(gp).copy())

                # Feed the RAW lerobot obs dict through the Omni seam; the
                # SmolVLA stage runs the official eval chain internally.
                out = omni.generate(
                    [task_description],
                    SamplingParams(
                        extra={"libero_obs": obs, "libero_task_suite": args.task_suite},
                    ),
                )[0]
                if out.error:
                    print(
                        f"[smolvla-libero] ep {ep_idx + 1}: inference failure: {out.error}",
                        file=sys.stderr,
                    )
                    break
                artifact = out.multimodal_output["actions"]
                a = np.asarray(artifact.array, dtype=np.float32)
                if a.ndim == 2:
                    a = a[0]

                obs, _r, term, trunc, info = env.step(a)
                step += 1
                if term or trunc or info.get("is_success"):
                    episode_success = bool(info.get("is_success", False))
                    break

            _write_video(frames, out_dir / f"ep{ep_idx + 1:02d}.mp4")
            successes.append(episode_success)
            tag = "OK" if episode_success else "miss"
            print(f"[smolvla-libero] ep {ep_idx + 1}/{args.num_episodes}: {tag}  steps={step}")
    finally:
        env.close()

    n = len(successes)
    sr = 100.0 * sum(successes) / n if n else 0.0
    print(f"\n[smolvla-libero] === Summary: {sum(successes)}/{n} successful ({sr:.1f}%) ===")
    return 0


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for f in frames:
            writer.append_data(f)


if __name__ == "__main__":
    sys.exit(main())
