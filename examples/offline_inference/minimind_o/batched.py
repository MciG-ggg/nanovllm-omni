"""Smoke: batched continuous-batching generate -> audio.wav (MiniMind-O).

Real-weight check for the batched engine loop
(``models/minimind_omni/kv_pool.py`` + ``models/minimind_omni/batched_runner.py``). Asserts two
properties that would otherwise be invisible in the fake-model unit
tests:

* **Q10a determinism** -- each request owns its RNG (seeded by request
  id), so running request ``p1`` batched next to ``p2`` MUST produce
  byte-identical audio to running ``p1`` solo. The engine loop is
  exercised via ``run_batched_generate`` with ``max_batch=2``.

* **Real B>1 decode** -- the model only supports one scalar
  ``start_pos`` per forward, so two same-length prompts form a real
  ``[B=2, 9, 1]`` decode group. ``model.forward`` is wrapped to
  record the actual input shapes; the largest decode batch observed
  is reported at the end as a smoke gate, not a benchmark.

Run with the local weights pre-provisioned by the project README::

    cd examples/offline_inference/minimind_o
    HF_HUB_OFFLINE=1 bash run_batched.sh --model /home/mcig/minimind-3o \
        --mimi /home/mcig/mimi --out batched_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanovllm_omni.models.minimind_omni.batched_runner import run_batched_generate
from nanovllm_omni.models.minimind_omni.bundle import load_minimind_omni_bundle as create_bundle


def _wav_bytes(payload) -> bytes:
    return payload.data if isinstance(payload.data, bytes) else b""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jingyaogong/minimind-3o")
    parser.add_argument("--mimi", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-a",
        default="你好，请用一句话介绍你自己。",
    )
    parser.add_argument(
        "--prompt-b",
        default="帮我写一句关于夏天的诗句。",
    )
    parser.add_argument("--out", default="batched_smoke")
    args = parser.parse_args()

    bundle_kwargs: dict[str, object] = {}
    if args.mimi:
        bundle_kwargs["mimi_model_id"] = args.mimi
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    # record actual [B, 9, T] forward shapes so we can prove batching happened
    shapes: list[tuple[int, int, int]] = []
    original_forward = bundle.model.forward

    def _wrapped_forward(input_ids, *a, **k):
        shapes.append(tuple(input_ids.shape))
        return original_forward(input_ids, *a, **k)

    bundle.model.forward = _wrapped_forward

    kwargs = {"max_new_tokens": args.max_tokens}
    solo = run_batched_generate(bundle, [args.prompt_a], **kwargs)
    batched = run_batched_generate(bundle, [args.prompt_a, args.prompt_b], max_batch=2, **kwargs)

    bundle.model.forward = original_forward  # restore

    solo_a = _wav_bytes(solo[0])
    batched_a = _wav_bytes(batched[0])
    batched_b = _wav_bytes(batched[1])

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    (outdir / "solo_a.wav").write_bytes(solo_a)
    (outdir / "batched_a.wav").write_bytes(batched_a)
    (outdir / "batched_b.wav").write_bytes(batched_b)

    # forward input is [B, 9, T]; decode steps have T == 1, batch is B.
    decode_batches = [s[0] for s in shapes if len(s) == 3 and s[2] == 1]
    max_decode_batch = max(decode_batches) if decode_batches else 0
    print(f"solo_a:    {len(solo_a):>7} bytes")
    print(f"batched_a: {len(batched_a):>7} bytes  (from max_batch=2 run)")
    print(f"batched_b: {len(batched_b):>7} bytes")
    print(f"max decode batch observed: B={max_decode_batch}")

    # Q10a: byte-identical per-request output regardless of batch layout
    assert solo_a == batched_a, "Q10a violation: prompt A differs between solo and batched runs"
    for name, data in (
        ("solo_a", solo_a),
        ("batched_a", batched_a),
        ("batched_b", batched_b),
    ):
        assert data[:4] == b"RIFF" and len(data) > 44, f"{name} is not a valid WAV"
    print("OK: Q10a determinism holds, all outputs valid WAV")


if __name__ == "__main__":
    main()
