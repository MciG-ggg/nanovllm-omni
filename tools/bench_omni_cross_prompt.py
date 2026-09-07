#!/usr/bin/env python3
"""End-to-end Omni() API cross-prompt verification on RTX 3050.

Tests the user-facing Python seam (not just the model-level run_generate):

  Omni(model_path, mimi_model_path).generate([prompt1, prompt2, ...])
  -> list[OmniRequestOutput] with valid audio bytes

Verifies:
  A. Each prompt produces a valid OmniRequestOutput with decodable WAV
  B. The deploy.use_thinker_cuda_graph=True (yaml default) is honored through
     Omni.generate (covered by test_deploy_default_cuda_graph.py contract
     test for the propagation; here we test the actual decode path runs)
  C. Cross-prompt: same prompt produces IDENTICAL WAV MD5 before and after
     a different prompt is interleaved. This is the §47 invariant
     translated to the user-facing API.

Run on WSL: cd ~/nanovllm-omni && PYTHONPATH=.  \\
  ~/venvs/vllm-omni/bin/python tools/bench_omni_cross_prompt.py
"""

from __future__ import annotations

import hashlib
import sys

sys.path.insert(0, "/home/mcig/nanovllm-omni")

from nanovllm_omni import Omni, SamplingParams

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
CONTROL = "你好。"
INTERLEAVE = "请用两句话介绍MiniMind-O模型。"


def _wav_md5(out) -> str:
    """MD5 of the WAV bytes in the OmniRequestOutput. The bytes include the
    RIFF header, so this is the full audio artifact."""
    payload = out.multimodal_output
    if payload is None or "audio" not in payload:
        return "<empty>"
    audio = payload["audio"]
    # AudioPayload has wav_bytes(); legacy raw-bytes payloads convert directly.
    if hasattr(audio, "wav_bytes"):
        return hashlib.md5(audio.wav_bytes()).hexdigest()
    return hashlib.md5(bytes(audio)).hexdigest()


def main() -> int:
    print("Loading Omni() with deploy.use_thinker_cuda_graph=True (yaml default)...", flush=True)
    omni = Omni(MODEL, mimi=MIMI)
    sp = SamplingParams(temperature=0.75, top_p=0.9, max_tokens=16)

    # A. single prompt produces valid output
    a0 = omni.generate([CONTROL], sampling_params=sp)
    if not a0 or _wav_md5(a0[0]) == "<empty>":
        print(f"FAIL: empty output for control: {a0}")
        return 1
    md5_a0 = _wav_md5(a0[0])
    print(f"call 0 (control baseline) md5={md5_a0}", flush=True)

    # B. cross-prompt cycle
    _ = omni.generate([INTERLEAVE], sampling_params=sp)
    a1 = omni.generate([CONTROL], sampling_params=sp)
    md5_a1 = _wav_md5(a1[0])
    print(f"call 1 (control after interleaved) md5={md5_a1}", flush=True)

    same_after_cycle = md5_a0 == md5_a1

    # C. repeat-determinism control
    b0 = omni.generate([CONTROL], sampling_params=sp)
    md5_b0 = _wav_md5(b0[0])
    print(f"call 2 (control repeat) md5={md5_b0}", flush=True)

    same_repeat = md5_a0 == md5_b0

    print(
        f"VERDICT: cross-prompt {'CLOSED' if same_after_cycle else 'OPEN'} "
        f"| repeat-deterministic {same_repeat}",
        flush=True,
    )
    return 0 if (same_after_cycle and same_repeat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
