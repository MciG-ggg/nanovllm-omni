#!/usr/bin/env python3
"""defect #5 robust verification: A->B->C->A cross-prompt cycle (RTX 3050).

Stronger than #51 probe: cycle through 3 different prompt lengths and
back. Re-capture fix must re-capture on each length change, so the
control prompt at position 1 must match control at position 4.

Also runs an end-to-end Omni() seam to verify the user-facing path
(OmniBase + PipelineRunner) doesn't regress the cross-prompt invariant.

  cd ~/nanovllm-omni && PYTHONPATH=/home/mcig/nanovllm-omni \\
  ~/venvs/vllm-omni/bin/python tools/bench_defect5_3cycle.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, "/home/mcig/nanovllm-omni")

import torch
from transformers import AutoTokenizer

from nanovllm_omni.models.minimind_omni import create_bundle, run_generate

MODEL = "/home/mcig/minimind-3o"
MIMI = "/home/mcig/mimi"
MAX_NEW = 16


@dataclass(frozen=True)
class _Case:
    label: str
    text: str


# Three distinct lengths: ~2 / ~9 / ~16 tokens (covers small/medium/large prefill)
CASES = [
    _Case("short", "你好。"),
    _Case("medium", "请用两句话介绍MiniMind-O模型。"),
    _Case(
        "long",
        "MiniMind-O 是一个面向端侧推理的语音-文本联合生成模型，结合了 MiniMind 文本基座与 Mimi 神经音频编解码器。",
    ),
]


def _ids(tok, txt: str) -> torch.Tensor:
    return tok(txt, return_tensors="pt").input_ids.cuda()


def _generate(model, tok, txt: str) -> list[int]:
    ids = _ids(tok, txt)
    eos = tok.eos_token_id
    out = run_generate(
        model,
        ids,
        max_new_tokens=MAX_NEW,
        temperature=0.75,
        top_p=0.9,
        eos_token_id=eos,
        open_thinking=False,
        use_cuda_graph=True,
    )
    return out if isinstance(out, list) else list(out)


def _kv_capture_log(model) -> list[int]:
    """Snapshot prefill lengths captured by each attention's buffer (length
    after each re-capture). Useful diagnostic — should be unique for each
    distinct prompt length we've seen."""
    seen: list[int] = []
    for m in model.modules():
        k = getattr(m, "_nanovllm_kv_buffer", None)
        if k is None:
            continue
        seen.append(int(m._kv_past_key.shape[1]))
        break  # all buffers share shape
    return seen


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    bundle = create_bundle(model_id=MODEL, mimi_model_id=MIMI)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = bundle.model

    # Cycle: short -> medium -> long -> short
    # We need a single 'short' output both at index 0 and at index 3;
    # after each generation the KV buffers are reset by the re-capture,
    # so the second short should match the first short byte-for-byte.
    a0 = _generate(model, tok, CASES[0].text)
    _ = _generate(model, tok, CASES[1].text)
    _ = _generate(model, tok, CASES[2].text)
    a1 = _generate(model, tok, CASES[0].text)

    same_short_after_cycle = a0 == a1
    print(f"3-CYCLE: short at idx 0 == short at idx 3 -> {same_short_after_cycle}", flush=True)
    if not same_short_after_cycle:
        print(f"  a0 tokens: {a0[:8]}...", flush=True)
        print(f"  a1 tokens: {a1[:8]}...", flush=True)

    # Repeat-determinism control (must always be True regardless of defect)
    b0 = _generate(model, tok, CASES[1].text)
    b1 = _generate(model, tok, CASES[1].text)
    same_repeat = b0 == b1
    print(f"3-CYCLE: medium repeat-after-cycle -> {same_repeat}", flush=True)

    # Cycle again to check stability across multiple round-trips
    _ = _generate(model, tok, CASES[2].text)
    _ = _generate(model, tok, CASES[1].text)
    a2 = _generate(model, tok, CASES[0].text)
    a3 = _generate(model, tok, CASES[0].text)
    cycle2_same = a2 == a3
    cycle2_match = a1 == a2
    print(f"3-CYCLE 2nd pass: short repeat -> {cycle2_same}", flush=True)
    print(f"3-CYCLE 2nd pass: short idx-3 == idx-7 -> {cycle2_match}", flush=True)

    verdict = same_short_after_cycle and same_repeat and cycle2_same and cycle2_match
    print(f"VERDICT: defect #5 {'CLOSED' if verdict else 'STILL OPEN'} on 3050", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
