#!/usr/bin/env python3
"""Standalone parity test: HF vendor vs nanovllm-omni audio codes.

NOT a pytest test — run directly to bypass conftest triton/flash_attn stubs:

  cd /home/mcig/nanovllm-omni
  HF_HUB_OFFLINE=1 \
  /home/mcig/venvs/nanovllm-omni-torch212/bin/python -u \
    tests/test_minimind_hf_parity.py
"""

from __future__ import annotations

import os
import sys

MODEL_DIR = os.environ.get("MINIMIND_PARITY_MODEL_DIR", "/home/mcig/minimind-3o")
MIMI_DIR = os.environ.get("MINIMIND_PARITY_MIMI_DIR", "/home/mcig/mimi")

SEED = 42
MAX_NEW_TOKENS = 24  # ~2 s of audio at 12.5 Hz
PROMPT = "你好，请用一句话介绍你自己。"


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA required"
    assert os.path.isdir(MODEL_DIR), f"Model not found: {MODEL_DIR}"
    assert os.path.isdir(MIMI_DIR), f"Mimi not found: {MIMI_DIR}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    print(f"Prompt tokens: {len(input_ids)}")

    # ---- HF vendor path ----
    print("\n=== HF vendor path ===")
    torch.manual_seed(seed := SEED)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True).to("cuda").eval()
    )
    input_ids_t = torch.tensor([input_ids], device="cuda")

    hf_frames: list[list[int]] = []
    hf_text_ids: list[int] = []
    torch.manual_seed(seed)
    gen = model.stream_generate(
        input_ids_t,
        eos_token_id=tokenizer.eos_token_id or 2,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        top_p=0.90,
        rp=1.05,
        use_cache=False,
        return_audio_codes=True,
    )
    for text_chunk, audio_frame in gen:
        if text_chunk is not None:
            hf_text_ids.extend(text_chunk[0].tolist())
        if audio_frame is not None:
            hf_frames.append(audio_frame)
    del model
    torch.cuda.empty_cache()

    print(f"HF text tokens: {hf_text_ids}")
    print(f"HF audio frames: {len(hf_frames)}")
    if hf_frames:
        print(f"HF frame 0: {hf_frames[0]}")

    # ---- nanovllm-omni path ----
    print("\n=== nanovllm-omni path ===")
    from nanovllm_omni import Omni, SamplingParams
    from nanovllm_omni.models.minimind_omni._engine import TalkerStage

    torch.manual_seed(seed := SEED)
    omni = Omni(MODEL_DIR, mimi_model_id=MIMI_DIR, device="cuda")

    captured: list = []
    _orig = TalkerStage.__call__

    def _spy(self, payload, sampling):
        out = _orig(self, payload, sampling)
        captured.append(out.audio_codes.detach().cpu().clone())
        return out

    TalkerStage.__call__ = _spy
    try:
        sp = SamplingParams(temperature=0.7, max_tokens=MAX_NEW_TOKENS)
        outs = omni.generate([formatted], sp)
    finally:
        TalkerStage.__call__ = _orig

    nano_codes = captured[0] if captured else None
    nano_text = list(outs[0].token_ids) if hasattr(outs[0], "token_ids") else []

    print(f"Nano text tokens: {nano_text}")
    if nano_codes is not None:
        print(f"Nano audio codes shape: {nano_codes.shape}")
        if nano_codes.numel() > 0:
            print(f"Nano frame 0: {nano_codes[0].tolist()}")
    else:
        print("Nano: no audio codes captured!")

    # ---- Compare ----
    print("\n=== Comparison ===")
    if not hf_frames:
        print("FAIL: HF produced zero frames")
        sys.exit(1)
    if nano_codes is None or nano_codes.numel() == 0:
        print("FAIL: Omni produced zero audio codes")
        sys.exit(1)

    hf_n = len(hf_frames)
    nano_n = nano_codes.shape[0]
    overlap = min(hf_n, nano_n)
    print(f"HF frames={hf_n}, Omni frames={nano_n}, overlap={overlap}")
    assert overlap > 0, "No overlapping frames"

    total_pos = overlap * 8
    match_pos = 0
    worst = (0, 9, [])
    for i in range(overlap):
        diffs = [j for j in range(8) if hf_frames[i][j] != nano_codes[i, j].item()]
        m = 8 - len(diffs)
        match_pos += m
        if m < worst[1]:
            worst = (i, m, diffs)

    ratio = match_pos / total_pos
    print(f"Matching positions: {match_pos}/{total_pos} = {ratio:.2%}")
    print(f"Worst frame {worst[0]}: {worst[1]}/8 codebooks, diffs={worst[2]}")

    for i in range(min(10, overlap)):
        hf_s = " ".join(f"{c:4d}" for c in hf_frames[i])
        nm_s = " ".join(f"{nano_codes[i, j].item():4d}" for j in range(8))
        diffs = [j for j in range(8) if hf_frames[i][j] != nano_codes[i, j].item()]
        mark = " OK" if not diffs else f" DIFF@{diffs}"
        print(f"  frame {i:2d} HF: [{hf_s}]")
        print(f"  frame {i:2d} NM: [{nm_s}]{mark}")

    if ratio >= 7 / 8:
        print(f"\nPASS: parity ratio {ratio:.2%} >= 87.5%")
    else:
        print(f"\nFAIL: parity ratio {ratio:.2%} < 87.5%")


if __name__ == "__main__":
    main()
