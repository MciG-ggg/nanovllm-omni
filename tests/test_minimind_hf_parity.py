#!/usr/bin/env python3
"""Standalone MiniMind-O parity gate (docs/adr/0011-parity-standard.md).

NOT a pytest test — run directly so the fork's triton/flash_attn stubs from
``conftest.py`` don't shadow the real GPU stack:

  PYTHONPATH=$REPO:$REPO/third_party/nano-vllm \\
  HF_HUB_OFFLINE=1 python -u tests/test_minimind_hf_parity.py

``PYTHONPATH`` is load-bearing on the WSL box: the venv's editable installs
point at a second checkout, so without it you measure that checkout, not
this one.

Measures the two layers that can be compared, and nothing else:

  P1  greedy text tokens — the fork thinker vs the vendor ``forward``.
      Both sides are near-greedy (``temperature=0`` becomes ``1e-5`` and the
      distribution collapses to one-hot), so only near-ties can differ.
  P2  bridge rows — the tensor the talker consumes. Cosine + relative
      error against the vendor's ``bridge_states``.

Sampled audio codes are not compared. Both sides draw with
``torch.multinomial``, but the vendor interleaves text and audio draws in one
loop while this repo runs the talker as a separate stage, so the two consume
the RNG stream in a different order: equal codes are not expected even when
the logits match, and a mismatch is not evidence of a bug.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODEL_DIR = os.environ.get("MINIMIND_PARITY_MODEL_DIR", "/home/mcig/models/minimind-3o")
MIMI_DIR = os.environ.get("MINIMIND_PARITY_MIMI_DIR", "/home/mcig/mimi")

SEED = 42
MAX_NEW_TOKENS = 24  # ~2 s of audio at 12.5 Hz
PROMPT = "你好，请用一句话介绍你自己。"

# P2 tolerances. Measured on RTX 3050 / fp16: cosine 1.0000000,
# relative mean 0.05%. ``MAX_RELATIVE_MEAN`` is ~2x that measured value;
# a large multiple of it means real implementation drift, not fp16 rounding.
MIN_COSINE = 0.9999
MAX_RELATIVE_MEAN = 1e-3


def _vendor_text_and_bridge(torch, prompt_ids: list[int]):
    """Greedy text tokens + bridge rows from the unmodified vendor model."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.float16
    ).to("cuda")
    model.eval()

    # ``stream_generate`` cannot express greedy: it divides logits by
    # ``temperature`` and softmax overflows. Drive the same forward and
    # take argmax; ``rp=1.0`` means no repetition penalty to apply.
    generated: list[int] = []
    current = list(prompt_ids)
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            out = model.forward(torch.tensor([current], device="cuda"), use_cache=False)
            token = out.logits[0, -1, :].argmax().item()
            generated.append(token)
            current.append(token)
            if token == 2:
                break

    # Replicate the thinker half of ``MiniMindOmni.forward`` so the
    # bridge rows at ``config.bridge_layer`` can be read out.
    thinker = model.thinker
    seq_len = len(current)
    ids = torch.tensor([current], device="cuda")
    with torch.no_grad():
        hidden = thinker.dropout(thinker.embed_tokens(ids))
        position_embeddings = (
            thinker.freqs_cos[:seq_len],
            thinker.freqs_sin[:seq_len],
        )
        bridge = None
        for index, layer in enumerate(thinker.layers):
            hidden, _ = layer(hidden, position_embeddings)
            if index == model.config.bridge_layer:
                bridge = hidden
    bridge = bridge[0].float().clone()
    del model
    torch.cuda.empty_cache()
    return generated, bridge


def _ours_text_bridge_and_codes(torch, prompt: str):
    """Greedy text tokens, bridge rows, and audio codes from this repo."""
    from nanovllm_omni import Omni, SamplingParams
    from nanovllm_omni.models.minimind_omni.stage import TalkerStage, ThinkerStage

    torch.manual_seed(SEED)
    omni = Omni(MODEL_DIR, mimi_model_id=MIMI_DIR, device="cuda")

    thinker_outputs: list = []
    audio_codes: list = []
    original_decode = ThinkerStage._decode
    original_talker = TalkerStage.__call__

    def _spy_decode(self, payloads, samplings):
        outputs = original_decode(self, payloads, samplings)
        thinker_outputs.extend(outputs)
        return outputs

    def _spy_talker(self, payload, sampling):
        output = original_talker(self, payload, sampling)
        audio_codes.append(output.audio_codes.detach().cpu().clone())
        return output

    ThinkerStage._decode = _spy_decode
    TalkerStage.__call__ = _spy_talker
    try:
        omni.generate(
            [prompt],
            SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=MAX_NEW_TOKENS,
                extra={"repetition_penalty": 1.0},
            ),
        )
    finally:
        ThinkerStage._decode = original_decode
        TalkerStage.__call__ = original_talker

    first = thinker_outputs[0]
    generated: list[int] = []
    for token in first.output_token_ids:
        generated.append(token)
        if token == 2:
            break

    codes = audio_codes[0] if audio_codes else torch.zeros(0, 8, dtype=torch.long)
    return generated, first.bridge_states, codes


def _assert_under_repo(name: str, module_file: str) -> None:
    resolved = Path(module_file).resolve()
    assert REPO in resolved.parents, (
        f"{name} resolved to {resolved}, which is not under {REPO}. "
        "Set PYTHONPATH=$REPO:$REPO/third_party/nano-vllm so this checkout is measured."
    )
    print(f"{name}: {resolved}")


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    assert torch.cuda.is_available(), "CUDA required"
    assert os.path.isdir(MODEL_DIR), f"Model not found: {MODEL_DIR}"
    assert os.path.isdir(MIMI_DIR), f"Mimi not found: {MIMI_DIR}"

    import nanovllm_omni

    # The WSL venv's editable installs point at a second checkout of both
    # packages. Importing that one runs stale code and reports plausible
    # numbers, so fail closed instead of printing a hint nobody reads.
    _assert_under_repo("nanovllm_omni", nanovllm_omni.__file__)
    import nanovllm

    _assert_under_repo("nanovllm", nanovllm.__file__)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    print(f"prompt tokens: {len(prompt_ids)}")

    torch.manual_seed(SEED)
    vendor_text, vendor_bridge = _vendor_text_and_bridge(torch, prompt_ids)
    print(f"\nvendor greedy ({len(vendor_text)}): {vendor_text}")

    torch.manual_seed(SEED)
    ours_text, ours_bridge, ours_codes = _ours_text_bridge_and_codes(torch, formatted)
    print(f"ours   greedy ({len(ours_text)}): {ours_text}")

    # ---- P1: greedy text tokens ----
    text_equal = vendor_text == ours_text
    print(f"\n[P1] greedy text tokens equal: {text_equal}")

    # ---- P2: bridge rows ----
    # Our bridge covers prompt + generated + post-EOS filler, a superset of the
    # vendor's prompt + generated, and row ``t`` depends only on tokens ``0..t``
    # — so the vendor rows are a prefix of ours. A shorter ours would silently
    # shrink the comparison, so assert the direction before slicing.
    vendor_rows_count = vendor_bridge.shape[0]
    assert ours_bridge.shape[0] >= vendor_rows_count, (
        f"our bridge has {ours_bridge.shape[0]} rows but the vendor arm has "
        f"{vendor_rows_count}; the P2 comparison would silently shrink"
    )
    ours_rows = ours_bridge[:vendor_rows_count].float()
    vendor_rows = vendor_bridge.float()
    cosine = torch.nn.functional.cosine_similarity(
        ours_rows.flatten(), vendor_rows.flatten(), dim=0
    ).item()
    mean_diff = (ours_rows - vendor_rows).abs().mean().item()
    mean_scale = vendor_rows.abs().mean().clamp_min(1e-6).item()
    relative = mean_diff / mean_scale
    bridge_ok = cosine >= MIN_COSINE and relative <= MAX_RELATIVE_MEAN
    print(
        f"[P2] bridge rows={vendor_rows_count} (ours={ours_bridge.shape[0]}) "
        f"cosine={cosine:.8f} relative_mean={relative:.4%} "
        f"({'PASS' if bridge_ok else 'FAIL'})"
    )

    # ---- informational: the audio the talker actually drew ----
    print(f"\n[info] ours audio frames: {ours_codes.shape[0]} (codes not compared)")

    if text_equal and bridge_ok:
        print("\nPASS: parity (P1 text equal, P2 bridge within tolerance)")
        return
    print("\nFAIL: parity")
    sys.exit(1)


if __name__ == "__main__":
    main()
