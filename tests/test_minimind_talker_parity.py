"""Talker parity gates (ADR 0011 three layers, ADR 0022 acceptance).

P0 runs anywhere torch supports (CUDA or MPS): the same payload and seed twice
must produce bitwise-equal ``audio_codes`` and the same WAV sha256. Its bridge
is synthetic because the thinker's fork attention needs CUDA; P0 checks the
talker + code2wav determinism, not parity.

P1 and P2 are CUDA gates and skip elsewhere with the WSL command in the reason.
P1 runs the full pipeline once per talker implementation and compares the
sampled code sequence. P2 captures the vendor runner's teacher-forced inputs,
then compares both ``TorchTalkerRunner._logits`` paths at active codebooks.

transformers / Mimi load inside helpers: only ``-m smoke`` runs reach them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams  # noqa: E402
from nanovllm_omni.models.minimind_omni.code2wav import (  # noqa: E402
    MiniMindOmniCode2Wav,
    load_mimi_codec,
)
from nanovllm_omni.models.minimind_omni.stage import TalkerStage  # noqa: E402
from nanovllm_omni.models.minimind_omni.stage_processors import (  # noqa: E402
    Code2WavInputPayload,
    TalkerInputPayload,
    ThinkerStageOutput,
    thinker2talker,
)

MODEL = os.environ.get("MINIMIND_PARITY_MODEL_DIR", os.path.expanduser("~/models/minimind-3o"))
MIMI = os.environ.get("MINIMIND_PARITY_MIMI_DIR", os.path.expanduser("~/models/mimi"))
PROMPT = "你好，请用一句话介绍你自己。"
SEED = 42
NUM_STEPS = 32
MAX_NEW_TOKENS = 24

# ADR 0011 §4.2 measured max 0.0625 / mean 0.0047 on RTX 3050 fp16 for the
# teacher-forced talker. This gate feeds both arms identical tensors; the
# doubled limits leave room for fp16 implementation drift while catching a
# wrong embedding, norm, or RoPE path (ADR 0022).
MAX_ABS_DIFF = 0.125
MAX_MEAN_ABS_DIFF = 0.01
TOP_K = 5
EXACT = "EXACT"
TOP_K_MATCH = "TOP_K_MATCH"
FAIL = "FAIL"

_WSL_HINT = (
    "CUDA-only parity gate. Run it on the WSL box where the RTX 3050 baseline "
    "lives (ADR 0014: do not compare across machines):\n"
    "  ssh mcigs-wsl && cd ~/Projects/nanovllm-omni\n"
    "  export PYTHONPATH=$PWD:$PWD/third_party/nano-vllm KMP_DUPLICATE_LIB_OK=TRUE\n"
    "  uv run python -m pytest tests/test_minimind_talker_parity.py -m smoke -v"
)

_CHILD = f"""
import json
import sys

import torch

from nanovllm_omni import Omni, SamplingParams
from nanovllm_omni.backend.torch_talker import TorchTalkerRunner
from nanovllm_omni.models.minimind_omni.stage import TalkerStage

codes = []
topk_candidates = []
original = TalkerStage.__call__
original_logits = TorchTalkerRunner._logits


def _spy(self, payload, sampling):
    output = original(self, payload, sampling)
    codes.append(output.audio_codes.tolist())
    return output


def _capture_logits(self, bridge, audio_codes, spk_emb, cache):
    logits = original_logits(self, bridge, audio_codes, spk_emb, cache)
    topk_candidates.append(
        [logits_i[0, -1, :].topk({TOP_K}).indices for logits_i in logits]
    )
    return logits


TalkerStage.__call__ = _spy
TorchTalkerRunner._logits = _capture_logits
omni = Omni(
    {MODEL!r},
    mimi_model_id={MIMI!r},
    device="cuda",
    dtype="float16",
    extra={{"talker_impl": sys.argv[1], "allow_hf_download": False}},
)
with torch.no_grad():
    omni.generate(
        [{PROMPT!r}],
        SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens={MAX_NEW_TOKENS},
            seed={SEED},
            extra={{"repetition_penalty": 1.0}},
        ),
    )
print(json.dumps({{"codes": codes[0], "topk": [
    [indices.cpu().tolist() for indices in step]
    for step in topk_candidates
]}}))
"""


def _local_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _args(device: str, impl: str = "rewritten") -> OmniEngineArgs:
    return OmniEngineArgs(
        model=MODEL,
        device=device,
        extra={"talker_impl": impl},
    )


def _prompt_ids() -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def _payload(num_steps: int = NUM_STEPS) -> TalkerInputPayload:
    """Fixed synthetic bridge + the real chat-template prompt.

    The bridge is ``[prompt + num_steps, 768]`` fp32 like the thinker's; its
    values are seeded so every process sees the same tensor.
    """
    prompt_ids = _prompt_ids()
    generator = torch.Generator().manual_seed(11)
    bridge = torch.randn(len(prompt_ids) + num_steps, 768, generator=generator) * 0.1
    output_ids = [201] * num_steps  # post-EOS filler clock token
    return thinker2talker(
        ThinkerStageOutput(
            bridge_states=bridge,
            prompt_token_ids=prompt_ids,
            output_token_ids=output_ids,
            text_token_ids=prompt_ids + output_ids,
            metadata={"content_len": num_steps, "talker_tail": 0},
        )
    )


def _run_pipeline(
    impl: str,
) -> tuple[list[list[int]], list[list[list[int]]]]:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, impl],
        capture_output=True,
        text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1"),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    captured = json.loads(result.stdout.strip().splitlines()[-1])
    return captured["codes"], captured["topk"]


def audio_code_parity_verdict(
    vendor_codes: list[list[int]],
    rewritten_codes: list[list[int]],
    vendor_topk: list[list[list[int]]],
    rewritten_topk: list[list[list[int]]],
) -> str:
    if vendor_codes == rewritten_codes:
        return EXACT

    # Length mismatch is a parity verdict (FAIL), not an error: the two arms
    # may stop at different frames, and that is exactly what this reports.
    for frame_index, (vendor_frame, rewritten_frame) in enumerate(
        zip(vendor_codes, rewritten_codes, strict=False)
    ):
        for codebook_index, (vendor_code, rewritten_code) in enumerate(
            zip(vendor_frame, rewritten_frame, strict=False)
        ):
            if vendor_code == rewritten_code:
                continue
            # Emitted frame f uses codebook i at sampled position f + i + 1.
            step = frame_index + codebook_index + 1
            vendor_candidates = vendor_topk[step][codebook_index]
            rewritten_candidates = rewritten_topk[step][codebook_index]
            if vendor_code in rewritten_candidates and rewritten_code in vendor_candidates:
                return TOP_K_MATCH
            return FAIL
    return FAIL


def test_audio_code_parity_verdict_uses_top_k_candidates() -> None:
    vendor_codes = [[11, 20, 12, 13, 14, 15, 16, 17]]
    rewritten_codes = [[11, 21, 12, 13, 14, 15, 16, 17]]
    matching_vendor_logits = torch.full((3, 8, 32), -100.0)
    matching_rewritten_logits = torch.full((3, 8, 32), -100.0)
    matching_vendor_logits[2, 1, 20] = 10.0
    matching_vendor_logits[2, 1, 21] = 9.0
    matching_rewritten_logits[2, 1, 21] = 10.0
    matching_rewritten_logits[2, 1, 20] = 9.0
    failing_vendor_logits = torch.full((3, 8, 32), -100.0)
    failing_rewritten_logits = torch.full((3, 8, 32), -100.0)
    failing_vendor_logits[2, 1, 20] = 10.0
    failing_vendor_logits[2, 1, 21] = 9.0
    failing_rewritten_logits[2, 1, 21] = 10.0
    failing_rewritten_logits[2, 1, 0:4] = torch.tensor([9.0, 8.0, 7.0, 6.0])

    matching_vendor_topk = matching_vendor_logits.topk(TOP_K, dim=-1).indices.tolist()
    matching_rewritten_topk = matching_rewritten_logits.topk(TOP_K, dim=-1).indices.tolist()
    failing_vendor_topk = failing_vendor_logits.topk(TOP_K, dim=-1).indices.tolist()
    failing_rewritten_topk = failing_rewritten_logits.topk(TOP_K, dim=-1).indices.tolist()
    exact_verdict = audio_code_parity_verdict(
        vendor_codes, vendor_codes, matching_vendor_topk, matching_vendor_topk
    )
    topk_verdict = audio_code_parity_verdict(
        vendor_codes, rewritten_codes, matching_vendor_topk, matching_rewritten_topk
    )
    fail_verdict = audio_code_parity_verdict(
        vendor_codes, rewritten_codes, failing_vendor_topk, failing_rewritten_topk
    )

    assert (exact_verdict, topk_verdict, fail_verdict) == (EXACT, TOP_K_MATCH, FAIL)


@pytest.mark.smoke
def test_talker_and_code2wav_are_deterministic() -> None:
    device = _local_device()
    if device is None:
        pytest.skip("P0 needs CUDA or MPS")
    if not os.path.isdir(MODEL):
        pytest.skip(f"MiniMind weights not found: {MODEL}")
    if not os.path.isdir(MIMI):
        pytest.skip(f"Mimi weights not found: {MIMI}")

    payload = _payload()
    stage = TalkerStage(SimpleNamespace(), _args(device))
    sampling = SamplingParams(seed=SEED)

    first = stage(payload, sampling).audio_codes
    second = stage(payload, sampling).audio_codes

    assert first.shape[0] > 0, "the gate needs at least one emitted frame"
    assert torch.equal(first, second)

    codec = MiniMindOmniCode2Wav(load_mimi_codec(MIMI, device=device), device)
    first_wav = codec(Code2WavInputPayload(first)).data
    second_wav = codec(Code2WavInputPayload(second)).data

    assert hashlib.sha256(first_wav).digest() == hashlib.sha256(second_wav).digest()


@pytest.mark.smoke
def test_rewritten_audio_codes_match_vendor() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_WSL_HINT)
    if not os.path.isdir(MODEL) or not os.path.isdir(MIMI):
        pytest.skip(f"MiniMind/Mimi weights not found: {MODEL}, {MIMI}")

    vendor_codes, vendor_topk = _run_pipeline("vendor")
    rewritten_codes, rewritten_topk = _run_pipeline("rewritten")

    assert vendor_codes, "the vendor arm emitted no frames; the gate would pass vacuously"
    assert rewritten_codes, "the rewritten arm emitted no frames; the gate would pass vacuously"
    verdict = audio_code_parity_verdict(
        vendor_codes,
        rewritten_codes,
        vendor_topk,
        rewritten_topk,
    )
    assert verdict in (EXACT, TOP_K_MATCH), f"P1 parity verdict: {verdict}"


@pytest.mark.smoke
def test_rewritten_audio_logits_match_vendor() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_WSL_HINT)
    if not os.path.isdir(MODEL):
        pytest.skip(f"MiniMind weights not found: {MODEL}")

    from nanovllm_omni.backend.base import TalkerRequest
    from nanovllm_omni.backend.talker_recipe import NUM_CODEBOOKS
    from nanovllm_omni.backend.torch_talker import TorchTalkerRunner

    vendor = TorchTalkerRunner(MODEL, device="cuda", dtype="float16", impl="vendor")
    rewritten = TorchTalkerRunner(MODEL, device="cuda", dtype="float16", impl="rewritten")
    payload = _payload()
    request = TalkerRequest(
        bridge=payload.bridge_states,
        num_steps=payload.metadata["num_steps"],
        start_pos=payload.metadata["start_pos"],
        seed=SEED,
        repetition_penalty=1.0,
        speaker_embedding=payload.speaker_embedding,
    )

    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]] = []
    original = vendor._logits

    def _capture(bridge, codes, spk_emb, cache):
        logits = original(bridge, codes, spk_emb, cache)
        calls.append(
            (
                bridge.detach().cpu(),
                codes.detach().cpu(),
                None if spk_emb is None else spk_emb.detach().cpu(),
                [row[0, -1, :].detach().float().cpu() for row in logits],
            )
        )
        return logits

    vendor._logits = _capture
    try:
        vendor.generate(request)
    finally:
        vendor._logits = original

    diffs = []
    for step, (bridge, codes, spk_emb, vendor_logits) in enumerate(calls):
        ours = rewritten._logits(
            bridge.to("cuda"),
            codes.to("cuda"),
            None if spk_emb is None else spk_emb.to("cuda"),
            None,
        )
        for index in range(min(step, NUM_CODEBOOKS)):
            diffs.append((ours[index][0, -1, :].float().cpu() - vendor_logits[index]).abs())

    assert diffs, "no active codebook position was compared"
    stacked = torch.stack(diffs)
    assert float(stacked.max()) <= MAX_ABS_DIFF
    assert float(stacked.mean()) <= MAX_MEAN_ABS_DIFF
