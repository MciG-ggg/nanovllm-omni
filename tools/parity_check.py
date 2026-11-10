#!/usr/bin/env python3
"""Parity: this repo's MLX backend vs mlx-lm's own loop, greedy, ADR 0021's three tiers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from nanovllm_omni.bench.prompts import BENCH_PROMPTS

DEFAULT_MODEL = os.path.expanduser("~/models/minimind-3o/mlx-thinker")

EXACT = "EXACT"
TOP_K_MATCH = "TOP_K_MATCH"
FAIL = "FAIL"
_TIER_RANK = {EXACT: 0, TOP_K_MATCH: 1, FAIL: 2}

ARMS = ("mlx-lm", "repo")


@dataclass(frozen=True)
class Arm:
    """One arm's greedy output: token ids, plus per-step top-k candidates when it has logprobs."""

    token_ids: list[int]
    top_k: list[list[tuple[int, float]]] | None = None


@dataclass(frozen=True)
class Verdict:
    """ADR 0021's tier for one prompt, plus what the first divergence looked like."""

    tier: str
    index: int | None = None
    reference_token: int | None = None
    candidate_token: int | None = None
    reference_top: list[tuple[int, float]] | None = None
    candidate_top: list[tuple[int, float]] | None = None
    reason: str = ""


def _first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    # strict=False: arms of different lengths have no divergence in the shared prefix.
    pairs = zip(reference, candidate, strict=False)
    return next((i for i, (a, b) in enumerate(pairs) if a != b), None)


def parity_report(reference: Arm, candidate: Arm, max_tokens: int) -> Verdict:
    """The tier for one prompt; ``max_tokens`` carries ADR 0021's length clause."""
    index = _first_divergence(reference.token_ids, candidate.token_ids)
    if index is None:
        if len(reference.token_ids) == max_tokens == len(candidate.token_ids):
            return Verdict(EXACT)
        return Verdict(
            FAIL,
            reason=(
                f"tokens agree but lengths {len(reference.token_ids)}/"
                f"{len(candidate.token_ids)} are not both max_tokens={max_tokens}"
            ),
        )

    reference_token = reference.token_ids[index] if index < len(reference.token_ids) else None
    candidate_token = candidate.token_ids[index] if index < len(candidate.token_ids) else None
    divergence = {
        "index": index,
        "reference_token": reference_token,
        "candidate_token": candidate_token,
    }
    if reference_token is None or candidate_token is None:
        return Verdict(FAIL, reason="one arm stopped before the other", **divergence)
    if reference.top_k is None or candidate.top_k is None:
        return Verdict(
            FAIL, reason="no logprobs from one arm, so top-k cannot be checked", **divergence
        )
    if index >= len(reference.top_k) or index >= len(candidate.top_k):
        return Verdict(FAIL, reason="no logprobs for the divergence step", **divergence)

    reference_top = reference.top_k[index]
    candidate_top = candidate.top_k[index]
    # Bidirectional on purpose: vLLM's check_logprobs_close asks each side's
    # token to be plausible under the other side's distribution.
    candidate_is_close = any(token == candidate_token for token, _ in reference_top)
    reference_is_close = any(token == reference_token for token, _ in candidate_top)
    tier = TOP_K_MATCH if candidate_is_close and reference_is_close else FAIL
    return Verdict(
        tier,
        reference_top=reference_top,
        candidate_top=candidate_top,
        **divergence,
    )


def _logprobs(logits: Any) -> Any:
    """One [vocab] row of log-softmax, in fp32 so fp16 rounding cannot decide the top-k order."""
    import mlx.core as mx

    row = logits[0].astype(mx.float32)
    return row - mx.logsumexp(row)


def _top_k(logprobs: Any, k: int) -> list[tuple[int, float]]:
    """The ``k`` likeliest (token, logprob) pairs, likeliest first."""
    import mlx.core as mx

    row = logprobs.astype(mx.float32)
    return [(token, float(row[token])) for token in mx.argsort(-row)[:k].tolist()]


def run_reference(model: Any, prompt_token_ids: list[int], max_tokens: int, top_k: int) -> Arm:
    """mlx-lm's own greedy loop; ``generate_step`` yields (token, logprobs) per step."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import greedy_sampler

    token_ids: list[int] = []
    per_step: list[list[tuple[int, float]]] = []
    for token, logprobs in generate_step(
        mx.array(prompt_token_ids), model, max_tokens=max_tokens, sampler=greedy_sampler
    ):
        token_ids.append(int(token))
        per_step.append(_top_k(logprobs, top_k))
    return Arm(token_ids, per_step)


def _fork_sequence(prompt_token_ids: list[int], max_tokens: int) -> Any:
    """The fork ``Sequence`` that ``prepare_*`` and the sampler read."""
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

    # The fork rejects temperature <= 1e-10, so greedy cannot be expressed as
    # temperature=0 here; ``run_candidate`` takes the argmax instead.
    return Sequence(
        prompt_token_ids,
        ForkSamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True),
    )


def run_candidate(
    runner: Any,
    prompt_token_ids: list[int],
    max_tokens: int,
    top_k: int,
    split_prefill: bool = False,
) -> Arm:
    """This repo's MlxStageRunner in a greedy AR loop, logprobs taken from its logits.

    ``split_prefill`` prefills all but the last prompt token and then steps that
    one, which is the chunking ``mlx_lm.generate.generate_step`` uses; it exists
    to attribute a divergence to prefill shape instead of to the decode math.
    """
    import mlx.core as mx

    sequence = _fork_sequence(prompt_token_ids, max_tokens)
    runner.set_context(is_prefill=True)
    if split_prefill:
        # Keep the full prompt on the sequence so positions and the sampler see
        # what a one-shot prefill would have given them; only the forward is split.
        sequence.token_ids = list(prompt_token_ids[:-1])
        ids, positions = runner.prepare_prefill([sequence])
        runner.forward(ids, positions, True)
        sequence.token_ids = list(prompt_token_ids)
        sequence.last_token = prompt_token_ids[-1]
        runner.set_context(is_prefill=False)
        ids, positions = runner.prepare_decode([sequence])
        logits = runner.forward(ids, positions, False)
    else:
        ids, positions = runner.prepare_prefill([sequence])
        logits = runner.forward(ids, positions, True)

    token_ids: list[int] = []
    per_step: list[list[tuple[int, float]]] = []
    while len(token_ids) < max_tokens:
        logprobs = _logprobs(logits)
        per_step.append(_top_k(logprobs, top_k))
        token = int(mx.argmax(logprobs).item())
        token_ids.append(token)
        if len(token_ids) == max_tokens:
            break
        sequence.append_token(token)
        runner.set_context(is_prefill=False)
        ids, positions = runner.prepare_decode([sequence])
        logits = runner.forward(ids, positions, False)
    runner.reset_context()
    return Arm(token_ids, per_step)


def prompt_token_ids(tokenizer: Any, prompt: Any) -> list[int]:
    """One bench prompt as chat-template ids, system prompt first when it has one."""
    messages = [{"role": "user", "content": prompt.text}]
    if prompt.system:
        messages.insert(0, {"role": "system", "content": prompt.system})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, open_thinking=False)


def _format_top(top: list[tuple[int, float]] | None) -> str:
    return "none" if top is None else " ".join(f"{token}:{score:.4f}" for token, score in top)


def _print_prompt(prompt: Any, reference: Arm, candidate: Arm, verdict: Verdict) -> None:
    print(f"prompt {prompt.id} ({len(reference.token_ids)} vs {len(candidate.token_ids)} tokens)")
    print(f"  reference {reference.token_ids}")
    print(f"  candidate {candidate.token_ids}")
    if verdict.reason:
        print(f"  why: {verdict.reason}")
    if verdict.index is None:
        print(f"  verdict: {verdict.tier}")
        return
    print(f"  verdict: {verdict.tier} at token {verdict.index}")
    print(
        f"    reference picked {verdict.reference_token}, top-k {_format_top(verdict.reference_top)}"
    )
    print(
        f"    candidate picked {verdict.candidate_token}, top-k {_format_top(verdict.candidate_top)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "PYTHONPATH=$PWD:$PWD/third_party/nano-vllm uv run --no-project "
            "--with mlx --with mlx-lm --python 3.12 python tools/parity_check.py"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompts", default=None, help="comma-separated bench prompt ids (default: all)"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--reference", choices=ARMS, default="mlx-lm")
    parser.add_argument("--candidate", choices=ARMS, default="repo")
    parser.add_argument(
        "--split-prefill",
        action="store_true",
        help="candidate arm only: prefill prompt[:-1] then step the last token, as mlx-lm does",
    )
    parser.add_argument("--output", default=None, help="write the report as JSON here")
    args = parser.parse_args()
    if args.split_prefill and args.candidate != "repo":
        parser.error("--split-prefill only applies to the repo candidate arm")

    import mlx.core as mx
    from mlx_lm.utils import load as mlx_load

    from nanovllm_omni.backend.mlx_backend import MlxBackend

    model_path = os.path.expanduser(args.model)
    backend = MlxBackend(dtype=args.dtype)
    runner = backend.make_stage_runner(object, backend.make_config(model_path))
    # A second, independent load: the reference arm must not share state with
    # the arm under test, and both must run the same weights and dtype.
    reference_model, _ = mlx_load(model_path)
    reference_model.set_dtype(getattr(mx, args.dtype))
    mx.eval(reference_model.parameters())

    prompts = list(BENCH_PROMPTS)
    if args.prompts:
        wanted = [name.strip() for name in args.prompts.split(",")]
        prompts = [next(p for p in BENCH_PROMPTS if p.id == name) for name in wanted]

    by_name = {"mlx-lm": reference_model, "repo": runner}
    print(
        f"model: {model_path} | dtype {args.dtype} | reference {args.reference} "
        f"| candidate {args.candidate} | greedy"
    )
    print(f"max_tokens={args.max_tokens} top_k={args.top_k} split_prefill={args.split_prefill}")

    results = []
    for prompt in prompts:
        ids = prompt_token_ids(runner.tokenizer, prompt)
        arms = {}
        for name in (args.reference, args.candidate):
            if name == "repo":
                arms[name] = run_candidate(
                    runner, ids, args.max_tokens, args.top_k, args.split_prefill
                )
            else:
                arms[name] = run_reference(by_name[name], ids, args.max_tokens, args.top_k)
        reference, candidate = arms[args.reference], arms[args.candidate]
        verdict = parity_report(reference, candidate, args.max_tokens)
        _print_prompt(prompt, reference, candidate, verdict)
        results.append(
            {
                "id": prompt.id,
                "prompt_tokens": len(ids),
                "verdict": verdict.tier,
                "divergence": verdict.index,
                "reason": verdict.reason,
                "reference_token_ids": reference.token_ids,
                "candidate_token_ids": candidate.token_ids,
                "reference_top": verdict.reference_top,
                "candidate_top": verdict.candidate_top,
            }
        )

    worst = max(results, key=lambda row: _TIER_RANK[row["verdict"]])
    divergence = next((row for row in results if row["divergence"] is not None), None)
    where = (
        "" if divergence is None else f" divergence {divergence['id']}@{divergence['divergence']}"
    )
    print(f"conclusion: {worst['verdict']}{where} ({len(results)} prompts)")

    if args.output:
        report = {
            "model": model_path,
            "dtype": args.dtype,
            "reference": args.reference,
            "candidate": args.candidate,
            "sampling": "greedy",
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "split_prefill": args.split_prefill,
            "conclusion": worst["verdict"],
            "prompts": results,
        }
        with open(os.path.expanduser(args.output), "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"report: {os.path.expanduser(args.output)}")


if __name__ == "__main__":
    main()
