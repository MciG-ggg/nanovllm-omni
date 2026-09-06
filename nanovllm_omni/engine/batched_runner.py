"""Minimal batched (continuous-batching) MiniMind-O thinker entry.

Single-scheduler drive loop — no StagePool / Orchestrator / LoadBalancer /
multi-replica: those measured zero contribution on the single-GPU scope
(docs/perf/model-family-bottlenecks-2026-08-27.md appendix, nr=1 == nr=2).
One ``RuntimeScheduler`` + one ``BatchedThinkerRunner`` already delivers the
measured ~+87% throughput for batch=2 vs serial (group forwarding of equal
KV-length requests); the ceremony above them was deleted.

Loop mirrors the reference's ``schedule() -> execute(prefill+decode) ->
update_from_output``; finished thinkers drop OUT to the serial
talker/mimi->wav chain (Q9a) via ``code2wav.decode_audio``.
"""

from __future__ import annotations

from itertools import count
from typing import Any

from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler
from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner
from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav
from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate
from nanovllm_omni.outputs import AudioPayload

# Globally-unique request id (leading ``seq-`` keeps ids distinct from
# replica-internal naming conventions).
_rids = count()


def run_batched_generate(
    bundle: Any,
    prompts: list[str],
    *,
    temperature: float = 0.75,
    top_p: float = 0.90,
    rp: float = 1.0,
    max_new_tokens: int = 1024,
    open_thinking: bool = False,
    max_batch: int | None = None,
    base_seed: int = 42,
    deploy: Any = None,
    kv_max_sequence_len: int | None = None,
) -> list[Any]:
    """Continuous-batching entry: tokenize -> drain one scheduler -> serial WAV.

    Submits every prompt to the single ``RuntimeScheduler`` +
    ``BatchedThinkerRunner``, then drives ``schedule -> prefill/decode ->
    update_from_output`` until all requests finish, finalizing each finished
    rid through the serial codec chain (mimi decode -> WAV). Returns one
    :class:`AudioPayload` per prompt in submission order.

    ``max_batch`` defaults to ``deploy.max_batch`` (or 2); the scheduler
    groups equal-KV-length requests so each batched forward is a single
    rectangular ``[B, 9, T]`` call.
    """
    if max_batch is None:
        max_batch = getattr(deploy, "max_batch", 2) if deploy is not None else 2

    sched = RuntimeScheduler(max_num_seqs=max_batch)
    runner = BatchedThinkerRunner(
        bundle,
        sched,
        temperature=temperature,
        top_p=top_p,
        rp=rp,
        max_new_tokens=max_new_tokens,
        open_thinking=open_thinking,
        base_seed=base_seed,
        kv_max_sequence_len=kv_max_sequence_len,
    )

    # Submit everything up-front; the scheduler admits <= max_batch as it
    # drains. Submission order is preserved by the deterministic drain.
    rids: list[str] = []
    for prompt in prompts:
        ids = tokenize_for_generate(bundle.tokenizer, prompt, open_thinking)
        rid = runner.add_request(ids[0].tolist(), request_id=f"seq-{next(_rids)}")
        rids.append(rid)

    results: dict[str, Any] = {}
    while sched.has_work():
        out = sched.schedule()
        if out.is_empty:
            break
        prefilled: set[str] = set()
        finished: set[str] = set()
        for group in out.prefill_groups:
            runner.prefill_group(group)
            prefilled.update(chunk.sequence.request_id for chunk in group.items)
        for group in out.decode_groups:
            runner.decode_group(group)
            for sequence in group.items:
                if runner.step_finished(sequence.request_id):
                    finished.add(sequence.request_id)
        sched.update_from_output(prefilled=prefilled, finished=finished)
        for rid in finished:
            st = runner.states[rid]
            if not st.frames:
                results[rid] = AudioPayload(data=b"", sample_rate=24_000)
            else:
                samples = decode_audio(bundle.mimi, st.frames, bundle.device)
                wav = encode_wav(samples, sample_rate=24_000)
                results[rid] = AudioPayload(data=wav, sample_rate=24_000)

    return [results[rid] for rid in rids]


__all__ = ["run_batched_generate"]
