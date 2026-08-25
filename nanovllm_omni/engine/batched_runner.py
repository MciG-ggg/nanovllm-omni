"""Batched engine loop: scheduler -> batched thinker forward -> serial WAV.

Engine-level counterpart to ``BatchedThinkerRunner`` (which stays in
``models/minimind_omni/batched_generation.py`` as the model-level batched
forward). This module owns the loop shape that mirrors vllm-omni's
``schedule() -> execute(groups) -> update_from_output()`` (Q8a/Q9a) and the
serial hand-off of finished thinkers into the codec chain.

Layering (locked in the TICKET design session):
- ``engine/runtime_scheduler.py`` -> per-stage scheduler + Sequence (TK-004)
- ``engine/sched.py``        -> fixed-slot KV pool (pure lifecycle)
- ``engine/batched_runner.py`` -> engine loop, tokenization, serial codec chain
- ``models/minimind_omni/batched_generation.py`` -> per-request runner + forward
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler


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
    kv_max_seq: int | None = None,
    num_replicas: int = 1,
    balancer: Any | None = None,
) -> list[Any]:
    """Continuous-batching entry point: tokenize -> schedule -> batched thinker -> WAV.

    The engine loop mirrors vllm-omni (Q8a/Q9a):

        while scheduler.has_requests():
            out = scheduler.schedule()          # prefill + decode groups
            for prefill group: runner.prefill_group(group)
            for decode  group: runner.decode_group(group)
            scheduler.update_from_output(...)   # finish -> serial WAV chain (Q9a)

    ``max_batch`` (continuous-batching width) defaults to the deploy knob
    when a ``DeployConfig`` is supplied, otherwise to ``None -> 2``. Returns
    one :class:`AudioPayload` per prompt, in submission order.
    """
    import torch

    from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner
    from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav
    from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate
    from nanovllm_omni.outputs import AudioPayload

    if max_batch is None:
        max_batch = getattr(deploy, "max_batch", 2) if deploy is not None else 2
    # TK-007: per-replica isolation. With ``num_replicas > 1`` each replica
    # owns its own scheduler + runner; ``balancer`` (default RoundRobin)
    # picks which replica takes each prompt. Single-device default is 1
    # replica so the balancer is consulted once per prompt and always
    # returns 0 -- execution path is unchanged.
    if balancer is None:
        from nanovllm_omni.engine.load_balancer import RoundRobinBalancer

        balancer = RoundRobinBalancer()
    replicas: list[tuple[int, RuntimeScheduler, BatchedThinkerRunner]] = []
    for replica_id in range(num_replicas):
        sched = RuntimeScheduler(max_num_seqs=max_batch)
        runner = BatchedThinkerRunner(
            bundle,
            sched,
            temperature=temperature,
            top_p=top_p,
            rp=rp,
            max_new_tokens=max_new_tokens,
            open_thinking=open_thinking,
            base_seed=base_seed + replica_id,  # distinct RNG per replica
            kv_max_seq=kv_max_seq,
        )
        replicas.append((replica_id, sched, runner))

    # Stage id 0 = thinker for MiniMind-O; the balancer selects a replica.
    stage_id = 0
    assignment: list[tuple[str, BatchedThinkerRunner]] = []  # rid -> runner
    order: list[str] = []
    for prompt in prompts:
        replica_id = balancer.select(stage_id, num_replicas)
        _rid, _sched, runner = replicas[replica_id]
        ids = tokenize_for_generate(bundle.tokenizer, prompt, open_thinking)
        rid = runner.add_request(ids[0].tolist())
        assignment.append((rid, runner))
        order.append(rid)

    payloads: dict[str, Any] = {}
    with torch.no_grad():
        # Drive each replica's scheduler independently. The outer loop
        # terminates when no replica has work left.
        any_work = True
        while any_work:
            any_work = False
            for _replica_id, sched, runner in replicas:
                if not sched.has_work():
                    continue
                any_work = True
                out = sched.schedule()
                if out.is_empty:
                    continue
                prefilled: set[str] = set()
                finished: set[str] = set()
                for group in out.prefill_groups:
                    runner.prefill_group(group)
                    prefilled.update(chunk.seq.request_id for chunk in group.items)
                for group in out.decode_groups:
                    runner.decode_group(group)
                    for seq in group.items:
                        rid = seq.request_id
                        if runner.step_finished(rid):
                            finished.add(rid)
                # ``update_from_output`` returns rid -> num_tokens for just-
                # finished sequences; we reuse that as the iteration order
                # for codec hand-off below.
                newly = sched.update_from_output(prefilled=prefilled, finished=finished)
                for rid in newly:
                    st = runner.states[rid]
                    if not st.frames:
                        payloads[rid] = AudioPayload(data=b"", sample_rate=24_000)
                        continue
                    samples = decode_audio(bundle.mimi, st.frames, bundle.device)
                    wav = encode_wav(samples, sample_rate=24_000)
                    payloads[rid] = AudioPayload(data=wav, sample_rate=24_000)

    return [payloads[rid] for rid in order]


__all__ = ["run_batched_generate"]
