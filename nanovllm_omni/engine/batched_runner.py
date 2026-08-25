"""Batched engine entry: Orchestrator-driven MiniMind-O continuous batching.

Engine-level entry for the MiniMind-O single-stage (thinker) batched path.
The engine loop itself (schedule -> execute groups -> update) lives in the
model-agnostic ``engine/orchestrator.py`` (single-device vllm-omni shape);
this module owns the MiniMind-O seams:

  - tokenize each prompt into ids (thinker stage)
  - build the thinker ``StagePool``: one ``RuntimeScheduler`` +
    ``BatchedThinkerRunner`` per replica
  - ``drive`` one replica's scheduler to completion, collecting finished
    rids -> Mimi frames
  - ``finalize`` a finished rid through the serial codec chain (talker /
    mimi decode -> WAV), matching Q9a

Layering:
- ``engine/runtime_scheduler.py`` -> per-replica scheduler + Sequence
- ``engine/orchestrator.py``     -> StagePool + request dispatch
- ``engine/batched_runner.py``   -> this file (MiniMind-O seams)
- ``models/minimind_omni/batched_generation.py`` -> per-request runner
"""

from __future__ import annotations

from itertools import count
from typing import Any

from nanovllm_omni.engine.orchestrator import Orchestrator, Replica, StagePool

# Globally-unique request id across replicas. Each replica's
# ``BatchedThinkerRunner.add_request`` would otherwise default to
# ``req-<local count>`` and collide across replicas (two ``req-0``). The
# leading ``seq-`` prefix keeps ids distinct from replica-internal naming.
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
    kv_max_seq: int | None = None,
    num_replicas: int = 1,
    balancer: Any | None = None,
) -> list[Any]:
    """Continuous-batching entry: tokenize -> Orchestrator -> serial WAV.

    Builds a single-stage (thinker) ``StagePool`` with ``num_replicas``
    replicas (each a ``RuntimeScheduler`` + ``BatchedThinkerRunner``), fans
    the prompts out via the ``LoadBalancer``, drives every replica to
    completion, then runs each finished rid through the serial codec chain
    (mimi decode -> WAV). Returns one :class:`AudioPayload` per prompt in
    submission order.
    """
    from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler
    from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner
    from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav
    from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate
    from nanovllm_omni.outputs import AudioPayload

    if max_batch is None:
        max_batch = getattr(deploy, "max_batch", 2) if deploy is not None else 2
    if balancer is None:
        from nanovllm_omni.engine.load_balancer import RoundRobinBalancer

        balancer = RoundRobinBalancer()

    # Per-replica isolation: each replica owns an independent scheduler +
    # runner (TK-007). Distinct base_seed per replica for independent RNG.
    pool = StagePool(stage_id=0, num_replicas=num_replicas, balancer=balancer)
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
            base_seed=base_seed + replica_id,
            kv_max_seq=kv_max_seq,
        )
        pool.add_replica(sched, runner)

    orch = Orchestrator(pool)

    def add_request(replica: Replica, prompt: str) -> str:
        ids = tokenize_for_generate(bundle.tokenizer, prompt, open_thinking)
        return replica.runner.add_request(ids[0].tolist(), request_id=f"seq-{next(_rids)}")

    def drive(replica: Replica) -> dict[str, Any]:
        sched = replica.sched
        runner = replica.runner
        done: dict[str, Any] = {}
        while sched.has_work():
            out = sched.schedule()
            if out.is_empty:
                break
            prefilled: set[str] = set()
            finished: set[str] = set()
            for group in out.prefill_groups:
                runner.prefill_group(group)
                prefilled.update(chunk.seq.request_id for chunk in group.items)
            for group in out.decode_groups:
                runner.decode_group(group)
                for seq in group.items:
                    if runner.step_finished(seq.request_id):
                        finished.add(seq.request_id)
            sched.update_from_output(prefilled=prefilled, finished=finished)
            for rid in finished:
                st = runner.states[rid]
                if not st.frames:
                    done[rid] = AudioPayload(data=b"", sample_rate=24_000)
                else:
                    samples = decode_audio(bundle.mimi, st.frames, bundle.device)
                    wav = encode_wav(samples, sample_rate=24_000)
                    done[rid] = AudioPayload(data=wav, sample_rate=24_000)
        return done

    return orch.submit(prompts, add_request=add_request, drive=drive)


__all__ = ["run_batched_generate"]
