"""Engine-level continuous-batching scheduler skeleton (MiniMind-O thinker).

Teaching re-hash of vllm-omni's `core/sched/` shape, remade for this
project's scope: single process, single GPU, stdlib dataclasses, no new
dependencies.  It does NOT reuse `VLLMScheduler` (vllm-omni inherits it);
here the waiting/running/finished lifecycle and the "one forward is a
rectangular batch" constraint are reimplemented explicitly so the shape
stays readable.

Scope notes locked in the TICKET design session:

- Q8a  mixed batching: one ``schedule()`` emits prefill groups (new
  requests, grouped by identical prompt length) AND decode groups (running
  requests, grouped by identical ``start_pos``) -- the only groupings this
  model's single-scalar ``start_pos`` forward supports.
- Q4b/Q6a  real concurrency with fixed-slot KV: each running request owns
  one preallocated [layers, 2, kv_heads, max_seq, head_dim] slot written in
  place (no per-step ``torch.cat`` growth); decode groups gather rows back
  into one [B, ...] tensor for a batched forward.
- Q9a  the scheduler only drives the thinker stage; finished requests drop
  out to the serial talker/mimi->wav chain owned by the caller.
- No preemption / no paged blocks: admission is capped by ``max_batch``
  (ponytail: fixed-slot budgets; add preemption when a 4 GB card OOMs, and
  paged KV only if you outgrow this teaching shape).

The names deliberately mirror vllm-omni so the comparison is direct:
``OmniSchedulerOutput`` ~ ``_wrap_omni_scheduler_output``, ``SchedulerGroup``
~ vLLM's schedule-group concept, ``FixedKvSlotPool`` ~ the KV block manager
reduced to its teaching minimum.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class OmniRequestState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class OmniRequest:
    """A request as the scheduler sees it: prompt tokens + lifecycle.

    Generation state (KV slot, step, sampled tokens, audio codes) is NOT
    here; it lives in the model runner (``BatchedThinkerState``), matching
    vllm-omni's split where the scheduler owns the request and the runner
    owns the mutable per-request model state.
    """

    request_id: str
    prompt_ids: list[int]
    sampling: Any  # SamplingParams-ish; kept opaque so sched.py stays decoupled
    state: OmniRequestState = OmniRequestState.WAITING


@dataclass
class SchedulerGroup:
    """One unit of work a single model forward can execute.

    ``kind=="prefill"``: all ``req_ids`` share one prompt length; ``start_pos``
    is that length. ``kind=="decode"``: all ``req_ids`` are at identical
    ``start_pos`` so one rectangular [B, 9, 1] forward (with gathered KV) is
    legal -- the model reads ``start_pos`` as a single scalar.
    """

    kind: str  # "prefill" | "decode"
    req_ids: list[str]
    start_pos: int

    @property
    def is_empty(self) -> bool:
        return not self.req_ids

    def __len__(self) -> int:
        return len(self.req_ids)


@dataclass
class OmniSchedulerOutput:
    """Everything one scheduling round produced, in execution order.

    Prefill groups run first, decode groups after, so a newly admitted
    request's KV is written before it becomes eligible for a decode group
    next round (the same ordering vllm-omni's scheduler output implies).
    """

    prefill_groups: list[SchedulerGroup] = field(default_factory=list)
    decode_groups: list[SchedulerGroup] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_groups and not self.decode_groups


class OmniScheduler:
    """waiting/running/finished lifecycle + group formation.

    Plain-Python state machine (no tensors) so it is unit-testable without a
    GPU. Only token lists, ints, and sets cross the runner boundary.
    """

    def __init__(
        self,
        *,
        max_batch: int = 2,
        max_seq: int = 4096,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.max_batch = max_batch
        self.max_seq = max_seq
        self._requests: dict[str, OmniRequest] = {}
        self._waiting: list[str] = []
        self._running: list[str] = []
        self._finished: list[str] = []
        # Requests whose prefill forward completed (KV written): eligible to
        # join a decode group. Runner reports them via update_from_output.
        self._ready: set[str] = set()
        # request_id -> number of generated (decode) tokens so far.
        self._generated: dict[str, int] = {}

    # -- submission ---------------------------------------------------------

    def add_request(self, prompt_ids: list[int], request_id: str | None = None) -> str:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if len(prompt_ids) > self.max_seq:
            raise ValueError(f"prompt length {len(prompt_ids)} exceeds max_seq={self.max_seq}")
        rid = request_id or f"req-{len(self._requests)}"
        if rid in self._requests:
            raise ValueError(f"duplicate request_id {rid!r}")
        self._requests[rid] = OmniRequest(
            request_id=rid, prompt_ids=list(prompt_ids), sampling=None
        )
        self._waiting.append(rid)
        return rid

    # -- scheduling ---------------------------------------------------------

    def schedule(self) -> OmniSchedulerOutput:
        """Form prefill groups (waiting, by prompt length) then decode groups (ready running, by start_pos)."""
        out = OmniSchedulerOutput()

        # Admission: fill running up to max_batch with waiting requests (FCFS).
        free = self.max_batch - len(self._running)
        admitted: list[str] = []
        pending: list[str] = []
        for rid in self._waiting:
            if len(admitted) < free:
                admitted.append(rid)
            else:
                pending.append(rid)
        self._waiting = pending

        # Prefill groups: requests with identical prompt length batch together.
        by_len: dict[int, list[str]] = {}
        for rid in admitted:
            length = len(self._requests[rid].prompt_ids)
            by_len.setdefault(length, []).append(rid)
        for length, req_ids in sorted(by_len.items()):
            out.prefill_groups.append(
                SchedulerGroup(kind="prefill", req_ids=req_ids, start_pos=length)
            )
            for rid in req_ids:
                self._running.append(rid)
                self._requests[rid].state = OmniRequestState.RUNNING

        # Decode groups: only requests whose prefill already ran (``_ready``),
        # grouped by identical current start_pos (= prompt len + generated).
        by_pos: dict[int, list[str]] = {}
        for rid in self._running:
            if rid not in self._ready:
                continue
            pos = self._total_len(rid)
            by_pos.setdefault(pos, []).append(rid)
        for pos, req_ids in sorted(by_pos.items()):
            out.decode_groups.append(SchedulerGroup(kind="decode", req_ids=req_ids, start_pos=pos))

        return out

    # -- progress -----------------------------------------------------------

    def _total_len(self, rid: str) -> int:
        return len(self._requests[rid].prompt_ids) + self._generated.get(rid, 0)

    def update_from_output(
        self,
        *,
        prefilled: set[str] | None = None,
        generated: dict[str, int] | None = None,
        finished: set[str] | None = None,
    ) -> dict[str, int]:
        """Advance progress from one round's runner output.

        ``prefilled``: request_ids whose prefill forward completed this round
        (now eligible for decode groups).
        ``generated``: request_id -> running count of generated tokens.
        ``finished``: request_ids whose thinker generation is done.

        Returns ``{request_id: generated_count}`` for just-finished requests
        so the caller frees their KV and hands them to the serial codec chain.
        """
        self._ready |= set(prefilled or ())
        for rid, count in (generated or {}).items():
            self._generated[rid] = count
        newly_finished: dict[str, int] = {}
        for rid in list(self._running):
            if rid in (finished or ()):
                self._running.remove(rid)
                self._requests[rid].state = OmniRequestState.FINISHED
                self._finished.append(rid)
                newly_finished[rid] = self._generated.get(rid, 0)
        return newly_finished

    # -- queries ------------------------------------------------------------

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def request(self, rid: str) -> OmniRequest:
        return self._requests[rid]

    def prompt_ids(self, rid: str) -> list[int]:
        return self._requests[rid].prompt_ids

    def is_finished(self, rid: str) -> bool:
        return self._requests[rid].state is OmniRequestState.FINISHED


class FixedKvSlotPool:
    """Per-request fixed-size KV slots with in-place writes.

    Each running request owns one ``[n_layers, 2, max_seq, kv_heads, head_dim]``
    buffer matching the model's KV layout ``[B, seq, kv_heads, d]`` (seq at
    dim 1 -- the vendored attention cats on ``dim=1`` and the omni forward
    reads ``start_pos = past_key_values[0][0].shape[1]``). The runner writes
    newly computed per-layer (key, value) rows into the slot; ``gather``
    re-stacks the alive rows of a decode group into one ``[B, seq, kv, d]``
    (key, value) pair per layer so a single forward processes them together.

    Tensor-free itself; all shapes are supplied by the runner at register time.
    """

    def __init__(self, max_seq: int) -> None:
        self.max_seq = max_seq
        self._slots: dict[str, dict[str, Any]] = {}

    def register(
        self,
        req_id: str,
        *,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: Any,
        dtype: Any,
    ) -> None:
        import torch

        buf = torch.zeros(
            n_layers,
            2,
            self.max_seq,
            n_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self._slots[req_id] = {"buf": buf, "ptr": 0}

    def write(self, req_id: str, layer: int, key: Any, value: Any, row: int = 0) -> None:
        """Copy one layer's (key, value) rows [B, seq, kv, d] into the slot.

        ``key[row]`` / ``value[row]`` are ``[seq, kv_heads, d]`` and are stored
        into layer ``layer``; the write pointer advances to the sequence length.
        """
        slot = self._slots[req_id]
        k = key[row]
        v = value[row]
        length = k.shape[0]
        if length > self.max_seq:
            raise ValueError(f"seq length {length} exceeds pool max_seq={self.max_seq}")
        slot["buf"][layer, 0, :length, :, :] = k
        slot["buf"][layer, 1, :length, :, :] = v
        slot["ptr"] = max(slot["ptr"], length)

    def length(self, req_id: str) -> int:
        return self._slots[req_id]["ptr"]

    def gather(self, req_ids: list[str]) -> list[tuple[Any, Any]]:
        """Return per-layer ``[(key, value), ...]`` stacked over the group.

        Caller guarantees all ``req_ids`` share the same ``ptr`` (a decode
        group formed on identical ``start_pos``), so stacking is rectangular.
        Each pair is ``[B, seq, kv_heads, head_dim]`` matching the model KV
        layout that ``cat(..., dim=1)`` expects.
        """
        import torch

        if not req_ids:
            return []
        slots = [self._slots[r] for r in req_ids]
        lengths = {s["ptr"] for s in slots}
        if len(lengths) != 1:
            raise ValueError(
                "gather requires a group of equal KV length, got "
                f"{[s['ptr'] for s in slots]} -- run prefill first"
            )
        length = next(iter(lengths))
        n_layers = slots[0]["buf"].shape[0]
        pairs: list[tuple[Any, Any]] = []
        for layer in range(n_layers):
            keys = torch.stack([s["buf"][layer, 0, :length, :, :] for s in slots])
            vals = torch.stack([s["buf"][layer, 1, :length, :, :] for s in slots])
            pairs.append((keys, vals))
        return pairs

    def release(self, req_id: str) -> None:
        self._slots.pop(req_id, None)


__all__ = [
    "FixedKvSlotPool",
    "OmniRequest",
    "OmniRequestState",
    "OmniScheduler",
    "OmniSchedulerOutput",
    "SchedulerGroup",
]
