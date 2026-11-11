"""MiniMind-O decode recipe on top of the unified fork-AR engine.

The fork-attention lifecycle (``set_context`` / ``forward`` / ``sample``
/ ``reset_context``) lives in ``engine/stage_runner.py::StageRunner`` —
shared by any family that uses fork attention. This module is the
MiniMind-O specific decode dance: vendor ``stream_generate`` prefill +
AR loop with the post-EOS filler clocking, plus the one-shot full-
sequence prefill that replaces the paged-KV bridge so the talker sees
bit-equal rows.

The decode is continuous: ``ThinkerDecodeSession`` owns one fork
``Scheduler`` for the life of the batch, and ``step`` advances it by one
``schedule()`` round. A request added between steps waits in the
scheduler and joins the running batch on the next prefill — prefill takes
priority over decode in the fork scheduler, exactly as in vLLM. Two
things make that safe per request:

- each sequence draws from its own ``torch.Generator`` (CUDA) or RNG key
  (MLX), so a request's tokens do not depend on which requests share its
  batch;
- the bridge rows are accumulated per sequence, so a batched step
  appends one row per sequence instead of only row 0.

We compose on top of ``StageRunner`` rather than subclassing, so the
engine has one cohesive AR adapter and model families stay side-effect
free at import time. ``decode_minimind`` / ``decode_minimind_mlx`` are
the one-shot drains built on the session; the stage calls ``step``
directly so new requests can join mid-generation.

The MLX track shares the scheduler with the CUDA track and forks only at
the physical step: MLX has no paged KV, so a joining request is prefilled
into a fresh ``BatchKVCache`` and ``extend``-ed onto the running one
(ADR 0025).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ThinkerDecode:
    """Per-request inputs for one thinker decode."""

    token_ids: list[int]
    max_tokens: int
    temperature: float
    repetition_penalty: float = 1.05
    top_p: float = 1.0
    seed: int | None = None
    stop_hit: Callable[[list[int]], bool] | None = None


@dataclass
class ThinkerDecoded:
    """Per-request outputs of one thinker decode."""

    generated: list[int]
    bridge_hidden: Any
    text_state: Any


@dataclass
class _SeqState:
    """Mutable per-sequence state inside the decode loop."""

    request: ThinkerDecode
    generated: list[int] = field(default_factory=list)
    bridge_hidden: Any = None
    generator: Any = None
    text_state: Any = None
    text_finished: bool = False
    emitted_filler: bool = False
    content_len: int | None = None


@dataclass(frozen=True)
class _MlxSampling:
    """The sampling fields the backend's MLX sampler reads."""

    top_p: float
    extra: dict[str, Any]


def _apply_repetition_penalty(logits: Any, token_ids: list[int], penalty: float) -> None:
    """Divide every already-seen token's logit by ``penalty``, in place.

    One indexed divide instead of one in-place op per token. The history is
    re-scanned on every step, and each per-token ``/=`` costs two kernel
    launches, so the loop's cost grows with the token set (~0.33 ms/step on
    RTX 3050 at 120 steps). Same arithmetic either way: every element is
    divided exactly once by the same scalar.
    """
    import torch

    history = torch.tensor(list(set(token_ids)), dtype=torch.long, device=logits.device)
    logits[history] /= penalty


def _sampler_with_rp(scheduled: list[Any], states: dict[Any, _SeqState]) -> Any:
    """Vendor sampler for one CUDA step: last row, penalty, top-p, multinomial."""

    def sampler(logits: Any, temperatures: Any) -> Any:
        # Match vendor stream_generate: take the LAST position of each row
        # only, divide by temperature, apply the full-history penalty and
        # top-p filter, then call torch.multinomial directly. Gumbel-max
        # is distributionally equivalent but consumes a different RNG
        # path; a different text token changes every later
        # bridge/audio-buffer row.
        #
        # ``.float()`` is load-bearing on RTX 3050 / torch 2.11: the
        # minimind-3o lm_head emits BF16 logits that overflow
        # ``torch.softmax`` in low precision (NaN/inf), then
        # ``torch.multinomial`` rejects with ``device-side assert:
        # probability tensor contains inf, nan or element < 0``.
        # Vendor's reference is FP32 throughout; we mirror that here.
        import torch

        sampled = []
        for position, sequence in enumerate(scheduled):
            state = states[sequence]
            logits_i = logits[position, :].float() / (temperatures[position] + 1e-9)
            _apply_repetition_penalty(
                logits_i, sequence.token_ids, state.request.repetition_penalty
            )
            if state.request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
                remove = (
                    torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > state.request.top_p
                )
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits_i[sorted_indices[remove]] = -float("inf")
            probs = torch.softmax(logits_i, dim=-1)
            if state.generator is None:
                # One generator per sequence: the draw a request gets must
                # not depend on how many other requests share its batch.
                state.generator = torch.Generator(device=logits.device)
                if state.request.seed is not None:
                    state.generator.manual_seed(int(state.request.seed))
                else:
                    # A fresh generator has a fixed default seed, so leaving
                    # it alone would make every unseeded request draw the
                    # same stream. Draw from the global RNG instead, which
                    # advances per request the way it did before.
                    state.generator.manual_seed(
                        int(torch.empty((), dtype=torch.int64).random_().item())
                    )
            sampled.append(torch.multinomial(probs, 1, generator=state.generator).view(-1))
        # Flat ``[batch]`` of token ids, matching the fork sampler's shape:
        # ``stage_runner.sample(...).tolist()`` is read as one id per
        # sequence.
        return torch.cat(sampled)

    return sampler


class ThinkerDecodeSession:
    """One continuous thinker batch: a fork ``Scheduler`` plus its state.

    ``step`` schedules one round, runs it, and returns the requests that
    finished in that round. ``add`` may be called between steps; the new
    sequence stays in the scheduler's waiting queue until the next
    ``schedule()``, which prefills it ahead of the running decode. A
    finished sequence leaves ``scheduler.running`` in ``postprocess`` and
    its KV row is recycled in the same step, so the batch never waits for
    the slowest member.
    """

    def __init__(
        self,
        stage_runner: Any,
        *,
        backend: str,
        eos_id: int | None = None,
        enter_token: int = 0,
        pad_token: int = 0,
        talker_tail: int = 8,
    ) -> None:
        from nanovllm.engine.scheduler import Scheduler

        if backend not in ("torch", "mlx"):
            raise ValueError(f"unknown thinker backend {backend!r}; expected 'torch' or 'mlx'")
        self._runner = stage_runner
        self._backend = backend
        self._enter_token = enter_token
        self._pad_token = pad_token
        self._talker_tail = talker_tail

        config = stage_runner.config
        if eos_id is None:
            eos_id = getattr(config, "eos", None)
        if eos_id is None and backend == "mlx":
            # A hand-built MlxConfig (the fake-model test) leaves ``eos``
            # None; the runner fills it from the tokenizer when it can.
            eos_id = min(int(token) for token in stage_runner.tokenizer.eos_token_ids)
        self._eos_id = eos_id

        self._scheduler = Scheduler(config)
        self._states: dict[Any, _SeqState] = {}
        self._requests: dict[Any, ThinkerDecode] = {}
        self._request_ids: dict[Any, str] = {}

        self._cache_sequences: list[Any] = []
        if backend == "mlx":
            import mlx.core as mx

            from nanovllm_omni.backend.mlx_backend import _make_sampler, _seed_key

            self._mx = mx
            self._make_sampler = _make_sampler
            self._seed_key = _seed_key
            self._sampling_by_sequence: dict[Any, _MlxSampling] = {}
            self._rng_key_by_sequence: dict[Any, Any] = {}

    # -- queue -------------------------------------------------------------

    def add(self, request: ThinkerDecode, request_id: str) -> None:
        """Queue one request; it joins the batch on the next step."""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

        sequence = Sequence(
            request.token_ids,
            ForkSamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=True,
            ),
        )
        self._scheduler.add(sequence)
        self._states[sequence] = _SeqState(request=request)
        self._requests[sequence] = request
        self._request_ids[sequence] = request_id
        if self._backend == "mlx":
            self._sampling_by_sequence[sequence] = _MlxSampling(
                top_p=request.top_p,
                extra={"repetition_penalty": request.repetition_penalty},
            )
            self._rng_key_by_sequence[sequence] = self._seed_key(self._mx, request)

    def has_work(self) -> bool:
        """Whether any added request has not finished."""
        return not self._scheduler.is_finished()

    def active_request_ids(self) -> list[str]:
        """Ids of every request still waiting or running."""
        sequences = (*self._scheduler.waiting, *self._scheduler.running)
        return [self._request_ids[sequence] for sequence in sequences]

    # -- one incremental step ---------------------------------------------

    def step(self) -> list[tuple[str, ThinkerDecode, ThinkerDecoded]]:
        """Run one ``schedule()`` round; return the requests that finished."""
        if not self.has_work():
            return []

        seqs, is_prefill = self._scheduler.schedule()
        if self._backend == "torch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            # The cache rows must match ``seqs`` before ``prepare_*``; a
            # sequence that finished or was preempted last step is gone
            # from ``scheduler.running`` and must lose its row here.
            self._filter_inactive_cache_rows()

        self._runner.set_context(is_prefill=is_prefill)
        input_ids, positions = (
            self._runner.prepare_prefill(seqs) if is_prefill else self._runner.prepare_decode(seqs)
        )
        temperatures = self._runner.prepare_sample(seqs)
        logits = self._runner.forward(input_ids, positions, is_prefill)

        if self._backend == "torch":
            self._collect_bridge_torch(seqs, is_prefill)
            token_ids = self._runner.sample(
                logits, temperatures, sampler=_sampler_with_rp(seqs, self._states)
            ).tolist()
        else:
            self._collect_bridge_mlx(seqs, is_prefill)
            token_ids = self._sample_mlx(seqs, logits, temperatures)

        effective_ids = self._advance_states(seqs, token_ids, is_prefill, logits)
        if is_prefill and self._backend == "mlx":
            self._cache_sequences.extend(seqs)
        self._scheduler.postprocess(seqs, effective_ids, is_prefill)
        self._runner.reset_context()
        if self._backend == "mlx":
            self._filter_inactive_cache_rows()
        return self._take_finished(seqs)

    # -- backend physical work --------------------------------------------

    def _collect_bridge_torch(self, seqs: list[Any], is_prefill: bool) -> None:
        """Bridge rows are packed by token, not by sequence.

        Sequence i owns ``num_scheduled_tokens`` consecutive rows of the
        buffer on prefill and one row on decode.
        """
        import torch

        bridge = self._runner.bridge_hidden()
        offset = 0
        for sequence in seqs:
            width = sequence.num_scheduled_tokens if is_prefill else 1
            if bridge is not None:
                rows = bridge[offset : offset + width].detach().clone()
                state = self._states[sequence]
                state.bridge_hidden = (
                    rows
                    if state.bridge_hidden is None
                    else torch.cat([state.bridge_hidden, rows], dim=0)
                )
            offset += width

    def _collect_bridge_mlx(self, seqs: list[Any], is_prefill: bool) -> None:
        import torch

        bridge = self._runner.bridge_hidden()
        for index, sequence in enumerate(seqs):
            if bridge is None:
                continue
            rows = bridge[index]
            rows = rows[-len(sequence.token_ids) :] if is_prefill else rows[-1:]
            captured = _mx_to_torch(rows)
            state = self._states[sequence]
            if is_prefill or state.bridge_hidden is None:
                state.bridge_hidden = captured
            else:
                state.bridge_hidden = torch.cat((state.bridge_hidden, captured), dim=0)

    def _sample_mlx(self, seqs: list[Any], logits: Any, temperatures: Any) -> list[int]:
        current_keys = [self._rng_key_by_sequence[sequence] for sequence in seqs]
        sampler = self._make_sampler(
            self._mx,
            seqs,
            [self._sampling_by_sequence[sequence] for sequence in seqs],
            current_keys,
        )
        sampled_ids = self._runner.sample(logits, temperatures, sampler=sampler).tolist()
        for sequence, key in zip(seqs, current_keys, strict=True):
            self._rng_key_by_sequence[sequence] = key
        return sampled_ids

    def _filter_inactive_cache_rows(self) -> None:
        keep_indices = [
            index
            for index, sequence in enumerate(self._cache_sequences)
            if sequence in self._scheduler.running
        ]
        if len(keep_indices) == len(self._cache_sequences):
            return
        self._runner.filter_cache_rows(keep_indices)
        self._cache_sequences = [self._cache_sequences[index] for index in keep_indices]

    # -- per-request bookkeeping ------------------------------------------

    def _advance_states(
        self,
        seqs: list[Any],
        token_ids: list[int],
        is_prefill: bool,
        logits: Any,
    ) -> list[int]:
        """Append this step's token per sequence and apply the stop rules."""
        effective_ids = []
        for position, (sequence, sampled_id) in enumerate(zip(seqs, token_ids, strict=True)):
            state = self._states[sequence]
            # Vendor ``stream_generate`` does NOT end the loop at text EOS:
            # it sets ``text_finished`` and keeps clocking the forward pass
            # with throwaway filler so the talker tail can drain. The
            # sampler still runs on those steps so the RNG stream advances
            # identically.
            if state.text_finished:
                effective = self._pad_token if state.emitted_filler else self._enter_token
                state.emitted_filler = True
            else:
                effective = sampled_id
                if sampled_id == self._eos_id:
                    state.text_finished = True
                    state.content_len = len(state.generated)
            effective_ids.append(effective)
            state.generated.append(effective)
            if not is_prefill:
                state.text_state = (
                    _mx_to_torch(logits[position])
                    if self._backend == "mlx"
                    else logits[position].detach()
                )
            if state.request.stop_hit is not None and state.request.stop_hit(state.generated):
                # Cap this sequence at the token it just produced so the
                # fork's own postprocess finishes it; the rest of the
                # batch keeps decoding.
                sequence.max_tokens = sequence.num_completion_tokens + 1
            if (
                state.text_finished
                and state.content_len is not None
                and len(state.generated) >= state.content_len + self._talker_tail
            ):
                # The talker only needs ``content_len + tail`` bridge rows;
                # stop the post-EOS filler clock for this sequence.
                sequence.max_tokens = sequence.num_completion_tokens + 1
        return effective_ids

    def _take_finished(self, seqs: list[Any]) -> list[tuple[str, ThinkerDecode, ThinkerDecoded]]:
        finished = []
        for sequence in seqs:
            if not sequence.is_finished:
                continue
            state = self._states[sequence]
            finished.append(
                (self._request_ids[sequence], self._requests[sequence], self._result(state))
            )
        return finished

    def _result(self, state: _SeqState) -> ThinkerDecoded:
        full_ids = list(state.request.token_ids) + list(state.generated)
        if self._backend == "mlx":
            # The decode loop forwarded one token per step against the
            # dense cache, but the vendor runs full-sequence attention
            # every step. The rows are not bit-equal and the talker
            # cross-attends to them, so the bridge comes from a full-
            # sequence re-run (ADR 0025).
            bridge = self._runner.recapture_bridge(full_ids)
            bridge_hidden = _mx_to_torch(bridge) if bridge is not None else state.bridge_hidden
        else:
            bridge_hidden = recapture_bridge_minimind(self._runner, full_ids, state.bridge_hidden)
        return ThinkerDecoded(
            generated=state.generated,
            bridge_hidden=bridge_hidden,
            text_state=state.text_state,
        )


def _drain_session(
    session: ThinkerDecodeSession, requests: list[ThinkerDecode]
) -> list[ThinkerDecoded]:
    request_ids = [str(index) for index in range(len(requests))]
    for request, request_id in zip(requests, request_ids, strict=True):
        session.add(request, request_id)
    results: dict[str, ThinkerDecoded] = {}
    while session.has_work():
        for request_id, _, decoded in session.step():
            results[request_id] = decoded
    return [results[request_id] for request_id in request_ids]


def decode_minimind(
    stage_runner: Any,
    requests: list[ThinkerDecode],
    *,
    eos_id: int | None = None,
    enter_token: int = 0,
    pad_token: int = 0,
    talker_tail: int = 8,
) -> list[ThinkerDecoded]:
    """Run the vendor decode dance for ``requests`` on one ``StageRunner``.

    One-shot drain of ``ThinkerDecodeSession``; the stage calls the
    session directly so requests can join between steps.

    ``enter_token`` is emitted on a sequence's first post-EOS step (the
    talker tail needs a stable anchor) and ``pad_token`` on every later
    one. ``stop_hit`` ends the one request that matched; the rest of the
    batch keeps decoding.
    """
    if not requests:
        return []
    session = ThinkerDecodeSession(
        stage_runner,
        backend="torch",
        eos_id=eos_id,
        enter_token=enter_token,
        pad_token=pad_token,
        talker_tail=talker_tail,
    )
    return _drain_session(session, requests)


def recapture_bridge_minimind(
    stage_runner: Any,
    full_ids: list[int],
    bridge_hidden: Any,
) -> Any:
    """One full-sequence prefill to replace the paged-KV bridge.

    The decode loop forwards one token per step (paged KV cache),
    but vendor ``MiniMindOmni`` runs full-sequence attention each
    step (``use_cache=False``). The rows are not bit-equal, and the
    talker cross-attends to them — so re-run the full sequence once
    and take that bridge instead.

    The re-run detaches the paged KV tensors first. A request can finish
    while others are still decoding, and a prefill writes the KV it
    computes; with the paged cache attached that write would land on
    slots owned by the running sequences. With the cache detached the
    attention uses the freshly computed K/V and writes nothing.
    """
    import torch

    if not full_ids or bridge_hidden is None:
        return bridge_hidden
    model = stage_runner.model
    n_total = len(full_ids)
    dev = bridge_hidden.device
    caches = [
        (module, module.k_cache, module.v_cache)
        for module in model.modules()
        if hasattr(module, "k_cache") and hasattr(module, "v_cache")
    ]
    empty = torch.tensor([])
    stage_runner.set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, n_total], dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.tensor([0, n_total], dtype=torch.int32, device=dev),
        max_seqlen_q=n_total,
        max_seqlen_k=n_total,
        slot_mapping=torch.arange(n_total, dtype=torch.int32, device=dev),
        context_lens=None,
        block_tables=None,
    )
    try:
        for module, _, _ in caches:
            module.k_cache = module.v_cache = empty
        with torch.no_grad():
            _in = torch.tensor(full_ids, dtype=torch.int64, device=dev)
            _pos = torch.arange(n_total, dtype=torch.int64, device=dev)
            _ = model(_in, _pos)
        _full_bridge = stage_runner.bridge_hidden()
        if _full_bridge is not None:
            # The buffer holds ``max_position`` rows; only the first
            # ``n_total`` carry this sequence. The talker embeds every
            # row it gets, so the trailing zero rows must not travel.
            bridge_hidden = _full_bridge[:n_total].detach().to(dev).clone()
    finally:
        for module, k_cache, v_cache in caches:
            module.k_cache = k_cache
            module.v_cache = v_cache
        stage_runner.reset_context()
    return bridge_hidden


def _mx_to_torch(array: Any) -> Any:
    """An MLX array as a float32 torch tensor."""
    import mlx.core as mx
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(array.astype(mx.float32)))


def decode_minimind_mlx(
    stage_runner: Any,
    requests: list[ThinkerDecode],
    *,
    eos_id: int | None = None,
    enter_token: int = 0,
    pad_token: int = 0,
    talker_tail: int = 8,
) -> list[ThinkerDecoded]:
    """Decode requests with the fork Scheduler and MLX cache rows.

    One-shot drain of ``ThinkerDecodeSession``. Sampling matches the CUDA
    thinker's temperature, repetition penalty, top-p, and per-request RNG
    keys. The bridge rows come from a full-sequence re-run per request,
    the same way the CUDA track gets them.
    """
    if not requests:
        return []
    session = ThinkerDecodeSession(
        stage_runner,
        backend="mlx",
        eos_id=eos_id,
        enter_token=enter_token,
        pad_token=pad_token,
        talker_tail=talker_tail,
    )
    return _drain_session(session, requests)


__all__ = [
    "ThinkerDecode",
    "ThinkerDecodeSession",
    "ThinkerDecoded",
    "decode_minimind",
    "decode_minimind_mlx",
    "recapture_bridge_minimind",
]
