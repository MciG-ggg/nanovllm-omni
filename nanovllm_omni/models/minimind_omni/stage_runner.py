"""MiniMind-O decode recipe on top of the unified fork-AR engine.

The fork-attention lifecycle (``set_context`` / ``forward`` / ``sample``
/ ``reset_context``) lives in ``engine/stage_runner.py::StageRunner`` —
shared by any family that uses fork attention. This module is the
MiniMind-O specific decode dance: vendor ``stream_generate`` prefill +
AR loop with the post-EOS filler clocking, plus the one-shot full-
sequence prefill that replaces the paged-KV bridge so the talker sees
bit-equal rows.

The decode is batched: every request the thinker stage is holding goes
into one fork ``Scheduler``, so one step serves all of them and how many
travel together is the fork's own ``max_num_seqs``. Two things make that
safe per request:

- each sequence draws from its own ``torch.Generator``, so a request's
  tokens do not depend on which requests share its batch;
- the bridge rows are accumulated per sequence, so a batched step
  appends one row per sequence instead of only row 0.

We compose on top of ``StageRunner`` rather than subclassing, so the
engine has one cohesive AR adapter and model families stay side-effect
free at import time. The recipe is two top-level functions —
``decode_minimind`` and ``recapture_bridge_minimind``.

The MLX track has its own driver, ``decode_minimind_mlx``: the vendor
dance over one dense batch, with the same sampling recipe (the one
``backend.mlx_backend.generate_mlx`` uses) so the two tracks draw from
the same distribution.
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

    Owns the per-sequence bridge-row accumulation and the post-EOS filler
    clocking.

    ``enter_token`` is emitted on a sequence's first post-EOS step (the
    talker tail needs a stable anchor) and ``pad_token`` on every later
    one. ``stop_hit`` ends the one request that matched; the rest of the
    batch keeps decoding.
    """
    import torch
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

    config = stage_runner.config
    if eos_id is None:
        eos_id = config.eos

    scheduler = Scheduler(config)
    states: dict[Any, _SeqState] = {}
    for request in requests:
        sequence = Sequence(
            request.token_ids,
            ForkSamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=True,
            ),
        )
        scheduler.add(sequence)
        states[sequence] = _SeqState(request=request)

    # Which sequences this step is serving, in logits-row order. The fork's
    # ``ParallelLMHead`` already reduces a prefill to one row per sequence
    # (``cu_seqlens_q[1:] - 1``), so row i belongs to ``scheduled[i]`` on
    # both prefill and decode. Written by the loop, read by the sampler.
    scheduled: list[Any] = []

    def _sampler_with_rp(logits, temperatures):
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

    while not scheduler.is_finished():
        seqs, is_prefill = scheduler.schedule()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_ids, positions = (
            stage_runner.prepare_prefill(seqs) if is_prefill else stage_runner.prepare_decode(seqs)
        )
        temperatures = stage_runner.prepare_sample(seqs)
        logits = stage_runner.forward(input_ids, positions, is_prefill)

        # Bridge rows are packed by token, not by sequence: sequence i
        # owns ``num_scheduled_tokens`` consecutive rows of the buffer.
        bridge = stage_runner.bridge_hidden()
        scheduled.clear()
        offset = 0
        for sequence in seqs:
            width = sequence.num_scheduled_tokens if is_prefill else 1
            scheduled.append(sequence)
            if bridge is not None:
                rows = bridge[offset : offset + width].detach().clone()
                state = states[sequence]
                state.bridge_hidden = (
                    rows
                    if state.bridge_hidden is None
                    else torch.cat([state.bridge_hidden, rows], dim=0)
                )
            offset += width

        token_ids = stage_runner.sample(logits, temperatures, sampler=_sampler_with_rp).tolist()

        # Vendor ``stream_generate`` does NOT end the loop at text EOS:
        # it sets ``text_finished`` and keeps clocking the forward pass
        # with throwaway filler so the talker tail can drain. The
        # sampler still runs on those steps so the RNG stream advances
        # identically.
        effective_ids = []
        for position, (sequence, sampled_id) in enumerate(zip(seqs, token_ids, strict=True)):
            state = states[sequence]
            if state.text_finished:
                effective_ids.append(pad_token if state.emitted_filler else enter_token)
                state.emitted_filler = True
            else:
                effective_ids.append(sampled_id)
                if sampled_id == eos_id:
                    state.text_finished = True
                    state.content_len = len(state.generated)
            state.generated.append(effective_ids[-1])
            if not is_prefill:
                state.text_state = logits[position].detach()
            if state.request.stop_hit is not None and state.request.stop_hit(state.generated):
                # Cap this sequence at the token it just produced so the
                # fork's own postprocess finishes it; the rest of the
                # batch keeps decoding.
                sequence.max_tokens = sequence.num_completion_tokens + 1
            if (
                state.text_finished
                and state.content_len is not None
                and len(state.generated) >= state.content_len + talker_tail
            ):
                # The talker only needs ``content_len + tail`` bridge rows;
                # stop the post-EOS filler clock for this sequence.
                sequence.max_tokens = sequence.num_completion_tokens + 1

        scheduler.postprocess(seqs, effective_ids, is_prefill)
        stage_runner.reset_context()

    return [
        ThinkerDecoded(
            generated=state.generated,
            bridge_hidden=state.bridge_hidden,
            text_state=state.text_state,
        )
        for state in states.values()
    ]


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
    """
    import torch

    if not full_ids or bridge_hidden is None:
        return bridge_hidden
    model = stage_runner.model
    n_total = len(full_ids)
    dev = bridge_hidden.device
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

    Sampling matches the CUDA thinker's temperature, repetition penalty,
    top-p, and per-request RNG keys. Bridge rows are collected from scheduled
    forwards; the last sampled token is not re-run to capture another row.
    """
    import mlx.core as mx
    import torch
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

    from nanovllm_omni.backend.mlx_backend import _make_sampler, _seed_key

    if not requests:
        return []
    if eos_id is None:
        # The runner fills ``config.eos`` from the tokenizer; a hand-built
        # MlxConfig (the fake-model test) leaves it None.
        eos_id = stage_runner.config.eos
        if eos_id is None:
            eos_id = min(int(token) for token in stage_runner.tokenizer.eos_token_ids)

    scheduler = Scheduler(stage_runner.config)
    states: dict[Any, _SeqState] = {}
    sampling_by_sequence: dict[Any, _MlxSampling] = {}
    rng_key_by_sequence: dict[Any, Any] = {}
    for request in requests:
        sequence = Sequence(
            request.token_ids,
            ForkSamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=True,
            ),
        )
        scheduler.add(sequence)
        states[sequence] = _SeqState(request=request)
        sampling_by_sequence[sequence] = _MlxSampling(
            top_p=request.top_p,
            extra={"repetition_penalty": request.repetition_penalty},
        )
        rng_key_by_sequence[sequence] = _seed_key(mx, request)

    cache_sequences = []

    def _filter_inactive_cache_rows() -> None:
        nonlocal cache_sequences
        keep_indices = [
            index for index, sequence in enumerate(cache_sequences) if sequence in scheduler.running
        ]
        if len(keep_indices) == len(cache_sequences):
            return
        stage_runner.filter_cache_rows(keep_indices)
        cache_sequences = [cache_sequences[index] for index in keep_indices]

    while not scheduler.is_finished():
        if not scheduler.waiting and not scheduler.running:
            break
        seqs, is_prefill = scheduler.schedule()
        _filter_inactive_cache_rows()
        stage_runner.set_context(is_prefill=is_prefill)
        input_ids, positions = (
            stage_runner.prepare_prefill(seqs) if is_prefill else stage_runner.prepare_decode(seqs)
        )
        temperatures = stage_runner.prepare_sample(seqs)
        logits = stage_runner.forward(input_ids, positions, is_prefill)

        bridge = stage_runner.bridge_hidden()
        for index, sequence in enumerate(seqs):
            if bridge is None:
                continue
            rows = bridge[index]
            rows = rows[-len(sequence.token_ids) :] if is_prefill else rows[-1:]
            captured = _mx_to_torch(rows)
            state = states[sequence]
            if is_prefill or state.bridge_hidden is None:
                state.bridge_hidden = captured
            else:
                state.bridge_hidden = torch.cat((state.bridge_hidden, captured), dim=0)

        current_keys = [rng_key_by_sequence[sequence] for sequence in seqs]
        sampler = _make_sampler(
            mx,
            seqs,
            [sampling_by_sequence[sequence] for sequence in seqs],
            current_keys,
        )
        sampled_ids = stage_runner.sample(logits, temperatures, sampler=sampler).tolist()
        for sequence, key in zip(seqs, current_keys, strict=True):
            rng_key_by_sequence[sequence] = key

        effective_ids = []
        for position, (sequence, sampled_id) in enumerate(zip(seqs, sampled_ids, strict=True)):
            state = states[sequence]
            if state.text_finished:
                effective = pad_token if state.emitted_filler else enter_token
                state.emitted_filler = True
            else:
                effective = sampled_id
                if sampled_id == eos_id:
                    state.text_finished = True
                    state.content_len = len(state.generated)
            effective_ids.append(effective)
            state.generated.append(effective)
            if not is_prefill:
                state.text_state = _mx_to_torch(logits[position])
            if state.request.stop_hit is not None and state.request.stop_hit(state.generated):
                sequence.max_tokens = sequence.num_completion_tokens + 1
            if (
                state.text_finished
                and state.content_len is not None
                and len(state.generated) >= state.content_len + talker_tail
            ):
                sequence.max_tokens = sequence.num_completion_tokens + 1

        if is_prefill:
            cache_sequences.extend(seqs)
        scheduler.postprocess(seqs, effective_ids, is_prefill)
        stage_runner.reset_context()
        _filter_inactive_cache_rows()

    return [
        ThinkerDecoded(
            generated=state.generated,
            bridge_hidden=state.bridge_hidden,
            text_state=state.text_state,
        )
        for state in states.values()
    ]


__all__ = [
    "ThinkerDecode",
    "ThinkerDecoded",
    "decode_minimind",
    "decode_minimind_mlx",
    "recapture_bridge_minimind",
]
