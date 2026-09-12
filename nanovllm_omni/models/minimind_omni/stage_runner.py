"""MiniMind-O decode recipe on top of the unified fork-AR engine.

The fork-attention lifecycle (``set_context`` / ``forward`` / ``sample``
/ ``reset_context``) lives in ``engine/stage_runner.py::StageRunner`` —
shared by any family that uses fork attention. This module is the
MiniMind-O specific decode dance: vendor ``stream_generate`` prefill +
AR loop with the post-EOS filler clocking, plus the one-shot full-
sequence prefill that replaces the paged-KV bridge so the talker
sees bit-equal rows.

We compose on top of ``StageRunner`` rather than subclassing, so the
engine has one cohesive AR adapter and model families stay
side-effect free at import time. The recipe is two top-level
functions — ``decode_minimind`` and ``recapture_bridge_minimind`` —
that take a ``StageRunner`` and the per-call inputs.
"""

from __future__ import annotations

from typing import Any


def decode_minimind(
    stage_runner: Any,
    token_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float = 1.05,
    top_p: float = 1.0,
    eos_id: int | None = None,
    enter_token: int = 0,
    pad_token: int = 0,
) -> tuple[list[int], Any, Any]:
    """Run the vendor decode dance on a ``StageRunner``.

    Owns the sampler wrap/restore pair (the one place that touches
    ``stage_runner.model_runner.sampler``), the bridge-row accumulation,
    and the post-EOS filler clocking. Returns
    ``(generated, bridge, text_state)``.

    Args:
        stage_runner: A ``StageRunner`` from ``engine.stage_runner``.
        token_ids: Prompt token ids to prefill.
        max_tokens: Hard cap on tokens generated.
        temperature: Sampler temperature.
        repetition_penalty: Per-call RP for the wrap sampler.
        top_p: Per-call top-p for the wrap sampler.
        eos_id: Text-EOS marker; defaults to ``stage_runner.config.eos``.
        enter_token: Token emitted on the first post-EOS step (talker
            tail needs a stable anchor).
        pad_token: Token emitted on subsequent post-EOS steps.
    """
    import torch
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

    config = stage_runner.config
    fork_sp = ForkSamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)
    scheduler = Scheduler(config)
    sequence = Sequence(token_ids, fork_sp)
    scheduler.add(sequence)

    model = stage_runner.model_runner.model
    runner = stage_runner.model_runner
    _base_sampler = runner.sampler

    def _sampler_with_rp(logits, temperatures):
        # Match vendor stream_generate: divide the last-token logits,
        # apply the full-history penalty and top-p filter, then call
        # torch.multinomial directly. Gumbel-max is distributionally
        # equivalent but consumes a different RNG path; a different
        # text token changes every later bridge/audio-buffer row.
        logits_i = logits[0].clone() / (temperatures[0] + 1e-9)
        for token in set(sequence.token_ids):
            logits_i[token] /= repetition_penalty
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
            remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits_i[sorted_indices[remove]] = -float("inf")
        return torch.multinomial(torch.softmax(logits_i, dim=-1), 1).view(-1)

    runner.sampler = _sampler_with_rp
    generated: list[int] = []
    bridge_hidden: Any = None
    logits: Any = None
    # Vendor ``stream_generate`` does NOT end the loop at text EOS:
    # it sets ``text_finished`` and keeps clocking the forward pass
    # with throwaway filler so the talker tail can drain. The sampler
    # still runs on those steps so the RNG stream advances identically.
    text_finished = False
    first_finished = True
    if eos_id is None:
        eos_id = config.eos

    try:
        while not scheduler.is_finished():
            seqs, is_prefill = scheduler.schedule()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            input_ids, positions = (
                runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
            )
            temperatures = runner.prepare_sample(seqs)
            logits = runner.run_model(input_ids, positions, is_prefill)

            # Decode steps only see the last token, so only the last
            # bridge row is new. Prefill covers the prompt; each decode
            # step appends exactly one row.
            bh = model.get_bridge_hidden()
            if bh is not None:
                bh = bh.detach().clone()
                if is_prefill:
                    bridge_hidden = bh
                elif bridge_hidden is not None:
                    bridge_hidden = torch.cat([bridge_hidden, bh[-1:]], dim=0)

            token_id_list = runner.sampler(logits, temperatures).tolist()
            sampled_id = token_id_list[0]
            if text_finished:
                effective_id = enter_token if first_finished else pad_token
                first_finished = False
                token_id_list = [effective_id]
            else:
                effective_id = sampled_id
            scheduler.postprocess(seqs, token_id_list, is_prefill)
            stage_runner.reset_context()

            generated.append(effective_id)
            if not text_finished and sampled_id == eos_id:
                text_finished = True

            if sequence.num_completion_tokens >= max_tokens:
                break
    finally:
        # Restore so a second call doesn't wrap an already-wrapped
        # callable (which would explode on signature mismatch).
        runner.sampler = _base_sampler

    text_state = logits[0].detach() if logits is not None else None
    if bridge_hidden is not None:
        bridge_hidden = bridge_hidden.clone()
    return generated, bridge_hidden, text_state


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
    model = stage_runner.model_runner.model
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
        _full_bridge = model.get_bridge_hidden()
        if _full_bridge is not None:
            bridge_hidden = _full_bridge.detach().to(dev).clone()
    finally:
        stage_runner.reset_context()
    return bridge_hidden


__all__ = ["decode_minimind", "recapture_bridge_minimind"]
