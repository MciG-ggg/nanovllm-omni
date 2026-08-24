"""Batched (continuous-batching) MiniMind-O thinker runner (Q5a/Q7/Q10a).

Project-owned batched decode loop built ON TOP of the frozen model: the
vendored ``MiniMindOmni.forward`` is called with a rectangular [B, 9, T]
input (text row 8, 8 audio-code rows) and ``past_key_values`` whose layout
is ``[B, seq, kv_heads, head_dim]`` (seq at dim 1 -- the vendored forward
reads ``start_pos = past_key_values[0][0].shape[1]``). The model only
supports ONE scalar ``start_pos`` per forward, so the scheduler groups
requests that share the same KV length (Q8a); this runner executes those
groups as one batched forward each.

Per-request sampling honours Q10a with a FIXED per-request
``torch.Generator``: batch layout never changes a request's draw order, so
each request stays bit-reproducible from its own seed regardless of how the
requests are grouped. The text token is sync'd once per row per step (the
next input depends on it -- identical cost to the single path); the 8 audio
codes are sampled on-device and moved to Python once per row.

Engine loop mirrors vllm-omni: ``schedule() -> execute(prefill+decode) ->
update_from_output``, and finished thinkers drop OUT to the serial
talker/mimi->wav chain (Q9a) via ``code2wav.decode_audio``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from nanovllm_omni.engine.sched import (
    FixedKvSlotPool,
    OmniScheduler,
    SchedulerGroup,
)

_AUDIO_VOCAB_BOUNDARY = 2048  # codes >= this are audio stop / special (MIMI_CODE_VOCAB_LIMIT)


@dataclass
class BatchedThinkerState:
    """Per-request mutable runner state (the "request state cache").

    Mirrors the single-path loop's buffers: text history, 8 audio-code
    streams, per-layer stop positions, the last-observed audio column (fed
    back into the next decode), emitted frames, and the request's own RNG.
    """

    rid: str
    prompt_ids: list[int]
    prompt_len: int
    gen: Any  # torch.Generator, seeded per request (Q10a)
    text_tokens: list[int] = field(default_factory=list)
    audio_codes: list[list[int]] = field(default_factory=lambda: [[] for _ in range(8)])
    audio_stop_pos: list[int | None] = field(default_factory=lambda: [None] * 8)
    last_audio: list[int] = field(default_factory=lambda: [0] * 8)
    frames: list[list[int]] = field(default_factory=list)

    # MiniMind text-state machine: after EOS emit enter-token once then pads.
    text_finished: bool = False
    first_finished: bool = True

    # decode calls completed (each appends one text token + 8 audio codes)
    step: int = 0


class BatchedThinkerRunner:
    """Execute scheduler groups as one rectangular forward per group.

    The runner owns per-request state + the fixed KV slot pool; the scheduler
    owns the lifecycle. KV slots are registered lazily on first prefill.
    """

    def __init__(
        self,
        bundle: Any,
        sched: OmniScheduler,
        *,
        temperature: float = 0.75,
        top_p: float = 0.90,
        rp: float = 1.0,
        max_new_tokens: int = 1024,
        eos_token_id: int = 2,
        open_thinking: bool = False,
        base_seed: int = 42,
    ) -> None:
        self.bundle = bundle
        self.sched = sched
        self.model = bundle.model
        self.sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "rp": rp,
            "max_new_tokens": max_new_tokens,
            "eos": eos_token_id,
            "open_thinking": open_thinking,
            "enter": getattr(self.model, "enter_token_id", 201),
            "pad": getattr(self.model, "pad_token_id", 0),
        }
        self.audio_pad: int = int(getattr(self.model, "audio_pad_token", 0))
        self.audio_stop: int = int(getattr(self.model, "audio_stop_token", 0))
        self.think_end_ids = list(
            getattr(getattr(self.model, "config", None), "think_end_ids", []) or []
        )
        self.base_seed = base_seed
        self.states: dict[str, BatchedThinkerState] = {}

        cfg = getattr(self.model, "config", None)
        n_thinker = len(getattr(getattr(self.model, "thinker", None), "layers", []) or [])
        n_talker = len(getattr(getattr(self.model, "talker", None), "layers", []) or [])
        self.n_layers = n_thinker + n_talker
        self.kv_pool = FixedKvSlotPool(max_seq=int(getattr(cfg, "max_position_embeddings", 4096)))
        params = list(self.model.parameters())
        self._device = params[0].device
        self._dtype = params[0].dtype

    # -- submission / registration -----------------------------------------

    def add_request(self, prompt_ids: list[int], request_id: str | None = None) -> str:
        rid = self.sched.add_request(prompt_ids, request_id=request_id)
        # stable seed from the name so admission order never changes draws
        seed = self.base_seed + int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16) % 1_000_000
        import torch

        gen = torch.Generator(device=self._device)
        gen.manual_seed(seed)
        self.states[rid] = BatchedThinkerState(
            rid=rid,
            prompt_ids=list(prompt_ids),
            prompt_len=len(prompt_ids),
            gen=gen,
            last_audio=[self.audio_pad] * 8,
        )
        return rid

    def _register_kv(self, rid: str) -> None:
        self.kv_pool.register(
            rid,
            n_layers=self.n_layers,
            n_heads=int(self._kv_heads()),
            head_dim=int(self._head_dim()),
            device=self._device,
            dtype=self._dtype,
        )

    def _kv_heads(self) -> int:
        cfg = getattr(self.model, "config", None)
        val = getattr(cfg, "num_key_value_heads", None)
        if val is None:
            val = getattr(cfg, "num_attention_heads", None)
        return val or 8

    def _head_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        return getattr(cfg, "head_dim", None) or 64

    # -- input assembly -----------------------------------------------------

    def _decode_col(self, req_ids: list[str]) -> Any:
        """Build the [B, 9, 1] decode input: 8 audio codes + 1 text token."""
        import torch

        audio = torch.tensor(
            [self.states[r].last_audio for r in req_ids], dtype=torch.long, device=self._device
        ).unsqueeze(
            2
        )  # [B, 8, 1]
        text = (
            torch.tensor(
                [self.states[r].text_tokens[-1] for r in req_ids],
                dtype=torch.long,
                device=self._device,
            )
            .unsqueeze(1)
            .unsqueeze(2)
        )  # [B, 1, 1]
        return torch.cat([audio, text], dim=1)

    # -- per-request sampling (isolated RNG, Q10a) --------------------------

    def _sample_text(self, st: BatchedThinkerState, logits_row: Any) -> int:
        import torch
        import torch.nn.functional as functional

        temp, top_p, rp = (
            self.sampling["temperature"],
            self.sampling["top_p"],
            self.sampling["rp"],
        )
        logits = logits_row.clone() / (temp + 1e-9)
        hist = st.prompt_ids + st.text_tokens
        if rp != 1.0:
            uniq = torch.unique(torch.tensor(hist, device=logits.device))
            logits[uniq] /= rp
        if top_p and top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            mask = torch.cumsum(functional.softmax(sorted_l, dim=-1), dim=-1) > top_p
            mask[1:], mask[0] = mask[:-1].clone(), False
            logits[sorted_i[mask]] = -float("Inf")
        return int(
            torch.multinomial(functional.softmax(logits, dim=-1), 1, generator=st.gen).item()
        )

    def _sample_audio_row(self, st: BatchedThinkerState, audio_logits_row: Any) -> list[int]:
        """Port of generation.py ``_sample_audio_codes`` for one request row."""
        import torch
        import torch.nn.functional as functional

        active = [i for i in range(8) if st.step - 1 >= i]
        codes = [self.audio_pad] * 8
        for layer in active:
            logits_i = audio_logits_row[layer].clone() / 0.2
            for prev in st.audio_codes[layer][-3:]:
                logits_i[prev] /= 1.05
            top_v, top_i = logits_i.topk(50)
            idx = int(
                torch.multinomial(functional.softmax(top_v, dim=-1), 1, generator=st.gen).item()
            )
            codes[layer] = int(top_i[idx])
        return codes

    # -- group execution ----------------------------------------------------

    def prefill_group(self, group: SchedulerGroup) -> None:
        """One rectangular [B, 9, P] forward over the group's prompts."""
        import torch

        req_ids = group.req_ids
        n_req = len(req_ids)
        n_pos = group.start_pos
        text = torch.tensor(
            [self.states[r].prompt_ids for r in req_ids],
            dtype=torch.long,
            device=self._device,
        )
        audio = torch.full((n_req, 8, n_pos), self.audio_pad, dtype=torch.long, device=self._device)
        inp = torch.cat([audio, text.unsqueeze(1)], dim=1)  # [B, 9, P]
        for r in req_ids:
            self._register_kv(r)
        out = self.model.forward(inp, past_key_values=None, use_cache=True, logits_to_keep=1)
        for layer, (k, v) in enumerate(out.past_key_values):
            for r, rid in enumerate(req_ids):
                self.kv_pool.write(rid, layer, k, v, row=r)

        # first token predicted by the prompt's last position; no audio yet
        text_logits = out.logits  # [B, 1, V]
        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            tok = self._sample_text(st, text_logits[r, 0, :])
            if st.text_finished:  # unreachable on first token, kept for symmetry
                tok = self._enter_or_pad(st)
            st.text_tokens.append(tok)
            self._note_audio(st, [self.audio_pad] * 8)
            if not st.text_finished and tok == self.sampling["eos"]:
                st.text_finished = True

    def decode_group(self, group: SchedulerGroup) -> None:
        """One rectangular [B, 9, 1] forward for a same-KV-length decode group."""

        req_ids = group.req_ids
        inp = self._decode_col(req_ids)
        past = self.kv_pool.gather(req_ids)
        out = self.model.forward(inp, past_key_values=past, use_cache=True, logits_to_keep=1)
        for layer, (k, v) in enumerate(out.past_key_values):
            for r, rid in enumerate(req_ids):
                self.kv_pool.write(rid, layer, k, v, row=r)

        text_logits = out.logits  # [B, 1, V]
        audio_logits = out.audio_logits  # list of 8 × [B, 1, V]
        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            # generation.py order: sample, then override with enter/pad when
            # text already finished, then append, then flip finished on EOS.
            tok = self._sample_text(st, text_logits[r, 0, :])
            if st.text_finished:
                tok = self._enter_or_pad(st)
            st.text_tokens.append(tok)
            st.step += 1
            codes = self._sample_audio_row(st, [al[r, 0, :] for al in audio_logits])
            self._note_audio(st, codes)
            if not st.text_finished and tok == self.sampling["eos"]:
                st.text_finished = True

    def _enter_or_pad(self, st: BatchedThinkerState) -> int:
        tok = self.sampling["enter"] if st.first_finished else self.sampling["pad"]
        st.first_finished = False
        return tok

    def _note_audio(self, st: BatchedThinkerState, codes: list[int]) -> None:
        """Append one step of audio codes, track stop positions, emit frames.

        Port of generation.py's exact indexing: ``step`` here is ``st.step``
        after the increment, matching ``step = current_len - start_pos`` there;
        frame index is ``step - 7 + i`` and the gate is ``active >= 8``.
        """
        for i, code in enumerate(codes):
            st.audio_codes[i].append(code)
            if code >= _AUDIO_VOCAB_BOUNDARY and st.audio_stop_pos[i] is None:
                st.audio_stop_pos[i] = len(st.audio_codes[i]) - 1
        st.last_audio = codes

        audio_step = st.step - 1
        if audio_step < 7:
            return
        active = sum(
            1
            for i in range(8)
            if st.audio_stop_pos[i] is None or st.step - 7 + i < st.audio_stop_pos[i]
        )
        if active >= 8:
            frame = [st.audio_codes[i][st.step - 7 + i] for i in range(8)]
            st.frames.append(frame)

    # -- termination --------------------------------------------------------

    def step_finished(self, rid: str) -> bool:
        """MiniMind thinker termination: text finished AND last layer hit stop."""
        st = self.states[rid]
        if st.step >= self.sampling["max_new_tokens"]:
            return True
        return bool(st.text_finished and st.audio_codes[7][-1] == self.audio_stop)


def run_batched_generate(
    bundle: Any,
    prompts: list[str],
    *,
    temperature: float = 0.75,
    top_p: float = 0.90,
    rp: float = 1.0,
    max_new_tokens: int = 1024,
    open_thinking: bool = False,
    max_batch: int = 2,
    base_seed: int = 42,
) -> list[Any]:
    """Continuous-batching entry point: tokenize -> schedule -> batched thinker -> WAV.

    The engine loop mirrors vllm-omni (Q8a/Q9a):

        while scheduler.has_requests():
            out = scheduler.schedule()          # prefill + decode groups
            for prefill group: runner.prefill_group(group)
            for decode  group: runner.decode_group(group)
            scheduler.update_from_output(...)   # finish -> serial WAV chain (Q9a)

    Returns one :class:`AudioPayload` per prompt, in submission order.
    """
    import torch

    from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav
    from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate
    from nanovllm_omni.outputs import AudioPayload

    cfg = getattr(bundle.model, "config", None)
    max_seq = int(getattr(cfg, "max_position_embeddings", 4096))
    sched = OmniScheduler(max_batch=max_batch, max_seq=max_seq)
    runner = BatchedThinkerRunner(
        bundle,
        sched,
        temperature=temperature,
        top_p=top_p,
        rp=rp,
        max_new_tokens=max_new_tokens,
        open_thinking=open_thinking,
        base_seed=base_seed,
    )

    order: list[str] = []
    for prompt in prompts:
        ids = tokenize_for_generate(bundle.tokenizer, prompt, open_thinking)
        rid = runner.add_request(ids[0].tolist())
        order.append(rid)

    payloads: dict[str, Any] = {}
    with torch.no_grad():
        while sched.has_requests():
            out = sched.schedule()
            if out.is_empty:
                break
            prefilled: set[str] = set()
            generated: dict[str, int] = {}
            finished: set[str] = set()
            for group in out.prefill_groups:
                runner.prefill_group(group)
                prefilled.update(group.req_ids)
            for group in out.decode_groups:
                runner.decode_group(group)
                for rid in group.req_ids:
                    generated[rid] = runner.states[rid].step
                    if runner.step_finished(rid):
                        finished.add(rid)
            newly = sched.update_from_output(
                prefilled=prefilled, generated=generated, finished=finished
            )
            for rid in newly:
                st = runner.states[rid]
                if not st.frames:
                    payloads[rid] = AudioPayload(data=b"", sample_rate=24_000)
                    continue
                samples = decode_audio(bundle.mimi, st.frames, bundle.device)
                wav = encode_wav(samples, sample_rate=24_000)
                payloads[rid] = AudioPayload(data=wav, sample_rate=24_000)

    return [payloads[rid] for rid in order]


__all__ = ["BatchedThinkerRunner", "BatchedThinkerState", "run_batched_generate"]
