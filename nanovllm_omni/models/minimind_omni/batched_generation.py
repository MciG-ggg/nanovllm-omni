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

import contextlib
import hashlib
from dataclasses import dataclass, field
from typing import Any

from nanovllm_omni.engine.kv_pool import FixedKvSlotPool
from nanovllm_omni.engine.runtime_scheduler import (
    RuntimeGroup,
    RuntimeScheduler,
)
from nanovllm_omni.engine.sequence import PrefillChunk, Sequence

from ._sampling import (
    AUDIO_VOCAB_BOUNDARY,
    NUM_AUDIO_LAYERS,
    sample_one_audio_layer,
    sample_text_token,
)


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
    post_eos_started: bool = False
    post_eos_remaining: int = 0
    internal_stop_emitted: bool = False

    # Decode calls completed (each appends one text token + 8 audio codes).
    # The prefill prediction is also a generated token, but does not advance
    # ``step`` because audio alignment starts at the first decode call.
    step: int = 0

    # open_thinking audio gating: when the trailing text tokens match
    # ``runner.think_end_ids``, audio is suppressed until 2 decode steps
    # later (matches generation.py L130-137). ``None`` until detection.
    think_end_step: int | None = None

    # Engine-native audio input (Q4): fbank + frame length for this request,
    # passed to the model's ``forward(audio_inputs=..., audio_lens=...)``
    # at prefill (``start_pos == 0``); the model injects the embeddings at
    # ``<|audio_pad|>`` markers inside ``MiniMindOmni.forward``.
    audio_inputs: Any = None
    audio_lens: Any = None

    # Bridge hidden states: one ``[hidden_size]`` tensor
    # per step the runner completes (prefill = 1 entry for the last prompt
    # position; each decode = 1 entry). Populated only when the runner was
    # constructed with ``capture_bridge_states=True``. The talker stage
    # consumes this list via ``extract_bridge_states(state)``.
    bridge_states: list[Any] = field(default_factory=list)


class BatchedThinkerRunner:
    """Execute scheduler groups as one rectangular forward per group.

    The runner owns per-request state + the fixed KV slot pool; the scheduler
    owns the lifecycle. KV slots are registered lazily on first prefill.
    """

    def __init__(
        self,
        bundle: Any,
        sched: RuntimeScheduler,
        *,
        temperature: float = 0.75,
        top_p: float = 0.90,
        rp: float = 1.0,
        max_new_tokens: int = 1024,
        eos_token_id: int = 2,
        open_thinking: bool = False,
        base_seed: int = 42,
        kv_max_sequence_len: int | None = None,
        capture_bridge_states: bool = False,
        post_eos_padding_count: int = 0,
        internal_stop_token_id: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.sched = sched
        self.model = bundle.model
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError(f"max_new_tokens must be an integer, got {max_new_tokens!r}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if isinstance(post_eos_padding_count, bool) or not isinstance(post_eos_padding_count, int):
            raise ValueError(
                "post_eos_padding_count must be an integer, " f"got {post_eos_padding_count!r}"
            )
        if post_eos_padding_count < 0:
            raise ValueError("post_eos_padding_count must be >= 0")
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
        self.post_eos_padding_count = int(post_eos_padding_count)
        self.internal_stop_token_id = (
            internal_stop_token_id
            if internal_stop_token_id is not None
            else getattr(self.model, "internal_stop_token_id", 17)
        )
        if isinstance(self.internal_stop_token_id, bool) or not isinstance(
            self.internal_stop_token_id, int
        ):
            raise ValueError(
                "internal_stop_token_id must be an integer, " f"got {self.internal_stop_token_id!r}"
            )
        self.audio_pad: int = int(getattr(self.model, "audio_pad_token", 0))
        self.audio_stop: int = int(getattr(self.model, "audio_stop_token", 0))
        # Bridge hidden-state capture. Off by default so
        # existing benches/tests that don't need talker handoff pay zero
        # overhead. When on, ``prefill_group`` and ``decode_group`` stash
        # the thinker's bridge-layer hidden state into
        # ``state.bridge_states`` after each forward.
        self.capture_bridge_states: bool = bool(capture_bridge_states)
        self._bridge_capture_layer = _resolve_bridge_layer(self.model)
        if self.capture_bridge_states:
            enable_bridge_capture(self.model, self._bridge_capture_layer)
        self.think_end_ids = list(
            getattr(getattr(self.model, "config", None), "think_end_ids", []) or []
        )
        self.open_thinking = open_thinking  # hot-path access; mirrors sampling["open_thinking"]
        self.base_seed = base_seed
        self.states: dict[str, BatchedThinkerState] = {}

        config = getattr(self.model, "config", None)
        num_thinker_layers = len(getattr(getattr(self.model, "thinker", None), "layers", []) or [])
        num_talker_layers = len(getattr(getattr(self.model, "talker", None), "layers", []) or [])
        self.num_layers = num_thinker_layers + num_talker_layers
        # Fixed-slot KV pool. Budget the slot length explicitly: the model's
        # ``max_position_embeddings`` (MiniMind-O: 32768) is far larger than any
        # practical single generation, and a full-size slot per request OOMs a
        # 4 GB card at max_batch>=2 (fixed-slot teaching shape; paged
        # KV is the deferred upgrade this knob approximates).
        max_embeddings = int(getattr(config, "max_position_embeddings", 4096))
        self.kv_max_sequence_len = kv_max_sequence_len or min(max_embeddings, 1024 + max_new_tokens)
        self.kv_pool = FixedKvSlotPool(max_sequence_len=self.kv_max_sequence_len)
        params = list(self.model.parameters())
        self._device = params[0].device
        self._dtype = params[0].dtype

    # -- submission / registration -----------------------------------------

    def add_request(
        self,
        prompt_ids: list[int],
        request_id: str | None = None,
        audio_inputs: Any = None,
        audio_lens: Any = None,
    ) -> str:
        # Build the per-stage Sequence first, then submit it to the scheduler. ``request_id`` defaults to a
        # scheduler-assigned id; we read it back via ``seq.request_id``
        # so callers can index ``self.states`` by the same key.
        rid = request_id or f"req-{len(self.sched.running) + len(self.sched.waiting)}"
        sequence = Sequence(
            request_id=rid,
            token_ids=list(prompt_ids),
            num_tokens=len(prompt_ids),
        )
        self.sched.add_sequence(sequence)
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
            audio_inputs=audio_inputs,
            audio_lens=audio_lens,
        )
        return rid

    def _register_kv(self, rid: str) -> None:
        self.kv_pool.register(
            rid,
            num_layers=self.num_layers,
            num_heads=int(self._kv_heads()),
            head_dim=int(self._head_dim()),
            device=self._device,
            dtype=self._dtype,
        )

    def _kv_heads(self) -> int:
        config = getattr(self.model, "config", None)
        val = getattr(config, "num_key_value_heads", None)
        if val is None:
            val = getattr(config, "num_attention_heads", None)
        return val or 8

    def _head_dim(self) -> int:
        config = getattr(self.model, "config", None)
        return getattr(config, "head_dim", None) or 64

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
        return sample_text_token(
            logits_row,
            history_ids=st.prompt_ids + st.text_tokens,
            temperature=self.sampling["temperature"],
            top_p=self.sampling["top_p"],
            rp=self.sampling["rp"],
            gen=st.gen,
        )

    def _sample_audio_row(self, st: BatchedThinkerState, audio_logits_row: Any) -> list[int]:
        """Port of generation.py ``_sample_audio_codes`` for one request row."""
        audio_step = self._compute_audio_step(st)
        active = [i for i in range(NUM_AUDIO_LAYERS) if audio_step >= i]
        codes = [self.audio_pad] * NUM_AUDIO_LAYERS
        for layer in active:
            codes[layer] = sample_one_audio_layer(
                audio_logits_row[layer],
                st.audio_codes[layer],
                gen=st.gen,
            )
        return codes

    def _compute_audio_step(self, st: BatchedThinkerState) -> int:
        """Audio column index for active-layer gating.

        Without ``open_thinking``: ``audio_step = st.step - 1`` (audio lags
        text by one position, matches generation.py L129).

        With ``open_thinking``: ``audio_step = -1`` until the trailing text
        tokens match ``think_end_ids``; once detected at step ``D``, audio
        starts at ``D + 2`` (matches generation.py L130-137).
        """
        base = st.step - 1
        if not self.open_thinking or not self.think_end_ids:
            return base
        if st.think_end_step is None:
            return -1
        return base - st.think_end_step

    # -- group execution ----------------------------------------------------

    def _audio_prefill_kwargs(self, req_ids: list[str]) -> dict[str, Any]:
        """Gather per-request audio into one batched ``audio_inputs`` tensor.

        Returns ``{}`` when no request in the group carries audio. Rows for
        audio-free requests are zero-filled; the model's ``encode_audio_inputs``
        batch-mask drops them (``audio_inputs.flatten(1).any(1)``), so mixed
        groups stay supported.
        """
        import torch

        here = [r for r in req_ids if self.states[r].audio_inputs is not None]
        if not here:
            return {}
        t_max = max(int(self.states[r].audio_inputs.shape[1]) for r in here)
        feat = self.states[here[0]].audio_inputs
        dtype, device = feat.dtype, feat.device
        n_freq = feat.shape[2]
        frames: list[Any] = []
        lens: list[int] = []
        for r in req_ids:
            st = self.states[r]
            if st.audio_inputs is not None:
                at = st.audio_inputs
                if at.shape[1] < t_max:
                    pad = torch.zeros(
                        (at.shape[0], t_max - at.shape[1], at.shape[2]),
                        dtype=dtype,
                        device=device,
                    )
                    at = torch.cat([at, pad], dim=1)
                frames.append(at)
                lens.append(int(st.audio_lens.reshape(-1)[0]))
            else:
                frames.append(torch.zeros((1, t_max, n_freq), dtype=dtype, device=device))
                lens.append(1)
        return {
            "audio_inputs": torch.cat(frames, dim=0),
            "audio_lens": torch.tensor(lens, dtype=torch.long, device=device),
        }

    def prefill_group(self, group: RuntimeGroup) -> None:
        """One rectangular [B, 9, P] forward over the group's prompt chunks."""
        import torch

        # RuntimeGroup.items is list[PrefillChunk] for a prefill group; the
        # rid is on each chunk's ``seq.request_id`` (Sequence per-stage).
        chunks: list[PrefillChunk] = group.items
        req_ids = [chunk.sequence.request_id for chunk in chunks]
        num_requests = len(req_ids)
        num_positions = chunks[0].end  # all chunks share end == seq.num_tokens
        text = torch.tensor(
            [self.states[r].prompt_ids for r in req_ids],
            dtype=torch.long,
            device=self._device,
        )
        audio = torch.full(
            (num_requests, 8, num_positions), self.audio_pad, dtype=torch.long, device=self._device
        )
        inp = torch.cat([audio, text.unsqueeze(1)], dim=1)  # [B, 9, P]
        for r in req_ids:
            self._register_kv(r)
        out = self.model.forward(
            inp,
            past_key_values=None,
            use_cache=True,
            logits_to_keep=1,
            **self._audio_prefill_kwargs(req_ids),
        )
        for layer, (k, v) in enumerate(out.past_key_values):
            for r, rid in enumerate(req_ids):
                self.kv_pool.write(rid, layer, key=k, value=v, row=r)

        # Bridge capture: stash the thinker's bridge-layer hidden state at
        # the LAST prompt position for each request (the position that
        # produced the predicted next-text token). Phase 1 stores only the
        # final-prompt-position row; multi-step bridge history is added when
        # the talker MTP path becomes a real per-stage module (Phase 3).
        if self.capture_bridge_states:
            self._capture_prefill_bridge(req_ids)

        # first token predicted by the prompt's last position; no audio yet
        text_logits = out.logits  # [B, 1, V]
        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            token = self._sample_text(st, text_logits[r, 0, :])
            if st.text_finished:  # unreachable on first token, kept for symmetry
                token = self._next_post_eos_token(st)
            st.text_tokens.append(token)
            self._note_audio(st, [self.audio_pad] * 8)
            if not st.text_finished and token == self.sampling["eos"]:
                st.text_finished = True

    def decode_group(self, group: RuntimeGroup) -> None:
        """One rectangular [B, 9, 1] forward for a same-KV-length decode group."""
        # RuntimeGroup.items is list[Sequence] for a decode group; rid is on
        # each Sequence.request_id.
        req_ids = [sequence.request_id for sequence in group.items]
        inp = self._decode_col(req_ids)
        past = self.kv_pool.gather(req_ids)
        out = self.model.forward(inp, past_key_values=past, use_cache=True, logits_to_keep=1)
        for layer, (k, v) in enumerate(out.past_key_values):
            for r, rid in enumerate(req_ids):
                self.kv_pool.write(rid, layer, key=k, value=v, row=r)

        text_logits = out.logits  # [B, 1, V]
        audio_logits = out.audio_logits  # list of 8 × [B, 1, V]

        # Bridge capture: stash the thinker's bridge-layer hidden state for
        # each request at this decode step (position 0 of the [B, 1, hidden]
        # captured tensor). One entry per step per request.
        if self.capture_bridge_states:
            self._capture_decode_bridge(req_ids)

        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            # generation.py order: sample, then override with enter/pad when
            # text already finished, then append, then flip finished on EOS.
            token = self._sample_text(st, text_logits[r, 0, :])
            if st.text_finished:
                token = self._next_post_eos_token(st)
            st.text_tokens.append(token)
            # Detect think_end (open_thinking audio gating). Detection must
            # happen before ``st.step += 1`` so the +2 offset uses the current
            # pre-increment step value, matching generation.py L135.
            if (
                self.open_thinking
                and self.think_end_ids
                and st.think_end_step is None
                and len(st.text_tokens) >= len(self.think_end_ids)
                and st.text_tokens[-len(self.think_end_ids) :] == list(self.think_end_ids)
            ):
                st.think_end_step = st.step + 2
            st.step += 1
            codes = self._sample_audio_row(st, [al[r, 0, :] for al in audio_logits])
            self._note_audio(st, codes)
            if not st.text_finished and token == self.sampling["eos"]:
                st.text_finished = True

    def _next_post_eos_token(self, st: BatchedThinkerState) -> int:
        """Return the next token after EOS for the selected pipeline mode.

        ``post_eos_padding_count == 0`` preserves the collapsed pipeline's
        legacy enter/pad behavior. Full stage execution opts into the
        reference sequence: enter, a bounded PAD tail, then internal stop.
        """
        if self.post_eos_padding_count == 0:
            token = self.sampling["enter"] if st.first_finished else self.sampling["pad"]
            st.first_finished = False
            return token
        if not st.post_eos_started:
            st.post_eos_started = True
            st.post_eos_remaining = self.post_eos_padding_count
            return self.sampling["enter"]
        if st.post_eos_remaining > 0:
            st.post_eos_remaining -= 1
            return self.sampling["pad"]
        st.internal_stop_emitted = True
        return self.internal_stop_token_id

    def _note_audio(self, st: BatchedThinkerState, codes: list[int]) -> None:
        """Append one step of audio codes, track stop positions, emit frames.

        Port of generation.py's exact indexing: ``step`` here is ``st.step``
        after the increment, matching ``step = current_len - start_pos`` there;
        frame index is ``step - 7 + i`` and the gate is ``active >= 8``.

        CRITICAL: ``audio_stop_pos`` is only recorded for layers that were
        actually sampled this step (``i in sampled`` in generation.py). The pad
        token is ``audio_pad >= AUDIO_VOCAB_BOUNDARY``, so applying the ``>=``
        test to inactive (pad) rows would mark every layer stopped at step 0
        and starve the ``active >= 8`` frame gate.
        """
        # which layers are being sampled this step = active set (i <= audio_step)
        active_here = {i for i in range(8) if st.step - 1 >= i}
        for i, code in enumerate(codes):
            st.audio_codes[i].append(code)
            if i in active_here and code >= AUDIO_VOCAB_BOUNDARY and st.audio_stop_pos[i] is None:
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
        if len(st.text_tokens) >= self.sampling["max_new_tokens"]:
            return True
        if self.post_eos_padding_count > 0:
            return st.internal_stop_emitted
        return bool(st.text_finished and st.audio_codes[7][-1] == self.audio_stop)

    # -- bridge capture ---------------------------------

    def _capture_prefill_bridge(self, req_ids: list[str]) -> None:
        """Stash per-request last-position bridge state after prefill."""
        captured = _read_bridge_capture(self.model, self._bridge_capture_layer)
        if captured is None:
            return  # capture not installed / layer not monkey-patched
        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            # captured shape: [B, num_positions, hidden_size]. Take the last
            # position (the one whose next-text token was predicted).
            per_request = captured[r, -1, :].detach()
            st.bridge_states.append(per_request)

    def _capture_decode_bridge(self, req_ids: list[str]) -> None:
        """Stash per-request bridge state for each decode step."""
        captured = _read_bridge_capture(self.model, self._bridge_capture_layer)
        if captured is None:
            return
        for r, rid in enumerate(req_ids):
            st = self.states[rid]
            # captured shape: [B, 1, hidden_size]. Squeeze to [hidden_size].
            per_request = captured[r, 0, :].detach()
            st.bridge_states.append(per_request)


__all__ = ["BatchedThinkerRunner", "BatchedThinkerState"]


# ---------------------------------------------------------------------------
# Bridge-state capture helpers
# ---------------------------------------------------------------------------

_BRIDGE_CAPTURE_ATTR = "_bridge_capture"
_BRIDGE_PATCH_MARKER = "_nanovllm_bridge_patched"


def _resolve_bridge_layer(model: Any) -> int:
    """Pick the thinker's bridge-layer index for ``model``.

    Defaults to ``num_hidden_layers // 2 - 1`` (matches vendored
    ``OmniConfig.bridge_layer`` and vllm-omni's MiniMindModel forward
    loop). Returns -1 when the model exposes neither ``num_hidden_layers``
    nor ``bridge_layer`` -- the caller treats -1 as "do not capture".
    """
    config = getattr(model, "config", None)
    configs = [config, getattr(config, "text_config", None), getattr(model, "thinker", None)]
    thinker_config = getattr(getattr(model, "thinker", None), "config", None)
    configs.append(thinker_config)
    for candidate in configs:
        explicit = getattr(candidate, "bridge_layer", None)
        if explicit is not None:
            return int(explicit)
    for candidate in configs:
        num_hidden = int(getattr(candidate, "num_hidden_layers", 0) or 0)
        if num_hidden > 0:
            return num_hidden // 2 - 1
    return -1


def enable_bridge_capture(model: Any, bridge_layer: int | None = None) -> int:
    """Monkey-patch ``model.thinker.layers[bridge_layer].forward`` to capture its output.

    The vendored joint model (``MiniMindOmni.forward``) routes through the
    thinker's ``MiniMindBlock`` instances and reads
    ``hidden_states[bridge_layer]`` as the talker's bridge hidden state
    *after* the layer's MLP+attention runs. We wrap the bridge layer's
    ``forward`` so its post-MLP output is stashed in
    ``block._bridge_capture``; the joint forward is otherwise unchanged.

    Returns the patched layer's index (or -1 when no patch was applied).
    Idempotent: re-patching the same layer is a no-op.

    # Monkey-patching the joint model's bridge layer avoids the
    # alternative of running the thinker alone (which would duplicate work
    # every step) at the cost of coupling to the vendored block signature.
    # If the vendored model ever changes block signature, this patch fails
    # loud (the forward call inside the wrapper will raise), keeping the
    # regression visible at first run instead of producing stale bridge
    # states silently.
    """
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        return -1
    if bridge_layer is None or bridge_layer < 0:
        bridge_layer = _resolve_bridge_layer(model)
    if bridge_layer < 0:
        return -1
    layers = getattr(thinker, "layers", None)
    if layers is None or bridge_layer >= len(layers):
        return -1
    block = layers[bridge_layer]
    if getattr(block, _BRIDGE_PATCH_MARKER, False):
        return bridge_layer

    original_forward = block.forward

    def _wrapped_forward(self, hidden_states, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = original_forward(hidden_states, *args, **kwargs)
        # The vendored HF ``MiniMindBlock.forward`` returns
        # ``(hidden_states, present_key_value)``. Capture the post-MLP
        # hidden state as the bridge hidden state. Test doubles that
        # return a single tensor are also accepted.
        captured = out[0] if isinstance(out, tuple) else out
        with contextlib.suppress(Exception):
            self._bridge_capture = captured
        return out

    import types

    block.forward = types.MethodType(_wrapped_forward, block)
    setattr(block, _BRIDGE_PATCH_MARKER, True)
    return bridge_layer


def _read_bridge_capture(model: Any, bridge_layer: int) -> Any:
    """Read the most recent ``_bridge_capture`` from the bridge layer.

    Returns ``None`` when capture is not installed (so callers can no-op).
    The returned tensor is the raw captured tensor (shape ``[B, T, hidden]``);
    per-request slicing happens in the runner.
    """
    if bridge_layer < 0:
        return None
    thinker = getattr(model, "thinker", None)
    layers = getattr(thinker, "layers", None) if thinker is not None else None
    if layers is None or bridge_layer >= len(layers):
        return None
    return getattr(layers[bridge_layer], _BRIDGE_CAPTURE_ATTR, None)


def extract_bridge_states(state: BatchedThinkerState) -> Any:
    """Stack the per-step bridge hidden states for one request.

    Returns a ``[num_steps, hidden_size]`` tensor (CPU clone so the
    caller can mutate without affecting the runner's buffer). Returns
    an empty ``[0, hidden_size]`` tensor when the runner was not
    constructed with ``capture_bridge_states=True``; consumers must
    guard against the empty shape or enable capture explicitly.
    """
    import torch

    if not state.bridge_states:
        return torch.zeros((0, 0), dtype=torch.float32)
    hidden_size = int(state.bridge_states[0].shape[-1])
    stacked = torch.stack(
        [t.detach().to(device="cpu", dtype=torch.float32) for t in state.bridge_states],
        dim=0,
    )
    if stacked.numel() == 0:
        return torch.zeros((0, hidden_size), dtype=torch.float32)
    return stacked  # [num_steps, hidden_size]


__all__ = [
    "BatchedThinkerRunner",
    "BatchedThinkerState",
    "enable_bridge_capture",
    "extract_bridge_states",
]
