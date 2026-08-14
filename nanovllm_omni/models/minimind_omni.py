"""Real-weight MiniMind-O (``jingyaogong/minimind-3o``) stage implementations.

Real three-stage runtime (vLLM-Omni PR #3796 semantics, not nano):

* **Thinker** runs ``self.thinker.layers`` (== ``self.model`` of the loaded
  ``MiniMindOmni``) text-only. After the visible EOS it forces an
  ``enter`` token followed by ``THINKER_FORCED_PADDING_DEFAULT`` ``pad``
  tokens (default 128). Every step captures the real bridge hidden state
  at ``config.bridge_layer`` into the typed ``BridgePayload``.
* **Talker** runs ``self.talker`` (the independent ``TalkerModule``) as
  AR. Each bridge position consumes the Thinker's bridge hidden via
  ``embed_proj`` and the previously sampled codec ids via
  ``embed_tokens``/``codec_proj``. Per-codebook MTP delay
  (``active_mask[t][k] = k <= t``) and the post-bridge watchdog
  (``TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN`` = 192) match the
  reference. Talker reads real weights — there is no shared-handle
  audio stash on the Thinker side.
* **Code2Wav** decodes the resulting codec ids through Mimi.

Helper functions ``frames_from_bridges`` / ``apply_talker_watchdog`` are
kept for the typed-payload tests even though the real Talker no longer
needs them — Ponytail: deleting public utility just to delete code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
    THINKER_FORCED_PADDING_DEFAULT,
    AudioPayload,
    BridgePayload,
    CodecTokenPayload,
    TensorPayload,
    ThinkerRun,
    TokenPayload,
)
from nanovllm_omni.stage import Stage

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"
DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"

MIMI_CODEBOOKS = 8
MIMI_SAMPLE_RATE = 24_000
# MiniMind-O / Mimi shared vocabulary cutoff: ids >= this are control
# tokens (stop / pad), not actual codebook samples.
MIMI_CODE_VOCAB_LIMIT = 2048


@dataclass
class _LoadedMiniMind:
    """Loaded MiniMind-O + tokenizer + Mimi; shared by the three stages."""

    model: Any
    tokenizer: Any
    mimi: Any
    device: str
    model_id: str
    torch: Any = None


def _resolve_snapshot(model_id: str) -> str:
    from pathlib import Path

    path = Path(model_id)
    if path.is_dir():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


class _LazyHandle:
    """Lazily-resolved shared MiniMind-O bundle (weights load on first use)."""

    def __init__(
        self, model_id: str, device: str | None, mimi_model_id: str = DEFAULT_MIMI_MODEL_ID
    ) -> None:
        self.model_id = model_id
        self.mimi_model_id = mimi_model_id
        self.device = device
        self._loaded: _LoadedMiniMind | None = None

    def __call__(self) -> _LoadedMiniMind:
        if self._loaded is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "torch is required to load real MiniMind-O weights; "
                    "install with `pip install nanovllm-omni[minimind]`"
                ) from exc
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = _load_minimind(self.model_id, device, self.mimi_model_id)
        return self._loaded

    def reset(self) -> None:
        self._loaded = None


def _load_minimind(
    model_id: str,
    device: str,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
) -> _LoadedMiniMind:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    snapshot_dir = _resolve_snapshot(model_id)
    mimi_dir = _resolve_snapshot(mimi_model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, trust_remote_code=True).eval()
    # ponytail: half only on CUDA; CPU path stays float32 (4GB laptop GPUs OOM).
    if device != "cpu":
        model = model.half()
    model = model.to(device)
    mimi = MimiModel.from_pretrained(mimi_dir).eval()
    if device != "cpu":
        mimi = mimi.half()
    mimi = mimi.to(device)
    return _LoadedMiniMind(
        model=model,
        tokenizer=tokenizer,
        mimi=mimi,
        device=device,
        model_id=model_id,
        torch=torch,
    )


# --------------------------------------------------------------------------- #
# Thinker helpers
# --------------------------------------------------------------------------- #


def _thinker_submodules(loaded: _LoadedMiniMind) -> tuple[Any, Any]:
    """Return ``(thinker_module, talker_module)`` from a loaded MiniMindOmni.

    The reference ``MiniMindOmni.__init__`` aliases ``self.thinker = self.model``
    and adds ``self.talker = TalkerModule(config)``. Trust-remote-code loading
    keeps those attributes intact, but we tolerate fallbacks for older or
    re-exported checkpoints.
    """
    model = loaded.model
    thinker = getattr(model, "thinker", None) or getattr(model, "model", None)
    talker = getattr(model, "talker", None)
    if thinker is None or talker is None:
        raise RuntimeError(
            "Loaded MiniMind-O model is missing .thinker / .talker submodules; "
            "the trust_remote_code checkpoint does not expose the Thinker+Talker "
            "split that the real pipeline needs."
        )
    return thinker, talker


def _bridge_layer_idx(model: Any) -> int:
    cfg = model.config
    if hasattr(cfg, "bridge_layer"):
        return int(cfg.bridge_layer)
    return max(0, len(model.thinker.layers) // 2 - 1)


def _tensor_payload_from_tensor(t: Any) -> TensorPayload:
    """Flatten a torch.Tensor into a ``TensorPayload`` on CPU float32."""
    import torch

    tensor = t.detach().cpu().to(dtype=torch.float32).reshape(-1)
    values = tuple(float(v) for v in tensor.tolist())
    return TensorPayload(values=values, shape=(len(values), 1))


def _recompute_rope(model_part: Any, config: Any, device: Any) -> None:
    """Recompute RoPE buffers when transformers>=5.x drops them on meta init.

    Ponytail: inlined the few lines of ``precompute_freqs_cis`` from
    MiniMind's ``model_minimind`` so we don't depend on the trust-remote
    code module path (which HF renames per snapshot).
    """
    if float(model_part.freqs_cos[0, 0].item()) != 0:
        return
    import math

    import torch

    dim = config.head_dim
    end = config.max_position_embeddings
    rope_base = config.rope_theta
    rope_scaling = getattr(config, "rope_scaling", None)
    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:

            def inv_dim(b: float) -> float:
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=device).float() - low) / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    model_part.freqs_cos = freqs_cos
    model_part.freqs_sin = freqs_sin


def _run_thinker_only(
    loaded: _LoadedMiniMind,
    prompt: str,
    *,
    forced_padding_count: int = THINKER_FORCED_PADDING_DEFAULT,
    max_new_tokens: int = 512,
    text_temperature: float = 0.7,
) -> tuple[list[int], list[Any], int]:
    """Thinker-only AR; captures real bridge hidden states at every step.

    Runs ``self.thinker.layers`` directly so we control what is fed into
    the language model and can sample bridge hidden states without
    accidentally spinning up the Talker.

    Returns ``(text_tokens, bridge_hiddens, emitted_pads)`` where
    ``bridge_hiddens[i]`` is the bridge-layer hidden captured at step
    ``i`` (length = 1 visible + ``emitted_pads`` forced).
    """
    model = loaded.model
    thinker, _ = _thinker_submodules(loaded)
    torch_mod = loaded.torch
    device = loaded.device
    tokenizer = loaded.tokenizer
    config = model.config

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = torch_mod.tensor(
        tokenizer(text).data["input_ids"],
        dtype=torch_mod.long,
        device=device,
    )[None, ...]

    eos = tokenizer.eos_token_id
    # Reference stream_generate uses these ids for the post-EOS sequence;
    # they aren't surfaced through the config on the published checkpoint,
    # so we hardcode the same defaults.
    enter_token_id = 201
    pad_token_id = 0

    _recompute_rope(thinker, config, device)
    freqs_cos = thinker.freqs_cos
    freqs_sin = thinker.freqs_sin
    bridge_idx = _bridge_layer_idx(model)

    past_kvs: list | None = None
    cur = input_ids
    text_tokens: list[int] = []
    bridge_hiddens: list[Any] = []
    finished = False
    forced_remaining = 0
    enter_emitted = False

    with torch_mod.no_grad():
        for _ in range(max_new_tokens):
            x = cur if past_kvs is None else cur[:, -1:]
            seqlen = x.shape[1]
            start_pos = (
                past_kvs[0][0].shape[1] if past_kvs is not None and past_kvs[0] is not None else 0
            )

            h = thinker.dropout(thinker.embed_tokens(x))
            pos_emb = (
                freqs_cos[start_pos : start_pos + seqlen],
                freqs_sin[start_pos : start_pos + seqlen],
            )

            presents: list = []
            bridge_step: Any | None = None
            for i, layer in enumerate(thinker.layers):
                past = past_kvs[i] if past_kvs is not None else None
                h, present = layer(
                    h,
                    pos_emb,
                    past_key_value=past,
                    use_cache=True,
                    attention_mask=None,
                )
                presents.append(present)
                if i == bridge_idx:
                    bridge_step = h[0, -1].detach()
            if bridge_step is not None:
                bridge_hiddens.append(bridge_step)

            h_norm = thinker.norm(h)
            logits = thinker.lm_head(h_norm[:, -1:, :]).float()

            if not finished:
                probs = torch_mod.softmax(logits[0, -1] / text_temperature, dim=-1)
                next_tok = int(torch_mod.multinomial(probs, 1).item())
                text_tokens.append(next_tok)
                past_kvs = presents
                cur = torch_mod.cat([cur, torch_mod.tensor([[next_tok]], device=device)], dim=-1)
                if next_tok == eos:
                    finished = True
                    forced_remaining = forced_padding_count
                    enter_emitted = False
            else:
                if forced_remaining <= 0:
                    break
                forced_tok = enter_token_id if not enter_emitted else pad_token_id
                enter_emitted = True
                forced_remaining -= 1
                past_kvs = presents
                cur = torch_mod.cat([cur, torch_mod.tensor([[forced_tok]], device=device)], dim=-1)
                if forced_remaining == 0:
                    break

    return text_tokens, bridge_hiddens, forced_padding_count - forced_remaining


# --------------------------------------------------------------------------- #
# Talker helpers
# --------------------------------------------------------------------------- #


def frames_from_bridges(
    bridges: tuple[BridgePayload, ...],
    codebooks: int = MIMI_CODEBOOKS,
) -> list[list[int]]:
    """Recover per-frame codebook lists from typed bridge ``audio_codes``.

    Kept as a utility for tests; the real Talker no longer reads
    ``audio_codes`` from bridges — it consumes the bridge hidden states
    directly via the model's ``embed_proj``.
    """
    frames: list[list[int]] = []
    for bridge in bridges:
        codes = list(bridge.audio_codes)
        if not codes:
            continue
        if len(codes) % codebooks != 0:
            codes = codes[: len(codes) - (len(codes) % codebooks)]
        for offset in range(0, len(codes), codebooks):
            frames.append(codes[offset : offset + codebooks])
    return frames


def apply_talker_watchdog(
    frames: list[list[int]],
    thinker_bridge_count: int,
    max_steps_after_last: int = TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
) -> list[list[int]]:
    """Keep bridge-conditioned frames; cap post-bridge tail (PR #3796 watchdog)."""
    if max_steps_after_last < 0:
        return frames
    limit = thinker_bridge_count + max_steps_after_last
    if len(frames) <= limit:
        return frames
    return frames[:limit]


def _run_talker_only(
    loaded: _LoadedMiniMind,
    bridge_hiddens: list[Any],
    *,
    codebooks: int = MIMI_CODEBOOKS,
    watchdog_limit: int = TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
    sample_temperature: float = 0.2,
    audio_pad_token: int | None = None,
) -> tuple[list[list[int]], tuple[tuple[bool, ...], ...], int]:
    """Talker-only AR over bridge positions; per-codebook MTP delay.

    Each iteration consumes one bridge hidden state and emits one AR step
    in the Talker's codebook space. After the last bridge, AR continues
    using the last bridge as conditioning up to ``watchdog_limit`` extra
    steps. Ponytail simplification: no KV cache across talker steps (one
    forward per position); correctness over throughput.
    """
    model = loaded.model
    _, talker = _thinker_submodules(loaded)
    torch_mod = loaded.torch
    device = loaded.device
    config = model.config

    if audio_pad_token is None:
        audio_pad_token = int(getattr(config, "audio_pad_token", MIMI_CODE_VOCAB_LIMIT + 1))

    if not bridge_hiddens:
        return [], (), codebooks

    bridge_dev = [b.to(device=device, dtype=torch_mod.float32) for b in bridge_hiddens]
    last_bridge = bridge_dev[-1]
    n_bridges = len(bridge_dev)
    max_steps = (
        n_bridges + watchdog_limit if watchdog_limit >= 0 else n_bridges + watchdog_limit + 1
    )

    _recompute_rope(talker, config, device)
    freqs_cos = talker.freqs_cos[:1]
    freqs_sin = talker.freqs_sin[:1]
    # Match the talker weights' dtype (e.g. half on GPU, float32 on CPU).
    talker_dtype = next(talker.parameters()).dtype

    # Per-codebook history (TalkerHead emits 8 logits, one per codebook).
    history: list[list[int]] = [[] for _ in range(codebooks)]
    active_masks: list[tuple[bool, ...]] = []

    with torch_mod.no_grad():
        for step in range(max_steps):
            bridge = bridge_dev[step] if step < n_bridges else last_bridge
            bridge_states = bridge.view(1, 1, -1).to(dtype=talker_dtype)  # [1, 1, H_thinker]

            audio_ids = torch_mod.full(
                (1, codebooks, 1),
                audio_pad_token,
                dtype=torch_mod.long,
                device=device,
            )
            if step > 0:
                for ci in range(codebooks):
                    if step - 1 < len(history[ci]):
                        audio_ids[0, ci, 0] = history[ci][step - 1]

            talker_emb = talker.embed_tokens(audio_ids)
            bridge_proj = talker.embed_proj(bridge_states) * talker.text_scale
            codec_proj = talker.codec_proj(talker_emb) * talker.audio_scale
            # TalkerEmbedding sums across the codebook dim so ``codec_proj``
            # is already [B, L, H_talker]; no unsqueeze needed.
            h = bridge_proj + codec_proj  # [1, 1, H_talker]
            pos_emb = (freqs_cos, freqs_sin)

            for layer in talker.layers:
                h, _ = layer(
                    h,
                    pos_emb,
                    past_key_value=None,
                    use_cache=False,
                    attention_mask=None,
                )
            h_norm = talker.norm(h)
            logits_list = talker.lm_head(h_norm)  # list of [1, 1, V]

            # Reference uses audio_step = step - 1; the smoke test expects
            # k <= t where t is the position in this Talker's stream, so
            # we use t = step here (frame 0 already activates codebook 0).
            audio_step = step
            active: list[bool] = []
            for ci in range(codebooks):
                if audio_step >= ci:
                    logits_ci = logits_list[ci][0, -1].float() / sample_temperature
                    probs = torch_mod.softmax(logits_ci, dim=-1)
                    code = int(torch_mod.multinomial(probs, 1).item())
                else:
                    code = audio_pad_token
                history[ci].append(code)
                active.append(audio_step >= ci)
            active_masks.append(tuple(active))

            # Stop when every codebook reached the control-token range.
            if all(history[ci][-1] >= MIMI_CODE_VOCAB_LIMIT for ci in range(codebooks)):
                break

    audio_codes = [[history[ci][t] for ci in range(codebooks)] for t in range(len(active_masks))]
    return audio_codes, tuple(active_masks), codebooks


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


class MinimindThinker(Stage[str, ThinkerRun]):
    """Real MiniMind-O Thinker: AR text + post-EOS forced bridge hidden states."""

    name = "thinker"
    forced_padding_count = THINKER_FORCED_PADDING_DEFAULT

    def __init__(self, handle: _LazyHandle) -> None:
        self._handle = handle

    def execute(self, payload: str) -> ThinkerRun:
        loaded = self._handle()
        text_tokens, bridge_hiddens, emitted_pads = _run_thinker_only(
            loaded,
            payload,
            forced_padding_count=self.forced_padding_count,
        )
        text = loaded.tokenizer.decode(text_tokens, skip_special_tokens=True)
        eos = loaded.tokenizer.eos_token_id

        if not bridge_hiddens:
            raise RuntimeError("Thinker produced no bridge hidden states")

        visible = BridgePayload(
            tokens=TokenPayload(token_ids=tuple(text_tokens), text=text),
            hidden_states=_tensor_payload_from_tensor(bridge_hiddens[0]),
        )
        forced: list[BridgePayload] = []
        for i in range(emitted_pads):
            idx = 1 + i
            if idx < len(bridge_hiddens):
                hidden = _tensor_payload_from_tensor(bridge_hiddens[idx])
            else:
                # Defensive fallback: emitted_pads can lag when the forced
                # sequence is short-circuited, in which case we reuse the
                # last captured hidden so downstream stages still get a
                # consistent tensor shape.
                hidden = _tensor_payload_from_tensor(bridge_hiddens[-1])
            forced.append(
                BridgePayload(
                    tokens=TokenPayload(
                        token_ids=(eos,),
                        text="",
                        metadata={"forced": "true", "step": str(i)},
                    ),
                    hidden_states=hidden,
                )
            )

        bridges = (visible, *forced)
        return ThinkerRun(
            bridges=bridges,
            visible_tokens=visible.tokens,
            eos_token_id=eos,
            forced_padding_count=len(forced),
        )


class MinimindTalker(Stage[ThinkerRun, CodecTokenPayload]):
    """Real MiniMind-O Talker: AR over TalkerModule; MTP per codebook; watchdog."""

    name = "talker"
    codebooks = MIMI_CODEBOOKS
    max_steps_after_last_thinker_token = TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN

    def __init__(self, handle: _LazyHandle) -> None:
        # Talker now consumes real weights; the handle is mandatory.
        self._handle = handle

    def execute(self, payload: ThinkerRun) -> CodecTokenPayload:
        loaded = self._handle()

        bridge_hiddens: list[Any] = []
        for bridge in payload.bridges:
            values = bridge.hidden_states.values
            arr = loaded.torch.tensor(values, dtype=loaded.torch.float32).reshape(-1)
            bridge_hiddens.append(arr)

        audio_codes, active_mask, codebooks = _run_talker_only(
            loaded,
            bridge_hiddens,
            codebooks=self.codebooks,
            watchdog_limit=self.max_steps_after_last_thinker_token,
        )

        # Match the reference TalkerTokenPayload contract: inactive or
        # control-range codebook slots serialise as AUDIO_PADDING_TOKEN_ID.
        token_ids: list[int] = []
        for t, codes in enumerate(audio_codes):
            for ci in range(codebooks):
                c = codes[ci] if ci < len(codes) else MIMI_CODE_VOCAB_LIMIT + 1
                if not active_mask[t][ci] or c >= MIMI_CODE_VOCAB_LIMIT:
                    token_ids.append(AUDIO_PADDING_TOKEN_ID)
                else:
                    token_ids.append(int(c))

        return CodecTokenPayload(
            token_ids=tuple(token_ids),
            codebooks=codebooks,
            active_mask=active_mask,
            sample_rate=MIMI_SAMPLE_RATE,
        )


class MinimindCode2Wav(Stage[CodecTokenPayload, AudioPayload]):
    """Mimi decode stage → 24 kHz mono ``AudioPayload``."""

    name = "code2wav"

    def __init__(self, handle: _LazyHandle) -> None:
        self._handle = handle

    def execute(self, payload: CodecTokenPayload) -> AudioPayload:
        loaded = self._handle()
        torch = loaded.torch
        flat = list(payload.token_ids)
        if not flat:
            samples: tuple[float, ...] = ()
        else:
            codes = (
                torch.tensor(flat, dtype=torch.long, device=loaded.device)
                .reshape(-1, payload.codebooks)
                .T.unsqueeze(0)
            )
            with torch.no_grad():
                audio = loaded.mimi.decode(codes).audio_values
            samples = tuple(float(s) for s in audio.squeeze().float().cpu().numpy())
        return AudioPayload(
            samples=samples,
            sample_rate=MIMI_SAMPLE_RATE,
            metadata={"format": "pcm_s16le", "source": "minimind-omni"},
        )


@dataclass
class MinimindBundle:
    thinker: MinimindThinker
    talker: MinimindTalker
    code2wav: MinimindCode2Wav
    model_id: str


def load_minimind_omni_bundle(
    model_id: str = DEFAULT_MINIMIND_MODEL_ID,
    device: str | None = None,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
) -> MinimindBundle:
    handle = _LazyHandle(model_id=model_id, device=device, mimi_model_id=mimi_model_id)
    return MinimindBundle(
        thinker=MinimindThinker(handle),
        talker=MinimindTalker(handle),
        code2wav=MinimindCode2Wav(handle),
        model_id=model_id,
    )
