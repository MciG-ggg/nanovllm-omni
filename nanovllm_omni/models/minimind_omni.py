"""Real-weight MiniMind-O (``jingyaogong/minimind-3o``) stage implementations.

Implements the three logical stages (Thinker, Talker, Code2Wav) for the
real MiniMind-O weights while preserving the existing public pipeline
seam.

Nano middle-form (issue #11 / vLLM-Omni PR #3796 semantics, not runtime):
Thinker emits typed post-EOS bridges (default 128 pads); Talker consumes
``ThinkerRun.bridges`` only (no shared-handle audio stash) and applies a
tail watchdog after the last Thinker bridge.
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

MIMI_AUDIO_PAD_TOKEN = 2049
MIMI_CODEBOOKS = 8
MIMI_SAMPLE_RATE = 24_000


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


def _run_generation(loaded: _LoadedMiniMind, prompt: str) -> tuple[list[int], list[list[int]]]:
    """Drive MiniMind-O ``generate`` and collect text tokens + audio frames."""
    torch = loaded.torch

    messages = [{"role": "user", "content": prompt}]
    text = loaded.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    x = torch.tensor(
        loaded.tokenizer(text).data["input_ids"],
        dtype=torch.long,
        device=loaded.device,
    )[None, ...]
    text_tokens: list[int] = []
    audio_frames: list[list[int]] = []
    with torch.no_grad():
        for y, audio_frame in loaded.model.generate(
            x,
            loaded.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.85,
            stream=True,
            return_audio_codes=True,
        ):
            if y is not None:
                text_tokens = y[0].tolist()
            if audio_frame:
                audio_frames.append(audio_frame)
    return text_tokens, audio_frames


def _pack_audio_onto_bridges(
    bridges: list[BridgePayload],
    audio_frames: list[list[int]],
) -> tuple[BridgePayload, ...]:
    """Attach flat codec ids onto typed bridges (connector-only, no handle stash).

    Frame ``i`` goes on bridge ``i``. Overflow frames (beyond ``len(bridges)``)
    are flattened onto the last bridge so Talker can still recover them.
    """
    if not bridges:
        return ()
    packed: list[BridgePayload] = []
    n = len(bridges)
    for i, bridge in enumerate(bridges):
        if i < len(audio_frames) and i < n - 1:
            codes = tuple(int(c) for c in audio_frames[i])
        elif i == n - 1:
            # Last bridge gets its own frame plus any overflow frames.
            rest = audio_frames[i:] if i < len(audio_frames) else []
            codes = tuple(int(c) for frame in rest for c in frame)
        else:
            codes = ()
        packed.append(
            BridgePayload(
                tokens=bridge.tokens,
                hidden_states=bridge.hidden_states,
                audio_codes=codes,
            )
        )
    return tuple(packed)


def frames_from_bridges(
    bridges: tuple[BridgePayload, ...],
    codebooks: int = MIMI_CODEBOOKS,
) -> list[list[int]]:
    """Recover per-frame codebook lists from typed bridge ``audio_codes``."""
    frames: list[list[int]] = []
    for bridge in bridges:
        codes = list(bridge.audio_codes)
        if not codes:
            continue
        if len(codes) % codebooks != 0:
            # Truncate a ragged tail rather than invent pads mid-frame.
            codes = codes[: len(codes) - (len(codes) % codebooks)]
        for offset in range(0, len(codes), codebooks):
            frames.append(codes[offset : offset + codebooks])
    return frames


def apply_talker_watchdog(
    frames: list[list[int]],
    thinker_bridge_count: int,
    max_steps_after_last: int = TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN,
) -> list[list[int]]:
    """Keep bridge-conditioned frames; cap post-bridge tail (PR #3796 watchdog).

    Negative ``max_steps_after_last`` disables the limit.
    """
    if max_steps_after_last < 0:
        return frames
    limit = thinker_bridge_count + max_steps_after_last
    if len(frames) <= limit:
        return frames
    return frames[:limit]


class MinimindThinker(Stage[str, ThinkerRun]):
    """MiniMind-O Thinker with nano post-EOS bridge stream (default 128 pads)."""

    name = "thinker"
    forced_padding_count = THINKER_FORCED_PADDING_DEFAULT

    def __init__(self, handle: _LazyHandle) -> None:
        self._handle = handle

    def execute(self, payload: str) -> ThinkerRun:
        loaded = self._handle()
        text_tokens, audio_frames = _run_generation(loaded, payload)
        text = loaded.tokenizer.decode(text_tokens, skip_special_tokens=True)
        eos = loaded.tokenizer.eos_token_id
        visible = BridgePayload(
            tokens=TokenPayload(token_ids=tuple(text_tokens), text=text),
            hidden_states=TensorPayload(
                values=tuple(float(t) for t in text_tokens[:1]) or (0.0,),
                shape=(max(len(text_tokens), 1), 1),
            ),
        )
        forced: list[BridgePayload] = []
        for step in range(self.forced_padding_count):
            forced.append(
                BridgePayload(
                    tokens=TokenPayload(
                        token_ids=(eos,),
                        text="",
                        metadata={"forced": "true", "step": str(step)},
                    ),
                    hidden_states=TensorPayload(
                        values=(float(eos + step + 1),),
                        shape=(1, 1),
                    ),
                )
            )
        bridges = _pack_audio_onto_bridges([visible, *forced], audio_frames)
        return ThinkerRun(
            bridges=bridges,
            visible_tokens=visible.tokens,
            eos_token_id=eos,
            forced_padding_count=self.forced_padding_count,
        )


class MinimindTalker(Stage[ThinkerRun, CodecTokenPayload]):
    """MiniMind-O Talker: bridges only + MTP mask + post-bridge watchdog."""

    name = "talker"
    codebooks = MIMI_CODEBOOKS
    max_steps_after_last_thinker_token = TALKER_MAX_STEPS_AFTER_LAST_THINKER_TOKEN

    def __init__(self, handle: _LazyHandle | None = None) -> None:
        # handle kept for bundle API symmetry; Talker no longer reads it.
        self._handle = handle

    def execute(self, payload: ThinkerRun) -> CodecTokenPayload:
        frames = apply_talker_watchdog(
            frames_from_bridges(payload.bridges, self.codebooks),
            thinker_bridge_count=len(payload.bridges),
            max_steps_after_last=self.max_steps_after_last_thinker_token,
        )
        active_mask = tuple(
            tuple(k <= t for k in range(self.codebooks)) for t in range(len(frames))
        )
        token_ids: list[int] = []
        for frame_idx, frame in enumerate(frames):
            for codebook_idx in range(self.codebooks):
                code = frame[codebook_idx] if codebook_idx < len(frame) else MIMI_AUDIO_PAD_TOKEN
                if not active_mask[frame_idx][codebook_idx] or code >= MIMI_AUDIO_PAD_TOKEN:
                    token_ids.append(AUDIO_PADDING_TOKEN_ID)
                else:
                    token_ids.append(int(code))
        return CodecTokenPayload(
            token_ids=tuple(token_ids),
            codebooks=self.codebooks,
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
