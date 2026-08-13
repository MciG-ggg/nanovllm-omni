"""Real-weight MiniMind-O (``jingyaogong/minimind-3o``) stage implementations.

Implements the three logical stages (Thinker, Talker, Code2Wav) for the
real MiniMind-O weights while preserving the existing public pipeline
seam. The upstream model performs Thinker + Talker in a single
``stream_generate`` pass, so the loaded model is shared between
``MinimindThinker`` and ``MinimindTalker`` via a lazy handle; the Thinker stashes
per-frame audio codes on the handle and the Talker reads them.

Weights are downloaded from the Hugging Face Hub on first execute (not
on bundle construction) so the package remains importable without
``torch`` / ``transformers`` and so CI can keep running the model-free
suite without network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.payloads import (
    AUDIO_PADDING_TOKEN_ID,
    AudioPayload,
    BridgePayload,
    CodecTokenPayload,
    TensorPayload,
    ThinkerRun,
    TokenPayload,
)
from nanovllm_omni.stage import Stage

# Default Hugging Face Hub id for MiniMind-O. Weights are ~1 GB and are
# not committed to the repository -- they are downloaded on first use.
DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"
DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"

# MiniMind-O's audio codec vocab uses id >= 2049 to signal "stop / no
# code". We surface that as the typed audio padding token the Talker MTP
# mask uses, so downstream consumers see one canonical padding id.
MIMI_AUDIO_PAD_TOKEN = 2049

# Number of Mimi codebook layers emitted per audio frame (Talker MTP
# heads in ``MiniMindOmni.TalkerHead``).
MIMI_CODEBOOKS = 8

# Sample rate Mimi decodes to (matches MiniMind-O's published rate).
MIMI_SAMPLE_RATE = 24_000


@dataclass
class _LoadedMiniMind:
    """A loaded MiniMind-O model + tokenizer + Mimi codec bundle.

    Holds the heavy state shared by ``MinimindThinker`` / ``MinimindTalker`` /
    ``MinimindCode2Wav`` so weights download + load exactly once per bundle.
    ``last_audio_frames`` is the per-request cache the Thinker writes
    after one ``stream_generate`` pass and the Talker reads.
    """

    model: Any
    tokenizer: Any
    mimi: Any
    device: str
    model_id: str
    last_audio_frames: list[list[int]]
    # Ponytail: keep the imported torch module on the bundle so the
    # downstream stages do not need a second `import torch` (which would
    # raise ``ModuleNotFoundError`` without our clean RuntimeError
    # message).
    torch: Any = None


def _resolve_snapshot(model_id: str) -> str:
    """Use a local directory as-is; otherwise download from the HF Hub."""
    from pathlib import Path

    path = Path(model_id)
    if path.is_dir():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


class _LazyHandle:
    """Lazily-resolved shared MiniMind-O bundle.

    ``MinimindThinker`` / ``MinimindTalker`` / ``MinimindCode2Wav`` constructed via
    ``load_minimind_omni_bundle`` share a single instance so the first
    ``execute`` triggers exactly one download + load and every later
    call reuses the same in-memory weights.
    """

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
            except ImportError as exc:  # pragma: no cover - smoke test path only
                raise RuntimeError(
                    "torch is required to load real MiniMind-O weights; "
                    "install with `pip install nanovllm-omni[minimind]`"
                ) from exc
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = _load_minimind(self.model_id, device, self.mimi_model_id)
        return self._loaded

    def reset(self) -> None:
        """Release the loaded bundle so the next call re-downloads.

        Optional explicit cleanup hook; the orchestrator does not require
        it because per-request state is already cleared in its ``finally``.
        """
        self._loaded = None


def _load_minimind(
    model_id: str,
    device: str,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
) -> _LoadedMiniMind:
    """Load MiniMind-O + Mimi from local dirs or the HF Hub."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    snapshot_dir = _resolve_snapshot(model_id)
    mimi_dir = _resolve_snapshot(mimi_model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, trust_remote_code=True
    ).eval()
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
        last_audio_frames=[],
        torch=torch,
    )


def _run_generation(loaded: _LoadedMiniMind, prompt: str) -> tuple[list[int], list[list[int]]]:
    """Drive MiniMind-O's ``stream_generate`` and collect text + audio codes.

    Mirrors the upstream ``eval_omni.eval_sample`` flow: builds a
    chat-template prompt and iterates ``model.generate(stream=True,
    return_audio_codes=True)``, accumulating both yield outputs. The
    generator yields ``(text_tokens, audio_frame)`` pairs; ``text_tokens``
    is the cumulative sequence and ``audio_frame`` is the per-step list
    of 8 codebook ints (or ``None`` for steps without a complete frame).
    """
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


class MinimindThinker(Stage[str, ThinkerRun]):
    """MiniMind-O Thinker.

    Drives the upstream ``stream_generate`` and returns a ``ThinkerRun``
    whose single bridge carries the visible text. ``forced_padding_count``
    is 0: MiniMind-O does not force post-EOS padding (the audio path
    stops per-layer on the codec stop token, not via a fixed post-EOS
    length). The orchestrator's state machine handles ``forced_padding_count
    <= 0`` by walking ``PENDING -> VISIBLE_EOS -> DOWNSTREAM_READY`` in
    one step.
    """

    name = "thinker"
    forced_padding_count = 0

    def __init__(self, handle: _LazyHandle) -> None:
        self._handle = handle

    def execute(self, payload: str) -> ThinkerRun:
        loaded = self._handle()
        # Reset the shared per-request audio cache before generation so a
        # partial failure on a prior request cannot leak into this one.
        loaded.last_audio_frames = []
        try:
            text_tokens, audio_frames = _run_generation(loaded, payload)
        except Exception:
            loaded.last_audio_frames = []
            raise
        loaded.last_audio_frames = audio_frames
        text = loaded.tokenizer.decode(text_tokens, skip_special_tokens=True)
        # Ponytail: hidden-state values are a thin typed stand-in for the
        # real bridge hidden state. The connector contract cares about
        # the BridgePayload shape; downstream consumers only need the
        # text token ids for the Talker (real Talker reads audio_frames
        # from the shared handle, not from this tensor).
        bridge = BridgePayload(
            tokens=TokenPayload(token_ids=tuple(text_tokens), text=text),
            hidden_states=TensorPayload(
                values=tuple(float(t) for t in text_tokens[:1]) or (0.0,),
                shape=(max(len(text_tokens), 1), 1),
            ),
        )
        return ThinkerRun(
            bridges=(bridge,),
            visible_tokens=bridge.tokens,
            eos_token_id=loaded.tokenizer.eos_token_id,
            forced_padding_count=0,
        )


class MinimindTalker(Stage[ThinkerRun, CodecTokenPayload]):
    """MiniMind-O Talker.

    Reads the per-frame audio codes the Thinker stashed on the shared
    handle and converts them into a ``CodecTokenPayload`` with the same
    delayed MTP active mask:

    * ``active_mask[t][k] = k <= t`` (delayed activation per codebook)
    * inactive positions are filled with ``AUDIO_PADDING_TOKEN_ID``
    * codes >= ``MIMI_AUDIO_PAD_TOKEN`` (2049) are mapped to padding
    """

    name = "talker"
    codebooks = MIMI_CODEBOOKS

    def __init__(self, handle: _LazyHandle) -> None:
        self._handle = handle

    def execute(self, payload: ThinkerRun) -> CodecTokenPayload:
        loaded = self._handle()
        frames = loaded.last_audio_frames
        active_mask = tuple(
            tuple(k <= t for k in range(self.codebooks)) for t in range(len(frames))
        )
        token_ids: list[int] = []
        for frame_idx, frame in enumerate(frames):
            for codebook_idx, code in enumerate(frame):
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
    """Mimi decode stage.

    Runs ``MimiModel.decode`` on the Talker's codec tokens and returns a
    24 kHz mono waveform typed as ``AudioPayload``. Inactive / padding
    positions are already ``AUDIO_PADDING_TOKEN_ID`` (0); MimiModel
    treats 0 as the legitimate "no code" symbol.
    """

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
            # Re-shape (frames * codebooks,) -> [batch=1, codebooks, frames].
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
    """Bundle of three MiniMind-O stages sharing one loaded model."""

    thinker: MinimindThinker
    talker: MinimindTalker
    code2wav: MinimindCode2Wav
    model_id: str


def load_minimind_omni_bundle(
    model_id: str = DEFAULT_MINIMIND_MODEL_ID,
    device: str | None = None,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
) -> MinimindBundle:
    """Return the three MiniMind-O stages sharing one lazy bundle.

    Weights download + load on the first ``execute`` (lazy loading), not
    on bundle construction, so a pipeline can be assembled in
    environments without HF access. Pass an explicit ``device`` to pin
    CPU / CUDA at construction time; otherwise it follows ``torch.cuda.
    is_available()``. Local directories are accepted for ``model_id`` /
    ``mimi_model_id`` (offline / air-gapped runs).
    """
    handle = _LazyHandle(model_id=model_id, device=device, mimi_model_id=mimi_model_id)
    return MinimindBundle(
        thinker=MinimindThinker(handle),
        talker=MinimindTalker(handle),
        code2wav=MinimindCode2Wav(handle),
        model_id=model_id,
    )
