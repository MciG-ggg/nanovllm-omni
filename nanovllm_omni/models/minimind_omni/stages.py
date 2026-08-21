"""MiniMind-O stage factories and end-to-end audio generation.

Loads the HF trust_remote_code MiniMindOmni checkpoint and runs its built-in
Thinker→Talker stream (`generate(..., return_audio_codes=True)`), then decodes
Mimi codebook frames to 24 kHz mono PCM via `MimiModel.decode`.

The four ``tokenize_for_generate`` / ``run_generate`` / ``decode_audio`` /
``encode_wav`` helpers are the public seam used by the Session-1 benchmark
harness (``nanovllm_omni.optim.bench``). Each helper opens a
``torch.profiler.record_function`` range whose name matches the bench
subpackage's CSV columns.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanovllm_omni.outputs import AudioPayload

DEFAULT_MINIMIND_MODEL_ID = "jingyaogong/minimind-3o"
DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"
MIMI_SAMPLE_RATE = 24_000
MIMI_CODE_VOCAB_LIMIT = 2048


@dataclass
class MinimindBundle:
    """Loaded MiniMind-O runtime pieces used by the aligned Omni path."""

    model: Any
    tokenizer: Any
    mimi: Any
    device: str
    model_id: str
    # Aliases kept for create_stages() callers that expect three handles.
    thinker: Any = None
    talker: Any = None
    code2wav: Any = None


def _resolve_snapshot(model_id: str) -> str:
    path = Path(model_id)
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_minimind_omni_bundle(
    model_id: str = DEFAULT_MINIMIND_MODEL_ID,
    device: str | None = None,
    mimi_model_id: str = DEFAULT_MIMI_MODEL_ID,
    **kwargs: Any,
) -> MinimindBundle:
    """Load MiniMind-O + tokenizer + Mimi onto ``device``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    device = _pick_device(device)
    snapshot_dir = _resolve_snapshot(model_id)
    mimi_dir = _resolve_snapshot(kwargs.get("mimi_model_id", mimi_model_id))

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, trust_remote_code=True).eval()
    if device != "cpu":
        model = model.half()
    model = model.to(device)

    mimi = MimiModel.from_pretrained(mimi_dir).eval()
    if device != "cpu":
        mimi = mimi.half()
    mimi = mimi.to(device)
    # Official eval_omni attaches mimi on the model for decode convenience.
    model.mimi_model = mimi

    return MinimindBundle(
        model=model,
        tokenizer=tokenizer,
        mimi=mimi,
        device=device,
        model_id=model_id,
        thinker=model,
        talker=getattr(model, "talker", None),
        code2wav=mimi,
    )


def create_bundle(model_id: str, device: str | None = None, **kwargs: Any) -> MinimindBundle:
    return load_minimind_omni_bundle(model_id=model_id, device=device, **kwargs)


def create_stages(model_id: str, device: str | None = None, **kwargs: Any):
    bundle = create_bundle(model_id, device, **kwargs)
    return bundle.thinker, bundle.talker, bundle.code2wav


def tokenize_for_generate(
    tokenizer: Any,
    prompt: str,
    open_thinking: bool,
    *,
    messages: list[dict[str, str]] | None = None,
) -> Any:
    """Apply the chat template and produce a 1xT ``input_ids`` tensor.

    Labeled ``tokenize`` for the benchmark harness; pure CPU, no model call.

    ``messages`` is an optional pre-built chat messages list (system +
    user, etc.). When omitted, the helper wraps ``prompt`` as a single
    user message -- the same path that ``generate_audio`` uses for the
    MiniMind-O single-prompt API.
    """
    import torch

    with torch.profiler.record_function("tokenize"):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=open_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Device is supplied by the caller in the generate_audio path; the
        # helper itself stays device-agnostic so unit tests can stub it.
        return torch.tensor(
            tokenizer(text).data["input_ids"],
            dtype=torch.long,
        )[None, ...]


def run_generate(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Any | None,
    open_thinking: bool,
) -> list[list[int]]:
    """Stream ``model.generate`` and collect Mimi codebook frames.

    Returns a list of 8-token frames (one per yielded audio chunk) that the
    codec stage consumes. Labeled ``generate`` for the benchmark harness.
    """
    import torch

    with torch.profiler.record_function("generate"):
        frames: list[list[int]] = []
        stream = model.generate(
            input_ids,
            eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            return_audio_codes=True,
            open_thinking=open_thinking,
        )
        for _text_ids, audio_frame in stream:
            # ``generate.step`` shows up as a sub-event of ``generate`` in the
            # Kineto trace so per-iteration cost is visible in chrome://tracing.
            with torch.profiler.record_function("generate.step"):
                if audio_frame and len(audio_frame) == 8:
                    frames.append(audio_frame)
        return frames


def decode_audio(
    mimi: Any,
    audio_frames: list[list[int]],
    device: str,
) -> Any:
    """Decode collected Mimi codebook frames to a float numpy array on CPU.

    Labeled ``decode`` for the benchmark harness. Moves codes to ``device``,
    runs ``mimi.decode`` under no_grad, then returns ``np.ndarray``.
    """
    import torch

    with torch.profiler.record_function("decode"):
        codes = torch.tensor(audio_frames, dtype=torch.long, device=device).T.unsqueeze(0)
        filtered = torch.where(codes >= MIMI_CODE_VOCAB_LIMIT, torch.zeros_like(codes), codes)
        with torch.no_grad():
            audio = mimi.decode(filtered).audio_values
        return audio.squeeze().float().cpu().numpy()


def encode_wav(samples: Any, sample_rate: int = 24_000) -> bytes:
    """Wrap float audio into a 16-bit mono PCM WAV byte string.

    Labeled ``wav`` for the benchmark harness. Uses stdlib ``wave`` only;
    accepts anything numpy can coerce to float32.
    """
    import numpy as np
    import torch

    with torch.profiler.record_function("wav"):
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return out.getvalue()


def generate_audio(
    bundle: MinimindBundle,
    prompt: str,
    *,
    max_tokens: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    open_thinking: bool = False,
) -> AudioPayload:
    """Run MiniMind-O stream generate and Mimi-decode to ``AudioPayload``.

    Public entry point used by both the Omni entrypoint and the bench
    harness. The four helper calls happen inside a single ``no_grad`` block
    so CUDA memory peaks are not doubled by intermediate allocations.
    """
    import torch

    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    with torch.no_grad():
        input_ids = tokenize_for_generate(bundle.tokenizer, prompt, open_thinking).to(bundle.device)
        frames = run_generate(
            bundle.model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            open_thinking=open_thinking,
        )
        if not frames:
            return AudioPayload(data=b"", sample_rate=MIMI_SAMPLE_RATE)
        samples = decode_audio(bundle.mimi, frames, bundle.device)
        wav_bytes = encode_wav(samples, sample_rate=MIMI_SAMPLE_RATE)

    return AudioPayload(data=wav_bytes, sample_rate=MIMI_SAMPLE_RATE)
