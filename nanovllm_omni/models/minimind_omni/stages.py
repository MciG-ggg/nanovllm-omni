"""MiniMind-O stage factories and end-to-end audio generation.

Loads the HF trust_remote_code MiniMindOmni checkpoint and runs its built-in
Thinker→Talker stream (`generate(..., return_audio_codes=True)`), then decodes
Mimi codebook frames to 24 kHz mono PCM via `MimiModel.decode`.
"""

from __future__ import annotations

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


def _float_to_pcm16(samples: Any) -> bytes:
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


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


def generate_audio(
    bundle: MinimindBundle,
    prompt: str,
    *,
    max_tokens: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    open_thinking: bool = False,
) -> AudioPayload:
    """Run MiniMind-O stream generate and Mimi-decode to ``AudioPayload``."""
    import torch

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    mimi = bundle.mimi

    messages = [{"role": "user", "content": prompt}]
    try:
        inputs_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=open_thinking,
        )
    except TypeError:
        inputs_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = torch.tensor(
        tokenizer(inputs_text).data["input_ids"],
        dtype=torch.long,
        device=device,
    )[None, ...]

    audio_frames: list[list[int]] = []
    with torch.no_grad():
        stream = model.generate(
            input_ids,
            tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            return_audio_codes=True,
            open_thinking=open_thinking,
        )
        for _text_ids, audio_frame in stream:
            if audio_frame and len(audio_frame) == 8:
                audio_frames.append(audio_frame)

    if not audio_frames:
        # Empty but valid silent WAV keeps the seam contract intact.
        return AudioPayload(data=b"", sample_rate=MIMI_SAMPLE_RATE)

    codes = torch.tensor(audio_frames, dtype=torch.long, device=device).T.unsqueeze(0)
    filtered = torch.where(codes >= MIMI_CODE_VOCAB_LIMIT, torch.zeros_like(codes), codes)
    with torch.no_grad():
        audio = mimi.decode(filtered).audio_values
    pcm = _float_to_pcm16(audio.squeeze().float().cpu().numpy())
    return AudioPayload(data=pcm, sample_rate=MIMI_SAMPLE_RATE)
