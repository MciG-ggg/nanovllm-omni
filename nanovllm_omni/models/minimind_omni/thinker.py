"""MiniMind-O thinker stage.

Owns the thinker's factory function (``_thinker_stage``) consumed by the
pipeline runner, plus the end-to-end ``generate_audio`` wrapper and the
two helper functions (``tokenize_for_generate`` / ``run_generate``) it
chains. Codec decode lives in ``code2wav.py``; bundle loading lives in
``bundle.py``.

The ``_thinker_stage`` factory wraps the end-to-end call behind the
thinker / talker / code2wav split. TICKET-05 is the correctness-side split
that turns this into a real 3-stage execution; this file is the prerequisite.
"""

from __future__ import annotations

from typing import Any

from nanovllm_omni.outputs import AudioPayload

from .bundle import MIMI_SAMPLE_RATE, MinimindBundle, create_bundle


def _thinker_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: returns a callable that runs the thinker.

    For TICKET 02, the "thinker" invokes the entire end-to-end pipeline via
    ``generate_audio`` so that the field topology is exercised without
    requiring the 3-stage split (TICKET 05).
    """
    extra_args = dict(getattr(args, "extra", None) or {})
    mimi_model_id = extra_args.pop("mimi_model_id", None) or extra_args.pop("mimi", None)
    bundle_kwargs: dict[str, Any] = {
        "trust_remote_code": getattr(args, "trust_remote_code", True),
        "dtype": getattr(args, "dtype", None),
    }
    if mimi_model_id:
        bundle_kwargs["mimi_model_id"] = mimi_model_id
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)

    def thinker_forward(payload: Any, sampling: Any) -> Any:
        prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
        extra = (
            (sampling.extra or {}) if sampling is not None and hasattr(sampling, "extra") else {}
        )
        return generate_audio(
            bundle,
            prompt,
            max_tokens=int(sampling.max_tokens) if sampling is not None else 16,
            temperature=float(sampling.temperature) if sampling is not None else 0.7,
            top_p=float(sampling.top_p) if sampling is not None else 1.0,
            open_thinking=bool(extra.get("open_thinking", False)),
        )

    return thinker_forward


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

    from .generation import stream_generate

    with torch.profiler.record_function("generate"):
        frames: list[list[int]] = []
        if all(
            hasattr(model, name)
            for name in ("forward", "audio_pad_token", "audio_stop_token", "audio_spk_token")
        ):
            stream = stream_generate(
                model,
                input_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                return_audio_codes=True,
                open_thinking=open_thinking,
            )
        else:
            # TODO: delete
            # Keep lightweight/test doubles compatible with the public seam.
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

    from .code2wav import decode_audio, encode_wav

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


__all__ = [
    "_thinker_stage",
    "generate_audio",
    "run_generate",
    "tokenize_for_generate",
]
