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

from collections.abc import Callable
from typing import Any

from nanovllm_omni.outputs import AudioPayload

from .audio import attach_audio_encoder, load_audio
from .bundle import MIMI_SAMPLE_RATE, MinimindBundle, create_bundle


def _thinker_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: returns a callable that runs the thinker.

    For TICKET 02, the "thinker" invokes the entire end-to-end pipeline via
    ``generate_audio`` so that the field topology is exercised without
    requiring the 3-stage split (TICKET 05).
    """
    extra_args = dict(getattr(args, "extra", None) or {})
    mimi_model_id = extra_args.pop("mimi_model_id", None) or extra_args.pop("mimi", None)
    audio_encoder_path = extra_args.pop("audio_encoder_path", None) or extra_args.pop(
        "audio_encoder", None
    )
    bundle_kwargs: dict[str, Any] = {
        "trust_remote_code": getattr(args, "trust_remote_code", True),
        "dtype": getattr(args, "dtype", None),
    }
    if mimi_model_id:
        bundle_kwargs["mimi_model_id"] = mimi_model_id
    bundle = create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)
    # deploy-layer default: route served requests through the CUDA-Graph
    # fast path when the yaml enables it (report §44). generate_audio(None)
    # resolves from bundle.use_cuda_graph.
    if bundle is not None:
        bundle.use_cuda_graph = bool(getattr(deploy, "use_cuda_graph", True))

    def thinker_forward(payload: Any, sampling: Any) -> Any:
        prompt = payload if isinstance(payload, str) else payload.get("prompt", "")
        extra = (
            (sampling.extra or {}) if sampling is not None and hasattr(sampling, "extra") else {}
        )
        audio_inputs = audio_lens = None
        transcript = None
        n_markers = 0
        audio = extra.get("audio")
        if audio is not None:
            # Q1/Q4: SenseVoice fbank -> engine-native prefill injection.
            # Q2: same model transcribes the speech (double-track ASR).
            sv = attach_audio_encoder(bundle, audio_encoder_path)
            samples = load_audio(audio)
            audio_inputs, audio_lens, n_markers = sv.fbank(samples)
            audio_inputs = audio_inputs.to(getattr(bundle, "device", "cpu"))
            audio_lens = audio_lens.to(getattr(bundle, "device", "cpu"))
            transcript = sv.transcribe(samples)
        out = generate_audio(
            bundle,
            prompt,
            max_tokens=int(sampling.max_tokens) if sampling is not None else 16,
            temperature=float(sampling.temperature) if sampling is not None else 0.7,
            top_p=float(sampling.top_p) if sampling is not None else 1.0,
            open_thinking=bool(extra.get("open_thinking", False)),
            audio_inputs=audio_inputs,
            audio_lens=audio_lens,
            audio_markers=n_markers,
        )
        if transcript:
            # AudioPayload is frozen; carry the double-track ASR transcript
            # on a thin wrapper so ``from_pipeline`` can surface it without
            # mutating the modal payload.
            from types import SimpleNamespace

            out = SimpleNamespace(audio=out, transcript=transcript)
        return out

    return thinker_forward


def tokenize_for_generate(
    tokenizer: Any,
    prompt: str,
    open_thinking: bool,
    *,
    messages: list[dict[str, str]] | None = None,
    audio_markers: int = 0,
    audio_special_token: str = "<|audio_pad|>",
) -> Any:
    """Apply the chat template and produce a 1xT ``input_ids`` tensor.

    Labeled ``tokenize`` for the benchmark harness; pure CPU, no model call.

    ``messages`` is an optional pre-built chat messages list (system +
    user, etc.). When omitted, the helper wraps ``prompt`` as a single
    user message -- the same path that ``generate_audio`` uses for the
    MiniMind-O single-prompt API.

    ``audio_markers`` prepends that many ``<|audio_pad|>`` tokens to the user
    content; the model's ``inject_audio_features`` replaces those positions
    with audio embeddings at prefill (engine-native audio input).
    """
    import torch

    with torch.profiler.record_function("tokenize"):
        if messages is None:
            content = audio_special_token * audio_markers if audio_markers else ""
            if prompt:
                content = (content + "\n" if content else "") + prompt
            messages = [{"role": "user", "content": content}]
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
    audio_inputs: Any = None,
    audio_lens: Any = None,
    use_cuda_graph: bool = False,
    seed: int | None = None,
    capture_bridge_states: bool = False,
    bridge_state_callback: Callable[[Any], None] | None = None,
    post_eos_padding_count: int = 0,
    internal_stop_token_id: int | None = None,
) -> list[list[int]]:
    """Stream ``model.generate`` and collect Mimi codebook frames.

    Returns a list of 8-token frames (one per yielded audio chunk) that the
    codec stage consumes. Labeled ``generate`` for the benchmark harness.
    ``audio_inputs`` / ``audio_lens`` (when set) ride through to the batched
    runner's prefill so the thinker sees user speech (engine-native audio in).

    ``use_cuda_graph=True`` (opt-in, default off — public path unchanged)
    routes text+audio decode through the CUDA-Graph fixed-KV-buffer decoder
    (``optim.cuda_graph``). It returns the same list-of-8-token frames; only
    the forward path differs. Falls back to eager ``stream_generate`` when
    CUDA is unavailable or the model isn't capture-compatible.

    ``seed`` (default None) controls the sampling RNG for the CUDA-Graph
    path. When None, the caller's current process seed
    (``torch.initial_seed()``) is used — same determinism contract as the
    eager path, whose caller seeds ``torch.manual_seed``. Previously the
    graph path hardcoded 42 and ignored the caller's seed (determinism-
    parity defect); now it honors it.

    ``capture_bridge_states`` and ``bridge_state_callback`` expose the
    additive eager capture seam used by the future talker stage. The CUDA
    Graph path does not capture bridge states in Phase 1.
    """
    import torch

    with torch.profiler.record_function("generate"):
        frames: list[list[int]] = []
        # Full post-EOS mode stays eager until the graph decoder accepts the
        # internal-stop sequence and bridge capture; its current visible-EOS
        # stop logic cannot satisfy that contract (Phase 4 integration).
        if (
            use_cuda_graph
            and post_eos_padding_count == 0
            and all(
                hasattr(model, name)
                for name in ("forward", "audio_pad_token", "audio_stop_token", "audio_spk_token")
            )
        ):
            # Graph fast path: joint text+audio decode, frames = transpose of
            # the 8 audio channels (Mimi codebook frames, codec-stage format).
            # Honor the caller's seed (fall back to the process RNG) so the
            # graph path matches eager determinism for a given manual_seed.
            # defect B: thread eos_token_id so graphed decode halts at the
            # content-natural end (parity with BatchedThinkerRunner.step_finished).
            # audio_stop_token falls back to ``model.audio_stop_token`` inside
            # ``enable_cuda_graph`` (matches eager's batched_generation.py:119).
            from nanovllm_omni.optim.cuda_graph import enable_cuda_graph

            decoder = enable_cuda_graph(model, n_steps=max_new_tokens, eos_token_id=eos_token_id)
            if decoder is not None:
                call_seed = seed if seed is not None else int(torch.initial_seed())
                _, audio_codes = decoder.generate_tokens(
                    input_ids, seed=call_seed, return_audio=True
                )
                num_frames = len(audio_codes[0])
                frames = [[audio_codes[ch][t] for ch in range(8)] for t in range(num_frames)]
                return frames
        if all(
            hasattr(model, name)
            for name in ("forward", "audio_pad_token", "audio_stop_token", "audio_spk_token")
        ):
            from .generation import stream_generate

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
                audio_inputs=audio_inputs,
                audio_lens=audio_lens,
                capture_bridge_states=capture_bridge_states,
                bridge_state_callback=bridge_state_callback,
                post_eos_padding_count=post_eos_padding_count,
                internal_stop_token_id=internal_stop_token_id,
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
    audio_inputs: Any = None,
    audio_lens: Any = None,
    audio_markers: int = 0,
    use_cuda_graph: bool | None = None,
) -> AudioPayload:
    """Run MiniMind-O stream generate and Mimi-decode to ``AudioPayload``.

    Public entry point used by both the Omni entrypoint and the bench
    harness. The four helper calls happen inside a single ``no_grad`` block
    so CUDA memory peaks are not doubled by intermediate allocations.

    ``use_cuda_graph`` (default None) routes decode through the CUDA-Graph
    fixed-KV-buffer decoder (report §40: 3.1-3.7x generate; determinism +
    robustness verified). None resolves from ``bundle.use_cuda_graph``
    (set by the deploy layer, deploy/minimind_omni.yaml), else False -- so
    the library call default stays eager while deployment opts in via yaml.
    """
    import torch

    if use_cuda_graph is None:
        use_cuda_graph = bool(getattr(bundle, "use_cuda_graph", False))

    from .code2wav import decode_audio, encode_wav

    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    audio_special_token = getattr(
        getattr(bundle.model, "config", None), "audio_special_token", "<|audio_pad|>"
    )
    with torch.no_grad():
        input_ids = tokenize_for_generate(
            bundle.tokenizer,
            prompt,
            open_thinking,
            audio_markers=audio_markers,
            audio_special_token=audio_special_token,
        ).to(bundle.device)
        frames = run_generate(
            bundle.model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            open_thinking=open_thinking,
            audio_inputs=audio_inputs,
            audio_lens=audio_lens,
            use_cuda_graph=use_cuda_graph,
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
