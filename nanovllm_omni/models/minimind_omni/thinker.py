"""MiniMind-O thinker stage.

Owns the ``_thinker_stage`` factory, the ``generate_audio`` end-to-end
wrapper, and its ``tokenize_for_generate`` / ``run_generate`` helpers.
Public symbols: ``_thinker_stage``, ``generate_audio``, ``run_generate``,
``tokenize_for_generate``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nanovllm_omni.outputs import AudioPayload

from ._stage import stage
from .bundle import MIMI_SAMPLE_RATE, MinimindBundle, create_bundle


def _thinker_stage(deploy: Any, args: Any) -> Any:
    """Stage 0 factory: runs the bridge-capturing thinker for the talker.

    Emits a ``ThinkerStageOutput`` carrying bridge hidden states + text span
    that ``thinker2talker`` converts for the talker stage.
    """
    extra_args = dict(getattr(args, "extra", None) or {})
    mimi_model_id = extra_args.pop("mimi_model_id", None) or extra_args.pop("mimi", None)
    provided_bundle = extra_args.pop("bundle", None)
    bundle_kwargs: dict[str, Any] = {
        "trust_remote_code": getattr(args, "trust_remote_code", True),
        "dtype": getattr(args, "dtype", None),
        "enforce_eager": bool(getattr(args, "enforce_eager", False)),
    }
    if mimi_model_id:
        bundle_kwargs["mimi_model_id"] = mimi_model_id
    # ``extra["bundle"]`` lets offline tests / bench harnesses inject a
    # prebuilt bundle so the factory never touches the network.
    bundle = (
        provided_bundle
        if provided_bundle is not None
        else create_bundle(model_id=args.model, device=args.device, **bundle_kwargs)
    )
    if bundle is not None:
        bundle.use_thinker_cuda_graph = bool(getattr(deploy, "use_thinker_cuda_graph", True))

    return _full_thinker_stage(bundle, deploy)


def _full_thinker_stage(bundle: Any, deploy: Any) -> Any:
    """Full-mode stage 0: emit a ``ThinkerStageOutput`` with bridge states.

    Runs the bridge-capturing generation with the deploy layer's post-EOS
    sequence and internal-stop token, then hands the aligned bridge +
    token ids to ``thinker2talker``. The thinker does NOT decode audio
    here -- that is the code2wav stage's job.
    """
    post_eos_padding_count = int(getattr(deploy, "post_eos_padding_count", 128) or 0)
    internal_stop_token_id = getattr(deploy, "internal_stop_token_id", None)

    def thinker_forward_full(payload: Any, sampling: Any) -> Any:
        import uuid

        import torch

        from .generation import stream_generate
        from .stage_processors import ThinkerStageOutput

        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict):
            prompt = payload.get("prompt", "")
        else:
            prompt = str(payload)
        extra = (
            dict(getattr(sampling, "extra", None) or {})
            if sampling is not None and hasattr(sampling, "extra")
            else {}
        )
        if extra.get("audio") is not None:
            raise NotImplementedError(
                "MiniMind full pipeline is text-to-audio only; audio input "
                "(ASR) is not supported on the three-stage path."
            )
        eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
        audio_special_token = getattr(
            getattr(bundle.model, "config", None), "audio_special_token", "<|audio_pad|>"
        )
        request_id = f"mmo-full-{uuid.uuid4().hex[:12]}"
        with torch.no_grad():
            input_ids = tokenize_for_generate(
                bundle.tokenizer,
                prompt,
                bool(extra.get("open_thinking", False)),
                audio_special_token=audio_special_token,
            ).to(bundle.device)
            captured_bridge: list[torch.Tensor] = []
            output_tokens: list[int] = []
            stream = stream_generate(
                bundle.model,
                input_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=int(sampling.max_tokens) if sampling is not None else 512,
                temperature=float(sampling.temperature) if sampling is not None else 0.7,
                top_p=float(sampling.top_p) if sampling is not None else 0.9,
                open_thinking=bool(extra.get("open_thinking", False)),
                capture_bridge_states=True,
                bridge_state_callback=captured_bridge.append,
                post_eos_padding_count=post_eos_padding_count,
                internal_stop_token_id=internal_stop_token_id,
            )
            # ``stream_generate`` yields a growing prefix each step. The
            # original code rebuilt the whole Python int list (and forced
            # a host sync) every iteration. Defer the single
            # ``detach().cpu().tolist()`` until after the stream so the
            # per-step host sync disappears. Audio numerics are unchanged
            # because the final prefix is identical to the last yielded
            # ``text_chunk``.
            final_text_chunk: Any = None
            for text_chunk, _audio_frame in stream:
                if text_chunk is not None:
                    final_text_chunk = text_chunk
            if final_text_chunk is not None:
                output_tokens = [
                    int(token) for token in final_text_chunk.detach().cpu().reshape(-1).tolist()
                ]
        bridge = (
            captured_bridge[0]
            if captured_bridge and captured_bridge[0].numel() > 0
            else torch.empty(0, 0, dtype=torch.float32)
        )
        prompt_ids = (
            input_ids[0].detach().cpu().tolist() if input_ids.ndim == 2 else list(input_ids)
        )
        # Bridge-aligned span: prompt + output[1:] -- the runner predicts
        # the first output token at prefill, so the talker only decodes
        # the remaining tokens and the span length matches the bridge rows.
        aligned_text = prompt_ids + (output_tokens[1:] if len(output_tokens) > 1 else [])
        return ThinkerStageOutput(
            bridge_states=bridge,
            prompt_token_ids=prompt_ids,
            output_token_ids=output_tokens,
            text_token_ids=aligned_text,
            input_ids=input_ids,
            request_id=request_id,
            metadata={
                "pipeline_kind": "full",
                "post_eos_padding_count": post_eos_padding_count,
            },
        )

    return thinker_forward_full


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

    with stage("tokenize"):
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
    use_thinker_cuda_graph: bool = False,
    seed: int | None = None,
    capture_bridge_states: bool = False,
    bridge_state_callback: Callable[[Any], None] | None = None,
    text_token_callback: Callable[[list[int]], None] | None = None,
    post_eos_padding_count: int = 0,
    internal_stop_token_id: int | None = None,
) -> list[list[int]]:
    """Stream ``model.generate`` and collect Mimi codebook frames.

    Returns a list of 8-token frames (one per yielded audio chunk) that the
    codec stage consumes. Labeled ``generate`` for the benchmark harness.
    ``audio_inputs`` / ``audio_lens`` (when set) ride through to the batched
    runner's prefill so the thinker sees user speech (engine-native audio in).

    ``use_thinker_cuda_graph=True`` (opt-in, default off) routes text+audio decode
    through the CUDA-Graph fixed-KV-buffer decoder (``optim.cuda_graph``);
    returns the same list-of-8-token frames. Falls back to eager
    ``stream_generate`` when CUDA is unavailable or the model isn't
    capture-compatible.

    ``seed`` (default None) controls the sampling RNG for the CUDA-Graph
    path; when None, ``torch.initial_seed()`` is used (same determinism
    contract as the eager path).

    ``capture_bridge_states`` and ``bridge_state_callback`` expose the
    bridge seam used by the talker stage. The CUDA Graph path returns graph-owned
    bridge states when requested. ``text_token_callback`` receives all decoded
    text tokens for the full pipeline's talker alignment.
    """
    import torch

    with stage("generate"):
        frames: list[list[int]] = []
        # Graph path rejects post-EOS mode.
        if (
            use_thinker_cuda_graph
            and post_eos_padding_count == 0
            and all(
                hasattr(model, name)
                for name in ("forward", "audio_pad_token", "audio_stop_token", "audio_spk_token")
            )
        ):
            # Graph fast path: joint text+audio decode, frames = transpose of
            # the 8 audio channels (Mimi codebook frames, codec-stage format).
            # ``audio_stop_token`` falls back to ``model.audio_stop_token``
            # inside ``enable_cuda_graph``.
            from nanovllm_omni.optim.cuda_graph import enable_cuda_graph

            decoder = enable_cuda_graph(model, n_steps=max_new_tokens, eos_token_id=eos_token_id)
            if decoder is not None:
                decoder.temperature = temperature
                decoder.top_p = top_p
                call_seed = seed if seed is not None else int(torch.initial_seed())
                result = decoder.generate_tokens(
                    input_ids,
                    seed=call_seed,
                    return_audio=True,
                    return_bridge=capture_bridge_states,
                )
                if capture_bridge_states:
                    text_tokens, audio_codes, bridge = result
                    if bridge_state_callback is not None:
                        bridge_state_callback(bridge)
                else:
                    text_tokens, audio_codes = result
                if text_token_callback is not None:
                    text_token_callback(text_tokens)
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
            # Lightweight/test doubles path; kept for the public seam.
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
        text_tokens: list[int] = []
        for text_ids, audio_frame in stream:
            # ``generate.step`` shows up as a sub-event of ``generate`` in the
            # Kineto trace so per-iteration cost is visible in chrome://tracing.
            with stage("generate.step"):
                if text_ids is not None:
                    if hasattr(text_ids, "detach"):
                        text_ids = text_ids.detach().cpu().reshape(-1).tolist()
                    text_tokens = [int(token) for token in text_ids]
                if audio_frame and len(audio_frame) == 8:
                    frames.append(audio_frame)
        if text_token_callback is not None:
            text_token_callback(text_tokens)
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
    use_thinker_cuda_graph: bool | None = None,
) -> AudioPayload:
    """Run MiniMind-O stream generate and Mimi-decode to ``AudioPayload``.

    Public entry point used by both the Omni entrypoint and the bench
    harness. The four helper calls happen inside a single ``no_grad`` block
    so CUDA memory peaks are not doubled by intermediate allocations.

    ``use_thinker_cuda_graph`` (default None) routes decode through the CUDA-Graph
    fixed-KV-buffer decoder. None resolves from ``bundle.use_thinker_cuda_graph``
    (set by the deploy layer, deploy/minimind_omni.yaml), else False.
    """
    import torch

    if use_thinker_cuda_graph is None:
        use_thinker_cuda_graph = bool(getattr(bundle, "use_thinker_cuda_graph", False))

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
            use_thinker_cuda_graph=use_thinker_cuda_graph,
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
