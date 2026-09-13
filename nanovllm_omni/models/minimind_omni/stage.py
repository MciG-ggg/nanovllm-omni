"""Stage wiring for MiniMind-O (the per-stage classes and the helpers
they need). Mirrors the ``stage.py`` convention used by
``models/sd_turbo/`` / ``smolvla/`` / ``smolvlm/``.

The fork-AR-Engine adapter layer (``StageRunner`` /
``_ensure_dist`` / ``get_stage_config`` /
``stage_kwargs_from_args``) lives in
:mod:`nanovllm_omni.engine.stage_runner`. This module owns
only the MiniMind-O specific bits:

  - weight slicing for thinker-only / talker-only safetensors
    (``_THINKER_KEEP_PREFIX`` / ``_TALKER_KEEP_PREFIX`` /
    ``_filter_only_dir`` / ``_resolve_model_path``)
  - HuggingFace tokenizer loader for the MiniMind-3o snapshot
  - ``ThinkerStage`` and ``TalkerStage`` high-level stage classes
    invoked by ``PipelineRunner``'s ``_stage_factory``.

Lazy load via factory: import this module only from inside
``thinker._thinker_stage`` / ``talker._talker_stage`` — never at the top
of a sibling module. ``ModelRunner.__init__`` (pulled in by
``ThinkerStage`` / ``TalkerStage``) drags in ``triton`` / ``flash-attn``,
which aren't available on CPU-only hosts.

Fork imports stay lazy — ``ModelRunner`` pulls in ``triton`` /
``flash-attn``, which isn't available on CPU-only hosts. The factory
functions (``_thinker_stage`` / ``_talker_stage`` in ``thinker.py`` /
``talker.py``) trigger the heavy ``ModelRunner.__init__``; smoke
tests that only import factory symbols don't.

nano-vllm is installed as an editable package via ``uv``
(``[tool.uv.sources]`` in the root ``pyproject.toml``), so
``from nanovllm...`` resolves through the normal package mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.engine.stage_runner import (
    StageRunner,
    _ensure_dist,
    get_stage_config,
    stage_kwargs_from_args,
)
from nanovllm_omni.models.minimind_omni.stage_runner import (
    decode_minimind,
    recapture_bridge_minimind,
)

# ---------------------------------------------------------------------------
# ThinkerStage / TalkerStage.
# ---------------------------------------------------------------------------


_THINKER_KEEP_PREFIX = ("embed_tokens.", "layers.", "norm.", "lm_head.")
# Vendor ``stream_generate`` post-EOS filler tokens (``enter_token_id``
# then ``pad_token_id`` in ``model_omni.py``). They never reach the user;
# they only keep the forward pass clocking so the talker tail drains.
_THINKER_ENTER_TOKEN = 201
_THINKER_PAD_TOKEN = 0
# Talker keys all live under submodules (``lm_head.base.`` /
# ``lm_head.adapters.`` / ...) — the bare ``lm_head.weight`` key
# belongs to the thinker's text head (shape 6400) and must be
# skipped. We match the dot-separated prefix and require at least
# one intermediate segment for ``lm_head``. ``speaker_proj.``
# replaces the vendor's ``spk_proj.`` (the fork loader substring
# replaces ``k_proj`` inside ``spk_proj``, which corrupts the path).
_TALKER_KEEP_PREFIX = (
    "embed_proj.",
    "codec_proj.",
    "embed_tokens.",
    "layers.",
    "norm.",
    "lm_head.base.",
    "lm_head.adapters.",
    "text_scale",
    "audio_scale",
    "speaker_proj.",
)
# Keys whose tail we rename while copying (src -> dst).
_TALKER_RENAME = {"spk_proj.weight": "speaker_proj.weight"}


def _resolve_model_path(args: Any, stage: str = "thinker") -> str:
    """Pull the model directory from ``OmniEngineArgs.model``.

    Fork ``Config.__post_init__`` asserts ``os.path.isdir(self.model)``.
    Resolves Hub IDs (``jingyaogong/minimind-3o``) to a local snapshot
    via ``_resolve_snapshot`` from ``bundle``. Local paths pass through.

    The directory is then shaped to the stage: ``_thinker_only_dir``
    keeps only the keys MiniMindThinker consumes; ``_talker_only_dir``
    keeps only the keys MiniMindTalker consumes. Fork ``load_model``
    walks every key and crashes on unknowns, so the two stages cannot
    share a single safetensors file.
    """
    from .bundle import _resolve_snapshot

    model = getattr(args, "model", None)
    if not model:
        raise ValueError(
            "OmniEngineArgs.model is required to construct a StageRunner "
            "(fork Config validates the model path is a real directory)."
        )
    snapshot = _resolve_snapshot(model)
    if stage == "thinker":
        return _thinker_only_dir(snapshot)
    if stage == "talker":
        return _talker_only_dir(snapshot)
    raise ValueError(f"unknown stage: {stage!r}")


def _filter_only_dir(
    src: str,
    dst_prefix: str,
    keep_prefix: tuple[str, ...],
    stage_prefix: str,
) -> str:
    """Shared implementation for thinker / talker only-dir builds.

    Strips ``stage_prefix`` (e.g. ``model.`` or ``talker.``) from each
    key, drops anything whose stripped name does not start with one
    of ``keep_prefix``, and writes a single ``model.safetensors``
    into a hash-keyed sibling directory under ``/tmp``.  Reuses the
    cache if it already exists.

    ``stage_prefix`` is also used as a *filter*: a key is only kept
    if it starts with ``stage_prefix``, or already matches
    ``keep_prefix`` with no prefix (HF MiniMind stores the thinker's
    tied ``lm_head.weight`` at the top level).  This prevents thinker
    layers (``model.layers.4-7.*``) from leaking into the talker build,
    since the talker's layer prefix is ``talker.layers.``.
    """
    import glob
    import hashlib
    import os
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = os.path.abspath(src)
    # Include stage_prefix so the previous thinker cache (missing
    # top-level ``lm_head.weight``) is not reused.
    cache_key = hashlib.sha1(f"{src}\0{stage_prefix}".encode()).hexdigest()[:12]
    dst = os.path.join("/tmp", f"{dst_prefix}-{cache_key}")
    target = os.path.join(dst, "model.safetensors")

    if os.path.isfile(target):
        return dst

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith((".json", ".py", ".jinja", ".md", ".txt")):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    def _name(k: str) -> str | None:
        if k.startswith(stage_prefix):
            name = k[len(stage_prefix) :]
            return _TALKER_RENAME.get(name, name)
        # Top-level keys that already match keep_prefix, e.g.
        # ``lm_head.weight`` in the HF MiniMind checkpoint.
        if k.startswith(keep_prefix):
            return k
        return None

    def _accept(name: str) -> bool:
        return name.startswith(keep_prefix)

    tensors: dict = {}
    seen: set = set()
    src_files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not src_files:
        src_bin = os.path.join(src, "pytorch_model.bin")
        if not os.path.isfile(src_bin):
            raise FileNotFoundError(f"no safetensors or pytorch_model.bin in {src}")
        state = torch.load(src_bin, map_location="cpu", weights_only=True)
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            name = _name(k)
            if name is None or not _accept(name):
                continue
            ptr = v.untyped_storage().data_ptr()
            if ptr in seen:
                v = v.clone()
            else:
                seen.add(ptr)
            tensors[name] = v.contiguous()
        save_file(tensors, target)
        return dst

    for src_file in src_files:
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
                name = _name(k)
                if name is None or not _accept(name):
                    continue
                v = f.get_tensor(k)
                ptr = v.untyped_storage().data_ptr()
                if ptr in seen:
                    v = v.clone()
                else:
                    seen.add(ptr)
                tensors[name] = v.contiguous()
    save_file(tensors, target)
    return dst


def _thinker_only_dir(src: str) -> str:
    """Build a sibling directory with only thinker-shaped weights."""
    return _filter_only_dir(src, "minimind-thinker-only", _THINKER_KEEP_PREFIX, "model.")


def _talker_only_dir(src: str) -> str:
    """Build a sibling directory with only talker-shaped weights.

    Mirrors ``_thinker_only_dir`` but keeps the talker's keys
    (``codec_proj.*``, ``embed_proj.*``, ``embed_tokens.*``,
    ``layers.*``, ``norm.*``, ``lm_head.base.*``, ``lm_head.adapters.*``,
    ``text_scale``, ``audio_scale``, ``spk_proj.*``). ``talker.``
    prefix stripped; the bare ``lm_head.weight`` (thinker text head)
    is filtered out by the prefix list.
    """
    return _filter_only_dir(src, "minimind-talker-only", _TALKER_KEEP_PREFIX, "talker.")


def _load_tokenizer(model_path: str) -> Any:
    """Load a HuggingFace tokenizer from the model directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class ThinkerStage:
    """Stage 0 — MiniMind thinker AR decode via fork ``ModelRunner``.

    The factory ``_thinker_stage`` (in ``thinker.py``) constructs this
    class; ``PipelineRunner`` then drives decoding by calling the
    instance with ``(payload, sampling)``.
    """

    def __init__(self, deploy: Any, args: Any, model_class: Any = None) -> None:
        from .thinker import MiniMindThinker

        self.deploy = deploy
        self.args = args
        self.model_class = model_class or MiniMindThinker
        # Graph capture fails on RTX 3050 with the current minimind-3o
        # fork Config (cudaErrorStreamCaptureInvalidated); eager is the
        # safe path. See ADR-0005. Bridge-buffer wiring lives in thinker.py
        # and is exercised by test_thinker_bridge_buffer.py.
        self.config = get_stage_config(
            "thinker",
            model_path=_resolve_model_path(args, stage="thinker"),
            **stage_kwargs_from_args(args),
        )
        self.config.enforce_eager = True
        # Ensure NCCL process group exists before ModelRunner tries to init.
        _ensure_dist()
        self.stage_runner = StageRunner(
            model_class=self.model_class,
            config=self.config,
        )

    def __call__(self, payload: Any, sampling: Any) -> Any:
        """Run the Thinker: tokenize prompt, prefill, AR decode, extract bridge.

        Args:
            payload: text prompt string.
            sampling: SamplingParams (or fork equivalent) with max_tokens, temperature.

        Returns:
            ThinkerStageOutput with bridge_states, token_ids, text_token_ids.
        """
        import torch

        from .stage_processors import ThinkerStageOutput

        prompt = str(payload)
        tokenizer = _load_tokenizer(_resolve_model_path(self.args, stage="thinker"))
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or 0]

        max_tokens = getattr(sampling, "max_tokens", None)
        if max_tokens is None:
            max_tokens = (getattr(sampling, "extra", None) or {}).get("max_tokens", 512)
        max_tokens = int(max_tokens)

        temperature = getattr(sampling, "temperature", None)
        if temperature is None:
            temperature = (getattr(sampling, "extra", None) or {}).get("temperature", 0.7)
        temperature = float(temperature)
        # Fork requires temperature > 1e-10 (no greedy).
        if temperature <= 1e-10:
            temperature = 1e-5

        # Create fork Sequence and Scheduler inside StageRunner.decode.

        _rp = float(
            (getattr(sampling, "extra", None) or {}).get("repetition_penalty", 1.05) or 1.05
        )
        _top_p = float(getattr(sampling, "top_p", 1.0) or 1.0)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = self.config.eos

        generated, bridge_hidden, text_state = decode_minimind(
            self.stage_runner,
            token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=_rp,
            top_p=_top_p,
            eos_id=eos_id,
            enter_token=_THINKER_ENTER_TOKEN,
            pad_token=_THINKER_PAD_TOKEN,
        )

        # Re-capture the bridge with one full-sequence prefill pass so the
        # talker sees full-attention rows.
        bridge_hidden = recapture_bridge_minimind(
            self.stage_runner,
            token_ids + list(generated),
            bridge_hidden,
        )

        return ThinkerStageOutput(
            bridge_states=bridge_hidden if bridge_hidden is not None else torch.empty(0),
            prompt_token_ids=tuple(token_ids),
            output_token_ids=tuple(generated),
            text_token_ids=tuple(token_ids + generated),
            text=tokenizer.decode(generated, skip_special_tokens=True),
            text_state=text_state,
            request_id=getattr(sampling, "request_id", None),
        )


@dataclass
class TalkerOutput:
    """Talker output: per-frame audio codes consumed by ``talker2code2wav``."""

    audio_codes: Any  # torch.Tensor [frames, 8]


# Vendor constant — single channel of the audio-buffer pad (model_omni.py).
_AUDIO_PAD_TOKEN = 2049
_AUDIO_STOP_TOKEN = 2050
_AUDIO_VENDOR_TEMPERATURE = 0.2
_AUDIO_REPETITION_PENALTY = 1.05


class TalkerStage:
    """Stage 1 — MiniMind talker MTP decode.

    Full-sequence per step (we re-run the growing audio sequence at
    every step, no incremental decode yet — that requires per-step
    block-table maintenance and a fork ``Sequence`` per request, see
    the pre-``prepare_inputs_embeds`` / ``body`` / ``compute_logits`` refactor is only partial, so full-sequence per-step is still used, not incremental decode). The fork ``Attention`` paged
    path is used so D = num_kv_heads * head_dim = 192 is no longer
    a Triton-kernel barrier (the kernel now pads to next-pow2).

    Audio sampling uses vendor's ``topk(50)`` plus
    ``torch.multinomial(softmax(...))`` with a repetition penalty on the
    last 3 codes per codebook. ``enforce_eager`` for the underlying fork
    flow is not applicable here because we don't go through a
    ``ModelRunner``; we set up the fork ``Context`` directly.
    """

    def __init__(self, deploy: Any, args: Any, model_class: Any = None) -> None:
        self.deploy = deploy
        self.args = args
        self.model_class = model_class
        self._model: Any = None
        self._device: str | None = None
        self._config: Any = None
        self._arange_cache: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the talker model from the talker-only safetensors."""
        if self._model is not None:
            return self._model

        import torch
        from nanovllm.config import Config
        from nanovllm.utils.loader import load_model

        from .talker import MiniMindTalker

        model_class = self.model_class or MiniMindTalker
        model_path = _resolve_model_path(self.args, stage="talker")

        # Build the fork Config with trust_remote_code so MiniMind's
        # ``auto_map`` config is loadable; the resulting ``hf_config`` is
        # what the talker model class consumes.
        cfg = Config(model=model_path, trust_remote_code=True)
        hf_cfg = cfg.hf_config
        self._config = hf_cfg

        # Ensure dist is ready (may already be initialised by the
        # thinker's ModelRunner inside the same process).
        _ensure_dist()

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_cfg.dtype)
        torch.set_default_device("cuda")
        try:
            model = model_class(hf_cfg)
            load_model(model, model_path)
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

        model.eval()
        self._model = model
        self._device = "cuda"
        return model

    def __call__(self, payload: Any, sampling: Any) -> Any:
        """Run the Talker: AR decode audio codes from bridge hidden states.

        Mirrors ``MiniMindOmni.stream_generate`` (vendor model_omni.py):
        at each step we run one forward (full sequence), sample one
        code per codebook from the 8 returned logit tensors using the
        vendor's staggered schedule (``audio_codes[i]`` lags codebook
        ``i`` by one position), and stop once codebook 7 has emitted
        ``audio_stop_token``.

        Args:
            payload: TalkerInputPayload (from thinker2talker processor).
            sampling: SamplingParams with temperature, max_tokens.

        Returns:
            TalkerOutput with audio_codes [frames, 8] tensor.
        """
        import torch
        from nanovllm.utils.context import reset_context, set_context

        from .stage_processors import TalkerInputPayload

        model = self._ensure_model()
        device = self._device

        # ---- Extract & validate inputs ----
        if not isinstance(payload, TalkerInputPayload):
            raise TypeError(
                f"TalkerStage expected TalkerInputPayload, got {type(payload).__name__}"
            )
        bridge = payload.bridge_states
        text_codes = list(payload.text_token_ids)
        if bridge is None or (isinstance(bridge, torch.Tensor) and bridge.numel() == 0):
            raise ValueError("TalkerStage received empty bridge hidden states")
        if not text_codes:
            raise ValueError("TalkerStage received empty text_token_ids")

        extra = getattr(sampling, "extra", None) or {}
        # ``watchdog_limit`` caps the number of talker decode steps the
        # way upstream's ``talker_max_steps_after_last_thinker_token``
        # does; unset means "replay every generated thinker row".
        watchdog_limit = int(extra["watchdog_limit"]) if extra.get("watchdog_limit") else None
        # Audio temperature is taken from the vendor default (0.2); the
        # text ``sampling.temperature`` is for the thinker stage.
        temperature = float(extra.get("audio_temperature", _AUDIO_VENDOR_TEMPERATURE))
        if temperature <= 1e-10:
            temperature = 1e-5

        bridge = bridge.unsqueeze(0).to(device=device, dtype=model.embed_proj[0].weight.dtype)
        spk_emb = payload.speaker_embedding
        # ``start_pos``/``num_steps`` arrive precomputed from thinker2talker
        # (prompt length / generated-row count); no second split here.
        from collections.abc import Mapping as _Mapping

        meta = dict(payload.metadata) if isinstance(payload.metadata, _Mapping) else {}
        bridge_len = bridge.shape[1]
        start_pos = int(meta.get("start_pos", len(payload.prompt_token_ids)))
        num_steps = int(meta.get("num_steps", bridge_len - start_pos))
        if start_pos >= bridge_len:
            # Degenerate payload (no generated rows): fall back to the
            # last row as the single decode position.
            start_pos = max(0, bridge_len - 1)
            num_steps = bridge_len - start_pos
        if watchdog_limit is not None:
            num_steps = min(num_steps, watchdog_limit)

        # ---- Staggered decode loop (mirrors vendor stream_generate) ----
        # Vendor ``step`` is the count of generated tokens BEFORE this
        # iteration's append, so the first iteration has ``step = 0`` and
        # ``audio_step = -1`` (every codebook pads). Codebook ``i`` starts
        # sampling real codes once ``audio_step >= i``.
        audio_codes: list[list[int]] = [[] for _ in range(8)]
        audio_stop_pos: list[int | None] = [None] * 8
        emitted_frames: list[list[int]] = []

        with torch.no_grad():
            for step in range(num_steps):
                audio_step = step - 1
                # Vendor's forward sees ``audio_buffer`` BEFORE this
                # iteration's append, so the sequence length is
                # ``start_pos + step`` — the very first forward covers
                # exactly the prompt.
                seq_len = start_pos + step
                # Audio buffer: prompt columns stay pad; generated column
                # ``start_pos + p`` holds ``audio_codes[i][p]`` for
                # ``i < min(p, 8)`` (vendor's diagonal backfill).
                buf = torch.full((1, 8, seq_len), _AUDIO_PAD_TOKEN, dtype=torch.long, device=device)
                for p in range(step):
                    lim = min(p, 8)
                    for i in range(lim):
                        buf[0, i, start_pos + p] = audio_codes[i][p]

                # Vendor recomputes the full sequence every step
                # (``use_cache=False`` on the parity path), so the talker
                # sees the whole prefix, not a sliding window.
                embeds, positions = model.prepare_inputs_embeds(
                    bridge[:, :seq_len, :], buf, spk_emb
                )

                # Fork ``Attention`` reads module-level ``Context`` to
                # pick prefill vs decode kernels. We always run the
                # prefill path because the talker recomputes the full
                # growing sequence every step. ``k_cache.numel() == 0``
                # on the freshly-loaded model so ``store_kvcache`` is
                # skipped — we don't need the paged storage here, only
                # the kernel.
                # Cache the arange base once per TalkerStage call so the
                # full-range slot/cu tensors share storage instead of
                # allocating 64 KB per decode step.
                if self._arange_cache is None or self._arange_cache.device != device:
                    self._arange_cache = torch.arange(32768, dtype=torch.int32, device=device)
                arange = self._arange_cache
                n_tokens = embeds.shape[0]
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=torch.tensor([0, n_tokens], dtype=torch.int32, device=device),
                    cu_seqlens_k=torch.tensor([0, n_tokens], dtype=torch.int32, device=device),
                    max_seqlen_q=n_tokens,
                    max_seqlen_k=n_tokens,
                    slot_mapping=arange[:n_tokens],
                    context_lens=None,
                    block_tables=None,
                )
                try:
                    hidden = model.body(positions, embeds)
                finally:
                    reset_context()
                logits_list = model.compute_logits(hidden)

                # Sample one code per codebook. HF ``model_omni.py``
                # stream_generate: ``logits_i = al[0,-1,:] / 0.2``, then
                # ``for prev in audio_codes[i][-3:]: logits_i[prev] /= 1.05``
                # (last-3 of THAT layer only), then ``topk(50)`` +
                # multinomial. The penalty is real in the HF reference —
                # it is only absent from the vLLM refactor.
                for i, logits in enumerate(logits_list):
                    if audio_step < i:
                        audio_codes[i].append(_AUDIO_PAD_TOKEN)
                        continue
                    # Match vendor exactly: keep the model dtype through
                    # temperature scaling and top-k, then draw with
                    # torch.multinomial from the 50-token softmax. Using the
                    # generic Gumbel sampler consumes a different RNG path;
                    # one different code recursively changes every later
                    # audio-buffer column.
                    logits_i = logits[-1, :].clone() / temperature
                    for previous_code in audio_codes[i][-3:]:
                        logits_i[previous_code] /= _AUDIO_REPETITION_PENALTY
                    top_value, top_index = logits_i.topk(50)
                    sampled = torch.multinomial(torch.softmax(top_value, dim=-1), 1)
                    code = top_index[sampled].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

                # Vendor break check runs BEFORE the append/emission.
                if audio_codes[7] and audio_codes[7][-1] == _AUDIO_STOP_TOKEN:
                    break

                # HF emission gate: ``audio_step >= 7`` with the diagonal
                # ``frame[i] = audio_codes[i][step - 7 + i]`` and the
                # active-layer filter ``stop_pos[i] is None or
                # step - 7 + i < stop_pos[i]`` over all 8 layers.
                if audio_step >= 7:
                    idx = [step - 7 + i for i in range(8)]
                    active = sum(
                        1
                        for i in range(8)
                        if audio_stop_pos[i] is None or idx[i] < audio_stop_pos[i]
                    )
                    if active >= 8:
                        emitted_frames.append([audio_codes[i][idx[i]] for i in range(8)])

        # ---- Pack output ----
        # ``emitted_frames`` mirrors the vendor: one row per emitted
        # frame, each row is the diagonal ``audio_codes[i][step - 7 + i]``.
        # Truncation is no longer needed because we only ever appended
        # full 8-code rows.
        if not emitted_frames:
            audio_codes_tensor = torch.zeros(0, 8, dtype=torch.long)
        else:
            audio_codes_tensor = torch.tensor(emitted_frames, dtype=torch.long)
        return TalkerOutput(audio_codes=audio_codes_tensor.cpu())


__all__ = [
    "ThinkerStage",
    "TalkerOutput",
    "TalkerStage",
]
