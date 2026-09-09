"""Stage engine wiring for MiniMind-O (MiniMind-O specific bits only).

The fork-AR-Engine adapter layer (``SharedBlockManager`` /
``StageRunner`` / ``_ensure_dist`` / ``get_stage_config`` /
``stage_kwargs_from_args``) lives in
:mod:`nanovllm_omni.engine.stage_runner` (extracted so the second model
family that needs the same fork-AR-Engine adapter can reuse it). This
module owns
only the MiniMind-O specific bits:

  - weight slicing for thinker-only / talker-only safetensors
    (``_THINKER_KEEP_PREFIX`` / ``_TALKER_KEEP_PREFIX`` /
    ``_filter_only_dir`` / ``_resolve_model_path``)
  - HuggingFace tokenizer loader for the MiniMind-3o snapshot
  - ``ThinkerStage`` and ``TalkerStage`` high-level stage classes
    invoked by ``PipelineRunner``'s ``_stage_factory``.

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
    SharedBlockManager,
    StageRunner,
    _ensure_dist,
    get_shared_block_manager,
    get_stage_config,
    stage_kwargs_from_args,
)

# ---------------------------------------------------------------------------
# ThinkerStage / TalkerStage.
# ---------------------------------------------------------------------------


_THINKER_KEEP_PREFIX = ("embed_tokens.", "layers.", "norm.", "lm_head.")
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

    def __init__(self, deploy: Any, args: Any) -> None:
        from .thinker import MiniMindThinker

        self.deploy = deploy
        self.args = args
        # ``enforce_eager=True`` so the thinker's bridge-layer hidden
        # state can be captured as a plain Python attribute inside
        # ``forward``. Direct forward is required because CUDA-graph
        # replay does not re-execute Python attribute assignments; a
        # registered-buffer side output doesn't survive cleanly across
        # replay (variable per-step bridge length). Eager direct
        # forward is the simplest reliable path here.
        self.config = get_stage_config(
            "thinker",
            model_path=_resolve_model_path(args, stage="thinker"),
            **stage_kwargs_from_args(args),
            enforce_eager=True,
        )
        # The first ``get_shared_block_manager`` call wins; the
        # talker's later call reuses this same instance.
        self.shared_block_manager = get_shared_block_manager(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
        )
        # Ensure NCCL process group exists before ModelRunner tries to init.
        _ensure_dist()
        self.stage_runner = StageRunner(
            model_class=MiniMindThinker,
            config=self.config,
            shared_block_manager=self.shared_block_manager,
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
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

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

        # Create fork Sequence and Scheduler.
        fork_sp = ForkSamplingParams(
            temperature=temperature, max_tokens=max_tokens, ignore_eos=True
        )
        scheduler = Scheduler(self.config)
        sequence = Sequence(token_ids, fork_sp)
        scheduler.add(sequence)

        model = self.stage_runner.model_runner.model
        runner = self.stage_runner.model_runner
        generated: list[int] = []
        bridge_hidden: torch.Tensor | None = None

        while not scheduler.is_finished():
            seqs, is_prefill = scheduler.schedule()
            torch.cuda.synchronize()
            input_ids, positions = (
                runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
            )
            temperatures = runner.prepare_sample(seqs)
            logits = runner.run_model(input_ids, positions, is_prefill)

            # Extract bridge hidden; accumulate across prefill + each decode step.
            bh = model.get_bridge_hidden()
            if bh is not None:
                bh = bh.detach().clone()
                if is_prefill:
                    bridge_hidden = bh
                elif bridge_hidden is not None:
                    bridge_hidden = torch.cat([bridge_hidden, bh], dim=0)

            token_id_list = runner.sampler(logits, temperatures).tolist()
            scheduler.postprocess(seqs, token_id_list, is_prefill)
            self.stage_runner.reset_context()

            new_id = token_id_list[0]
            generated.append(new_id)

            # Stop on EOS or max_tokens reached.
            if (
                not sequence.ignore_eos and new_id == self.config.eos
            ) or sequence.num_completion_tokens >= max_tokens:
                break

        # Extract text_state (final hidden from last forward).
        text_state = logits[0].detach() if logits is not None else None

        # Clone bridge_hidden so it survives the talker's model load.
        if bridge_hidden is not None:
            bridge_hidden = bridge_hidden.clone()

        return ThinkerStageOutput(
            bridge_states=bridge_hidden if bridge_hidden is not None else torch.empty(0),
            prompt_token_ids=tuple(token_ids),
            output_token_ids=tuple(generated),
            text_token_ids=tuple(token_ids + generated),
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
_AUDIO_SPK_TOKEN = 2051
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

    Sampling goes through fork ``Sampler`` with ``top_k=50`` and a
    repetition penalty on the last 3 codes per codebook, matching
    the vendor recipe. ``enforce_eager`` for the underlying fork
    flow is not applicable here because we don't go through a
    ``ModelRunner``; we set up the fork ``Context`` directly.
    """

    def __init__(self, deploy: Any, args: Any) -> None:
        from nanovllm.layers.sampler import Sampler

        self.deploy = deploy
        self.args = args
        self._model: Any = None
        self._sampler = Sampler()
        self._device: str | None = None
        self._config: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the talker model from the talker-only safetensors."""
        if self._model is not None:
            return self._model

        import torch
        from nanovllm.config import Config
        from nanovllm.utils.loader import load_model

        from .talker import MiniMindTalker

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
            model = MiniMindTalker(hf_cfg)
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
        sampler = self._sampler
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
        max_steps = min(
            len(text_codes),
            int(extra["watchdog_limit"]) if extra.get("watchdog_limit") else len(text_codes),
        )
        # Audio temperature is taken from the vendor default (0.2); the
        # text ``sampling.temperature`` is for the thinker stage.
        temperature = float(extra.get("audio_temperature", _AUDIO_VENDOR_TEMPERATURE))
        if temperature <= 1e-10:
            temperature = 1e-5

        bridge = bridge.unsqueeze(0).to(device=device, dtype=model.embed_proj[0].weight.dtype)
        spk_emb = payload.speaker_embedding

        # ---- Staggered decode loop (matches vendor stream_generate) ----
        # ``audio_codes[i]`` lags codebook i by i+1 steps — i.e. at
        # ``step=s`` we have sampled ``s - i`` real codes for codebook
        # ``i`` (or zero when ``s <= i``). Vendor uses pad for the
        # lag positions; the model learned this pattern.
        audio_codes: list[list[int]] = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8

        with torch.no_grad():
            for step in range(1, max_steps + 1):
                t_audio = step  # vendor invariant: t_audio grows 1:1 with text
                # Build audio_codes buffer for this step: [1, 8, t_audio].
                buf = torch.full((1, 8, t_audio), _AUDIO_PAD_TOKEN, dtype=torch.long, device=device)
                audio_step = step - 1
                for i in range(8):
                    # Only fill positions we have real codes for.
                    available = len(audio_codes[i])
                    fill = min(audio_step + 1, i + 1, available)
                    if fill > 0:
                        buf[0, i, :fill] = torch.tensor(
                            audio_codes[i][:fill], dtype=torch.long, device=device
                        )

                # Prep + body via the decomposed model API. ``prepare_inputs_embeds``
                # also returns the (possibly shifted) ``positions`` when
                # ``spk_emb`` is provided; without a speaker, positions
                # stays as 0..t_audio-1.
                embeds, positions = model.prepare_inputs_embeds(
                    bridge[:, :t_audio, :], buf, spk_emb
                )

                # Fork ``Attention`` reads module-level ``Context`` to
                # pick prefill vs decode kernels. We always run the
                # prefill path because the talker recomputes the full
                # growing sequence every step. ``k_cache.numel() == 0``
                # on the freshly-loaded model so ``store_kvcache`` is
                # skipped — we don't need the paged storage here, only
                # the kernel.
                n_tokens = embeds.shape[0]
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=torch.tensor([0, n_tokens], dtype=torch.int32, device=device),
                    cu_seqlens_k=torch.tensor([0, n_tokens], dtype=torch.int32, device=device),
                    max_seqlen_q=n_tokens,
                    max_seqlen_k=n_tokens,
                    slot_mapping=torch.arange(n_tokens, dtype=torch.int32, device=device),
                    context_lens=None,
                    block_tables=None,
                )
                try:
                    hidden = model.body(positions, embeds)
                finally:
                    reset_context()
                logits_list = model.compute_logits(hidden)

                # Sample one code per codebook. We pass the temperature
                # via the fork Sampler's own kwargs so the math lives
                # in one place; top-50 + last-3-codes repetition penalty
                # mirror the vendor recipe.
                temperature_tensor = torch.tensor(
                    [temperature], dtype=model.embed_proj[0].weight.dtype, device=device
                )
                for i, logits in enumerate(logits_list):
                    if audio_step < i:
                        audio_codes[i].append(_AUDIO_PAD_TOKEN)
                        continue
                    last = logits[-1:, :]  # [1, vocab]
                    recent = audio_codes[i][-3:]
                    history = (
                        torch.tensor(recent, dtype=torch.long, device=device) if recent else None
                    )
                    code = sampler(
                        last,
                        temperature_tensor,
                        top_k=50,
                        history=history,
                        repetition_penalty=_AUDIO_REPETITION_PENALTY,
                    ).item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

                # Stop once codebook 7 emits the stop token.
                if audio_codes[7] and audio_codes[7][-1] == _AUDIO_STOP_TOKEN:
                    break

        # ---- Pack output ----
        # audio_codes: list of 8 lists, each length = max_steps (or fewer
        # if we broke out early). Truncate to the shortest codebook to
        # form a square [frames, 8] tensor.
        n_frames = min(len(c) for c in audio_codes)
        if n_frames == 0:
            audio_codes_tensor = torch.zeros(1, 8, dtype=torch.long)
        else:
            audio_codes_tensor = (
                torch.tensor([c[:n_frames] for c in audio_codes], dtype=torch.long, device=device)
                .t()
                .contiguous()
            )  # [frames, 8]
        return TalkerOutput(audio_codes=audio_codes_tensor.cpu())


__all__ = [
    "ThinkerStage",
    "TalkerOutput",
    "TalkerStage",
    # Re-exported from engine.stage_runner for backward compat
    "SharedBlockManager",
    "StageRunner",
    "get_shared_block_manager",
    "get_stage_config",
]
