"""Stage wiring for MiniMind-O (the per-stage classes and the helpers
they need). Mirrors the ``stage.py`` convention used by
``models/sd_turbo/`` / ``smolvla/`` / ``smolvlm/``.

The fork-AR-Engine adapter layer (``StageRunner`` /
``_ensure_dist`` / ``get_stage_config`` /
``stage_kwargs_from_args``) lives in
:mod:`nanovllm_omni.engine.stage_runner`. This module owns
only the MiniMind-O specific bits:

  - weight slicing for the thinker-only safetensors
    (``_THINKER_KEEP_PREFIX`` / ``_thinker_only_dir`` /
    ``_resolve_model_path``)
  - HuggingFace tokenizer loader for the MiniMind-3o snapshot
  - ``ThinkerStage`` and ``TalkerStage`` high-level stage classes
    invoked by ``PipelineRunner``'s ``_stage_factory``.

Lazy load via factory: import this module only from inside the stage factory
functions in ``thinker.py`` — never at the top
of a sibling module. ``ModelRunner.__init__`` (pulled in by
``ThinkerStage``) drags in ``triton`` / ``flash-attn``,
which aren't available on CPU-only hosts.

Fork imports stay lazy — ``ModelRunner`` pulls in ``triton`` /
``flash-attn``, which isn't available on CPU-only hosts. The factory
functions (``_thinker_stage`` in ``thinker.py`` / ``_talker_stage``
here) trigger the heavy ``ModelRunner.__init__``; smoke
tests that only import factory symbols don't.

nano-vllm is installed as an editable package via ``uv``
(``[tool.uv.sources]`` in the root ``pyproject.toml``), so
``from nanovllm...`` resolves through the normal package mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm_omni.engine.stage_engine import StageEngine
from nanovllm_omni.engine.stage_runner import (
    StageRunner,
    _ensure_dist,
    get_stage_config,
    stage_kwargs_from_args,
)
from nanovllm_omni.models.minimind_omni.hf_config import register_minimind_config
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


def _resolve_model_path(args: Any) -> str:
    """Pull the model directory from ``OmniEngineArgs.model``.

    Fork ``Config.__post_init__`` asserts ``os.path.isdir(self.model)``.
    Resolves Hub IDs (``jingyaogong/minimind-3o``) to a local snapshot
    via ``_resolve_snapshot`` from ``bundle``. Local paths pass through.

    The directory is then shaped to the thinker: ``_thinker_only_dir``
    keeps only the keys ``MiniMindThinker`` consumes, because fork
    ``load_model`` walks every key and crashes on unknowns.
    """
    from .bundle import _resolve_snapshot

    model = getattr(args, "model", None)
    if not model:
        raise ValueError(
            "OmniEngineArgs.model is required to construct a StageRunner "
            "(fork Config validates the model path is a real directory)."
        )
    return _thinker_only_dir(_resolve_snapshot(model))


def _thinker_only_dir(src: str) -> str:
    """Build a sibling directory with only thinker-shaped weights.

    Keeps the ``model.`` prefix on the backbone keys: ``MiniMindThinker``
    holds its backbone at ``self.model`` (the upstream ``MiniMindForCausalLM``
    shape), so the fork loader looks up ``model.embed_tokens.weight`` /
    ``model.layers.*`` / ``model.norm.weight``. The bare ``lm_head.weight``
    stays bare because the thinker's head lives at ``self.lm_head``.
    """
    import glob
    import hashlib
    import os
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = os.path.abspath(src)
    # Part of the cache key so a differently shaped checkpoint is not reused.
    cache_key = hashlib.sha1(f"{src}\0model.".encode()).hexdigest()[:12]
    dst = os.path.join("/tmp", f"minimind-thinker-only-{cache_key}")
    target = os.path.join(dst, "model.safetensors")

    if os.path.isfile(target):
        return dst

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith((".json", ".py", ".jinja", ".md", ".txt")):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    def _name(k: str) -> str | None:
        if k.startswith("model."):
            return k
        # Top-level keys the thinker also consumes, e.g. ``lm_head.weight``
        # in the HF MiniMind checkpoint.
        if k.startswith(_THINKER_KEEP_PREFIX):
            return k
        return None

    def _accept(name: str) -> bool:
        stripped = name[len("model.") :] if name.startswith("model.") else name
        return stripped.startswith(_THINKER_KEEP_PREFIX)

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


def _load_tokenizer(model_path: str) -> Any:
    """Load a HuggingFace tokenizer from the model directory.

    The MiniMind snapshot ships a plain ``PreTrainedTokenizerFast``, so
    no remote code is involved.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


_THINKER_CHAT_MARKER = "<|im_start|>"


def _format_prompt(tokenizer: Any, prompt: str, open_thinking: bool) -> str:
    """Wrap a raw user prompt in the checkpoint chat template."""
    if _THINKER_CHAT_MARKER in prompt:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=open_thinking,
    )


class ThinkerStage(StageEngine):
    """Stage 0 — MiniMind thinker AR decode via fork ``ModelRunner``.

    The factory ``_thinker_stage`` (in ``thinker.py``) constructs this
    class; ``PipelineRunner`` then drives decoding by calling the
    instance with ``(payload, sampling)``.
    """

    def __init__(self, deploy: Any, args: Any, model_class: Any = None) -> None:
        super().__init__()
        from .thinker import MiniMindThinker

        self.deploy = deploy
        self.args = args
        self.model_class = model_class or MiniMindThinker
        # The decode graph is the default: on the RTX 3050 it cuts the
        # thinker stage from ~2250 ms to ~595 ms per request (ADR-0007),
        # and eager and graph produce byte-identical audio for the same
        # seed (tests/test_minimind_graph_parity.py). The fork falls back
        # to eager when capture fails, so a device that cannot capture
        # still runs.
        #
        # ``extra["use_thinker_cuda_graph"] = False`` opts back out; that
        # is the bench harness's comparison knob (ADR-0005 keeps the
        # choice at the stage, out of DeployConfig).
        extra = dict(getattr(args, "extra", None) or {})
        # The local config class replaces the hub's ``auto_map`` file, so
        # no remote code runs and the fork Config can ask for none.
        register_minimind_config()
        stage_kwargs = stage_kwargs_from_args(args)
        stage_kwargs["trust_remote_code"] = False
        # ``Config.max_capture_bs`` caps which decode batch sizes get a
        # CUDA graph. The fork captures them largest-first into one shared
        # memory pool, and a batch size captured alongside bigger ones
        # reduces in a different order than the eager path: the final
        # logits still agree, but the bridge rows (intermediate hidden
        # states) differ in the last bits, so graph and eager stop
        # producing byte-identical audio.
        #
        # Only the single-request graph is captured by default. That is the
        # configuration ADR-0005 and ADR-0007 measured and
        # ``tests/test_minimind_graph_parity.py`` pins, and it leaves the
        # bs=1 path byte-identical to that baseline. A batch (bs > 1) has no
        # graph and runs eager, which is slower per step than the graph but
        # is the only way to decode the batch at all without changing the
        # bs=1 arithmetic.
        #
        # ``extra["thinker_max_capture_bs"]`` raises the cap for a bench
        # cell: graphed batches decode ~4x faster on the thinker, at the
        # cost of the cross-bucket byte-identity above. The choice lives at
        # the stage, out of DeployConfig (ADR-0005).
        stage_kwargs["max_capture_bs"] = int(extra.get("thinker_max_capture_bs", 1))
        # Official eval_omni.py runs the released weights in fp16 (`.half()`).
        if stage_kwargs.get("dtype") is None:
            stage_kwargs["dtype"] = "float16"
        self.config = get_stage_config(
            model_path=_resolve_model_path(args),
            **stage_kwargs,
        )
        self.config.enforce_eager = not extra.get("use_thinker_cuda_graph", True)
        # Ensure NCCL process group exists before ModelRunner tries to init.
        _ensure_dist()
        self.stage_runner = StageRunner(
            model_class=self.model_class,
            config=self.config,
        )

    def __call__(self, payload: Any, sampling: Any) -> Any:
        """Decode one prompt end to end (direct-call form of ``step``)."""
        return self._decode([payload], [sampling])[0]

    def step(self) -> list[tuple[str, Any]]:
        """Decode every request this stage is holding in one batched pass."""
        if not self._pending:
            return []
        requests = list(self._pending)
        outputs = self._decode(
            [request.payload for request in requests],
            [request.sampling for request in requests],
        )
        # Clear only after the decode returns: a failure leaves the queue
        # intact so ``drop_pending`` can retire these requests instead of
        # leaving the runner waiting on them forever.
        self._pending.clear()
        return [
            (request.request_id, output) for request, output in zip(requests, outputs, strict=True)
        ]

    def _decode(self, payloads: list[Any], samplings: list[Any]) -> list[Any]:
        """Tokenize every prompt, run one batched decode, re-capture bridges."""
        import torch

        from .stage_processors import ThinkerStageOutput

        tokenizer = _load_tokenizer(_resolve_model_path(self.args))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = self.config.eos

        prepared = [
            self._prepare_decode(payload, sampling, tokenizer)
            for payload, sampling in zip(payloads, samplings, strict=True)
        ]
        decoded = decode_minimind(
            self.stage_runner,
            prepared,
            eos_id=eos_id,
            enter_token=_THINKER_ENTER_TOKEN,
            pad_token=_THINKER_PAD_TOKEN,
            talker_tail=_TALKER_TAIL,
        )

        outputs = []
        for request, result in zip(prepared, decoded, strict=True):
            bridge_hidden = recapture_bridge_minimind(
                self.stage_runner,
                request.token_ids + list(result.generated),
                result.bridge_hidden,
            )
            eos_index = next(
                (i for i, token in enumerate(result.generated) if token == eos_id), None
            )
            content_len = len(result.generated) if eos_index is None else eos_index
            outputs.append(
                ThinkerStageOutput(
                    bridge_states=bridge_hidden if bridge_hidden is not None else torch.empty(0),
                    prompt_token_ids=tuple(request.token_ids),
                    output_token_ids=tuple(result.generated),
                    text_token_ids=tuple(request.token_ids + result.generated),
                    text=tokenizer.decode(result.generated[:content_len], skip_special_tokens=True),
                    text_state=result.text_state,
                    request_id=getattr(request, "request_id", None),
                    metadata={"content_len": content_len, "talker_tail": _TALKER_TAIL},
                )
            )
        return outputs

    def _prepare_decode(
        self,
        payload: Any,
        sampling: Any,
        tokenizer: Any,
    ) -> Any:
        """Turn one prompt + ``SamplingParams`` into a ``ThinkerDecode``."""
        from .stage_runner import ThinkerDecode

        open_thinking = bool((getattr(sampling, "extra", None) or {}).get("open_thinking", False))
        prompt = _format_prompt(tokenizer, str(payload), open_thinking)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or 0]

        max_tokens = getattr(sampling, "max_tokens", None)
        if max_tokens is None:
            max_tokens = (getattr(sampling, "extra", None) or {}).get("max_tokens", 512)

        temperature = getattr(sampling, "temperature", None)
        if temperature is None:
            temperature = (getattr(sampling, "extra", None) or {}).get("temperature", 0.7)
        temperature = float(temperature)
        if temperature <= 1e-10:
            temperature = 1e-5

        stop_strings = list(getattr(sampling, "stop", None) or [])

        def _stop_hit(ids: list[int]) -> bool:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return any(s in text for s in stop_strings)

        return ThinkerDecode(
            token_ids=token_ids,
            max_tokens=int(max_tokens),
            temperature=temperature,
            repetition_penalty=float(
                (getattr(sampling, "extra", None) or {}).get("repetition_penalty", 1.05) or 1.05
            ),
            top_p=float(getattr(sampling, "top_p", 1.0) or 1.0),
            seed=getattr(sampling, "seed", None),
            stop_hit=_stop_hit if stop_strings else None,
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
# The vendor talker needs post-EOS filler rows to reach its own stop token;
# 64 covers the 56 frames measured for the teacher-forced reference.
_TALKER_TAIL = 64


def _can_emit_frame(stop_positions: list[int | None], step: int) -> bool:
    """Whether the emission gate opens at ``step`` or at any later step.

    The gate needs all eight codebooks active, ``idx[i] = step - 7 + i``.
    ``idx`` only grows, so a codebook that stays inactive now stays inactive
    forever: a closed gate is permanent, which is what lets the decode loop
    stop early without changing the emitted frames.
    """
    return all(stop_positions[i] is None or step - 7 + i < stop_positions[i] for i in range(8))


class TalkerStage(StageEngine):
    """Stage 1 — MiniMind talker decode on the vendor ``TalkerModule`` (full sequence per step)."""

    def __init__(self, deploy: Any, args: Any) -> None:
        super().__init__()
        self.deploy = deploy
        self.args = args
        self._model: Any = None
        self._device: str | None = None
        self._config: Any = None

    def _ensure_model(self) -> Any:
        """Load the vendor talker from the original checkpoint."""
        if self._model is not None:
            return self._model

        import os

        import torch
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        from .bundle import _resolve_snapshot

        model_dir = _resolve_snapshot(getattr(self.args, "model", None))
        config_cls = get_class_from_dynamic_module("model_omni.OmniConfig", model_dir)
        talker_cls = get_class_from_dynamic_module("model_omni.TalkerModule", model_dir)
        config = config_cls.from_pretrained(model_dir)

        talker = talker_cls(config).eval()
        weights = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(weights):
            state = torch.load(weights, map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file

            state = {}
            for name in sorted(os.listdir(model_dir)):
                if name.endswith(".safetensors"):
                    state.update(load_file(os.path.join(model_dir, name)))
        talker.load_state_dict(
            {
                key[len("talker.") :]: value
                for key, value in state.items()
                if key.startswith("talker.")
            },
            strict=True,
        )
        talker = talker.half().to("cuda")

        self._model = talker
        self._config = talker.talker_config
        self._device = "cuda"
        return talker

    def _talker_logits(self, bridge: Any, audio_codes: Any, spk_emb: Any) -> Any:
        """Run the vendor talker over the full sequence; return its 8 logit tensors."""
        import torch

        talker = self._model
        embeds = talker.embed_tokens(audio_codes)
        if spk_emb is not None:
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            mask = (audio_codes[:, 0, :] == _AUDIO_SPK_TOKEN).unsqueeze(-1)
            embeds = torch.where(mask, talker.spk_proj(spk_emb).unsqueeze(1), embeds)
        hidden = (
            talker.embed_proj(bridge) * talker.text_scale
            + talker.codec_proj(embeds) * talker.audio_scale
        )
        seq_len = hidden.shape[1]
        position_embeddings = (talker.freqs_cos[:seq_len], talker.freqs_sin[:seq_len])
        for layer in talker.layers:
            hidden, _ = layer(
                hidden,
                position_embeddings,
                past_key_value=None,
                use_cache=False,
            )
        hidden = talker.norm(hidden)
        return talker.lm_head(hidden)

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
        # Seed here as well as in the thinker: the talker's own sampling is
        # the last RNG consumer, and the first request in a process also
        # consumes RNG building the talker weights (``_ensure_model`` above).
        # Without this, the same (prompt, seed) yields different audio
        # depending on how many requests the process served before.
        # Mirrors ``models/smolvlm/stage.py``.
        seed = getattr(sampling, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
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

        # Audio buffer: prompt columns stay pad; generated column
        # ``start_pos + p`` holds ``audio_codes[i][p]`` for ``i < min(p, 8)``
        # (vendor's diagonal backfill). Column ``start_pos + p`` is final as
        # soon as codebook ``p - 1`` has sampled step ``p``, and
        # ``audio_codes[i][p]`` never changes after it is appended — so each
        # step writes only the column that just became final instead of
        # rebuilding the whole buffer. Same values either way; ADR 0007 has
        # the timings that motivated this.
        buf = torch.full(
            (1, 8, start_pos + num_steps),
            _AUDIO_PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            for step in range(num_steps):
                audio_step = step - 1
                # Vendor's forward sees ``audio_buffer`` BEFORE this
                # iteration's append, so the sequence length is
                # ``start_pos + step`` — the very first forward covers
                # exactly the prompt.
                seq_len = start_pos + step
                column = step - 1
                if column >= 0:
                    filled = min(column, 8)
                    if filled:
                        buf[0, :filled, start_pos + column] = torch.tensor(
                            [audio_codes[i][column] for i in range(filled)],
                            dtype=torch.long,
                            device=device,
                        )

                # Vendor recomputes the full sequence every step
                # (``use_cache=False`` on the parity path), so the talker
                # sees the whole prefix, not a sliding window.
                logits_list = self._talker_logits(
                    bridge[:, :seq_len, :], buf[:, :, :seq_len], spk_emb
                )

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
                    logits_i = logits[0, -1, :].clone() / temperature
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
                    if _can_emit_frame(audio_stop_pos, step):
                        idx = [step - 7 + i for i in range(8)]
                        emitted_frames.append([audio_codes[i][idx[i]] for i in range(8)])
                    else:
                        # A closed gate never reopens, so every remaining step
                        # would be a full-sequence forward with no observable
                        # effect. Stopping here keeps the frames byte-identical.
                        break

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


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory — the vendor talker, loaded from the full checkpoint."""
    return TalkerStage(deploy, args)


__all__ = [
    "TalkerStage",
    "TalkerOutput",
    "ThinkerStage",
    "_talker_stage",
]
