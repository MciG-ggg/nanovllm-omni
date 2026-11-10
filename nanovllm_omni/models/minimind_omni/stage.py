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
    ``_resolve_model_path``) and the converted ``mlx_lm`` directory
    (``_resolve_mlx_model_path``)
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

from nanovllm_omni.backend import get_backend
from nanovllm_omni.engine.stage_engine import StageEngine
from nanovllm_omni.engine.stage_runner import _ensure_dist, stage_kwargs_from_args
from nanovllm_omni.models.minimind_omni.hf_config import register_minimind_config
from nanovllm_omni.models.minimind_omni.stage_runner import (
    decode_minimind,
    decode_minimind_mlx,
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


def _stage_backend(args: Any, extra: dict[str, Any]) -> str:
    """Which backend this stage runs on.

    ``args.backend`` is the declared knob (``OmniEngineArgs.backend``, also
    settable per stage in the deploy YAML); ``extra["backend"]`` stays as the
    older escape hatch so existing callers keep working. The declared value
    wins, because that is the one the argument validator checked.
    """
    declared = getattr(args, "backend", None)
    if declared:
        return str(declared)
    return str(extra.get("backend", "torch"))


def _resolve_mlx_model_path(args: Any, subdir: str = "mlx-thinker") -> str:
    """A converted MLX directory next to the checkpoint.

    ``scripts/convert_minimind_thinker_mlx.py`` writes ``<checkpoint>/mlx-thinker``
    and ``scripts/convert_minimind_talker_mlx.py`` writes
    ``<checkpoint>/mlx-talker``; the MLX backend loads those directories
    instead of the torch checkpoint.
    """
    import os

    from .bundle import _resolve_snapshot

    model = getattr(args, "model", None)
    if not model:
        raise ValueError("OmniEngineArgs.model is required to construct a StageRunner")
    path = os.path.join(_resolve_snapshot(model), subdir)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no {subdir} at {path}; run scripts/convert_minimind_{subdir.split('-')[-1]}_mlx.py"
        )
    return path


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

    def _keep(k: str) -> str | None:
        """The key to write, or None when the thinker does not consume it.

        Keeps the ``model.`` backbone prefix on the way out; bare top-level
        keys (``lm_head.weight`` in the HF MiniMind checkpoint) pass through.
        """
        stripped = k[len("model.") :] if k.startswith("model.") else k
        return k if stripped.startswith(_THINKER_KEEP_PREFIX) else None

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
            name = _keep(k)
            if name is None:
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
                name = _keep(k)
                if name is None:
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

    ``extra["backend"]`` picks the track: ``"torch"`` (default) runs the
    fork CUDA path, ``"mlx"`` runs the Apple Silicon path.
    """

    def __init__(self, deploy: Any, args: Any, model_class: Any = None) -> None:
        super().__init__()
        self.deploy = deploy
        self.args = args
        extra = dict(getattr(args, "extra", None) or {})
        self._backend_name = _stage_backend(args, extra)
        if self._backend_name == "mlx":
            # The MLX runner loads an ``mlx_lm`` directory and has no
            # torch model class.
            self.model_class = model_class
            model_path = _resolve_mlx_model_path(args)
        else:
            from .thinker import MiniMindThinker

            self.model_class = model_class or MiniMindThinker
            model_path = _resolve_model_path(args)
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
        backend_kwargs = {"dtype": stage_kwargs["dtype"]}
        if self._backend_name != "mlx":
            # ``args.device`` is a torch knob and reads "cpu" on a host
            # without CUDA, while MLX runs on the unified-memory GPU.
            backend_kwargs["device"] = getattr(args, "device", None)
        backend = get_backend(self._backend_name, **backend_kwargs)
        self.config = backend.make_config(
            model_path=model_path,
            **stage_kwargs,
        )
        if self._backend_name != "mlx":
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
            self.config.enforce_eager = not extra.get("use_thinker_cuda_graph", True)
            # Ensure NCCL process group exists before ModelRunner tries to init.
            _ensure_dist()
        self.stage_runner = backend.make_stage_runner(
            model_class=self.model_class,
            config=self.config,
        )
        self._tokenizer_path = model_path

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

        tokenizer = _load_tokenizer(self._tokenizer_path)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.config, "eos", None)

        prepared = [
            self._prepare_decode(payload, sampling, tokenizer)
            for payload, sampling in zip(payloads, samplings, strict=True)
        ]
        if self._backend_name == "mlx":
            decoded = decode_minimind_mlx(
                self.stage_runner,
                prepared,
                eos_id=eos_id,
                enter_token=_THINKER_ENTER_TOKEN,
                pad_token=_THINKER_PAD_TOKEN,
                talker_tail=_TALKER_TAIL,
            )
            # Every forwarded position contributes one bridge row; the last
            # sampled token is never forwarded, so it has none (ADR 0025).
            bridges = [result.bridge_hidden for result in decoded]
        else:
            decoded = decode_minimind(
                self.stage_runner,
                prepared,
                eos_id=eos_id,
                enter_token=_THINKER_ENTER_TOKEN,
                pad_token=_THINKER_PAD_TOKEN,
                talker_tail=_TALKER_TAIL,
            )
            bridges = [
                recapture_bridge_minimind(
                    self.stage_runner,
                    request.token_ids + list(result.generated),
                    result.bridge_hidden,
                )
                for request, result in zip(prepared, decoded, strict=True)
            ]

        outputs = []
        for request, result, bridge_hidden in zip(prepared, decoded, bridges, strict=True):
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
# The vendor talker needs post-EOS filler rows to reach its own stop token;
# 64 covers the 56 frames measured for the teacher-forced reference.
_TALKER_TAIL = 64


class TalkerStage(StageEngine):
    """Stage 1 — MiniMind talker decode; dense rewrite by default, vendor arm selectable."""

    def __init__(self, deploy: Any, args: Any) -> None:
        super().__init__()
        self.deploy = deploy
        self.args = args
        self._runner: Any = None
        # ADR 0020: the checkpoint stays as weights + parity reference. The
        # vendor class is the P1/P2 reference arm, not the default.
        extra = getattr(args, "extra", None) or {}
        self._impl = str(extra.get("talker_impl", "rewritten"))
        # ADR 0017 windowed KV stays opt-in until its parity gate is green on
        # CUDA (ADR 0022); the dense path is the one with a measured baseline.
        self._kv = str(extra.get("talker_kv", "dense"))
        if self._kv not in ("dense", "windowed"):
            raise ValueError(f"unknown talker_kv {self._kv!r}; expected 'dense' or 'windowed'")
        if self._kv == "windowed" and self._impl != "rewritten":
            raise ValueError("talker_kv='windowed' requires talker_impl='rewritten'")

    def _ensure_runner(self) -> Any:
        """Build the talker runner for the selected backend."""
        if self._runner is not None:
            return self._runner

        from .bundle import _resolve_snapshot

        extra = getattr(self.args, "extra", None) or {}
        backend_name = _stage_backend(self.args, extra)
        backend_kwargs: dict[str, Any] = {"dtype": getattr(self.args, "dtype", None)}
        if backend_name == "mlx":
            # The MLX runner loads the converted directory and runs on the
            # unified-memory GPU, so the torch device does not apply.
            model_path = _resolve_mlx_model_path(self.args, "mlx-talker")
        else:
            backend_kwargs["device"] = getattr(self.args, "device", None)
            model_path = _resolve_snapshot(getattr(self.args, "model", None))
        backend = get_backend(backend_name, **backend_kwargs)
        self._runner = backend.make_talker_runner(
            model_path,
            impl=self._impl,
            kv=self._kv,
        )
        return self._runner

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

        from ...backend.base import TalkerRequest
        from ...backend.talker_recipe import (
            AUDIO_REPETITION_PENALTY,
            AUDIO_VENDOR_TEMPERATURE,
            NUM_CODEBOOKS,
        )
        from .stage_processors import TalkerInputPayload

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
        seed = getattr(sampling, "seed", None)
        # Audio temperature is taken from the vendor default (0.2); the
        # text ``sampling.temperature`` is for the thinker stage.
        temperature = float(extra.get("audio_temperature", AUDIO_VENDOR_TEMPERATURE))
        if temperature <= 1e-10:
            temperature = 1e-5
        # ``watchdog_limit`` caps the number of talker decode steps the
        # way upstream's ``talker_max_steps_after_last_thinker_token``
        # does; unset means "replay every generated thinker row".
        watchdog_limit = int(extra["watchdog_limit"]) if extra.get("watchdog_limit") else None

        # ``start_pos``/``num_steps`` arrive precomputed from thinker2talker
        # (prompt length / generated-row count); no second split here.
        from collections.abc import Mapping as _Mapping

        meta = dict(payload.metadata) if isinstance(payload.metadata, _Mapping) else {}
        bridge_len = bridge.shape[-2]
        start_pos = int(meta.get("start_pos", len(payload.prompt_token_ids)))
        num_steps = int(meta.get("num_steps", bridge_len - start_pos))
        if start_pos >= bridge_len:
            # Degenerate payload (no generated rows): fall back to the
            # last row as the single decode position.
            start_pos = max(0, bridge_len - 1)
            num_steps = bridge_len - start_pos
        if watchdog_limit is not None:
            num_steps = min(num_steps, watchdog_limit)

        runner = self._ensure_runner()
        frames = runner.generate(
            TalkerRequest(
                bridge=bridge,
                num_steps=num_steps,
                start_pos=start_pos,
                temperature=temperature,
                seed=seed,
                speaker_embedding=payload.speaker_embedding,
                repetition_penalty=AUDIO_REPETITION_PENALTY,
            )
        )

        # ``frames`` mirrors the vendor: one row per emitted frame, each row
        # is the diagonal ``audio_codes[i][step - 7 + i]``. Truncation is
        # not needed because only full 8-code rows are ever appended.
        if not frames:
            audio_codes_tensor = torch.zeros(0, NUM_CODEBOOKS, dtype=torch.long)
        else:
            audio_codes_tensor = torch.tensor(frames, dtype=torch.long)
        return TalkerOutput(audio_codes=audio_codes_tensor.cpu())


def _talker_stage(deploy: Any, args: Any) -> Any:
    """Stage 1 factory — the talker selected by ``extra["talker_impl"]``."""
    return TalkerStage(deploy, args)


__all__ = [
    "TalkerStage",
    "TalkerOutput",
    "ThinkerStage",
    "_talker_stage",
]
