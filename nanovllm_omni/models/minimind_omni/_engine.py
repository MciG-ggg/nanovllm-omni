"""Stage engine wiring for MiniMind-O via the fork ``ModelRunner``.

Internal module — implements ADR-002 (each stage owns one fork
``ModelRunner``; both stages share one ``SharedBlockManager`` so
logical block IDs are consistent while each stage holds its own
physical KV tensors). Public surface:

  - ``SharedBlockManager``  — thin wrapper over fork ``BlockManager``.
  - ``StageRunner``         — one fork ``ModelRunner`` + the
                              set/forward/sample/reset lifecycle the
                              fork exposes (made explicit per ADR-002).
  - ``ThinkerStage`` / ``TalkerStage`` — high-level stage classes.
                              ``PipelineRunner`` invokes these via
                              ``__call__(payload, sampling)``.

Fork imports are deliberately lazy: the fork submodule requires
``triton`` / ``flash-attn`` to import ``ModelRunner``, which is not
available on a CPU-only host. The factory functions (``_thinker_stage``
/ ``_talker_stage`` in ``thinker.py`` / ``talker.py``) call the heavy
``ModelRunner`` constructor; smoke tests that only import the factory
function symbols don't trigger the fork import.

nano-vllm is installed as an editable package via ``uv``
(``[tool.uv.sources]`` in the root ``pyproject.toml``), so ``from nanovllm...``
resolves through the normal package mechanism.

Why a module-level cache for ``SharedBlockManager``: ``ThinkerStage`` and
``TalkerStage`` are constructed by independent factory calls in
``PipelineRunner._ensure_stages``. Without a shared handle the second
stage would mint a fresh ``BlockManager`` and the two stages' block
IDs would collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# SharedBlockManager.
# ---------------------------------------------------------------------------


class SharedBlockManager:
    """Thin wrapper over fork ``BlockManager``.

    ADR-002: the *logical* block table (which block ID holds which
    tokens) is shared across stages; the *physical* KV tensors are
    per-stage because layer counts differ (thinker 12 layers, talker
    4 layers — incompatible tensor shapes). Fork ``BlockManager``
    only tracks logical block IDs; the per-stage KV tensors are
    allocated by each stage's ``ModelRunner`` independently.

    ponytail: single instance shared across two stages. If we ever
    run stages concurrently in different threads, add per-stage
    ``BlockManager`` instances + a mapping layer.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        from nanovllm.engine.block_manager import BlockManager

        self._bm = BlockManager(num_blocks, block_size)

    @property
    def block_size(self) -> int:
        return self._bm.block_size

    def can_allocate(self, seq: Any) -> int:
        return self._bm.can_allocate(seq)

    def allocate(self, seq: Any, num_cached_blocks: int) -> None:
        self._bm.allocate(seq, num_cached_blocks)

    def deallocate(self, seq: Any) -> None:
        self._bm.deallocate(seq)

    def can_append(self, seq: Any) -> bool:
        return self._bm.can_append(seq)

    def may_append(self, seq: Any) -> None:
        self._bm.may_append(seq)

    def hash_blocks(self, seq: Any) -> None:
        self._bm.hash_blocks(seq)


# Module-level handle so the two stage factories can rendezvous on a
# single ``SharedBlockManager``. The first caller (ThinkerStage,
# constructed first per ``PipelineRunner._ensure_stages`` order) wins;
# later callers reuse the same instance regardless of the ``num_blocks``
# they pass. Mismatches are surfaceable via ``reset_shared_block_manager``
# in tests.

_shared_block_manager: SharedBlockManager | None = None


def get_shared_block_manager(num_blocks: int, block_size: int) -> SharedBlockManager:
    """Return the process-wide ``SharedBlockManager`` (lazy-init).

    ``num_blocks`` / ``block_size`` are taken from the first caller's
    ``Config``. Both stages use the same ``block_size`` (fork default
    256); ``num_blocks`` is set per-stage by ``ModelRunner.allocate_kv_cache``
    based on per-stage ``gpu_memory_utilization``, so the two stages'
    values may differ. We accept the first caller's value; the second
    stage's own ``ModelRunner`` will still allocate its own KV tensors
    sized to the right per-stage ``num_kvcache_blocks``. Sharing the
    BlockManager is about logical block-ID uniqueness, not KV sizing.
    """
    global _shared_block_manager
    if _shared_block_manager is None:
        _shared_block_manager = SharedBlockManager(num_blocks, block_size)
    return _shared_block_manager


def reset_shared_block_manager() -> None:
    """Drop the cached ``SharedBlockManager`` (test hook)."""
    global _shared_block_manager
    _shared_block_manager = None


# ---------------------------------------------------------------------------
# Per-stage Config cache.
# ---------------------------------------------------------------------------
#
# Fork ``Config`` is a dataclass whose ``__post_init__`` reads
# ``hf_config`` via ``AutoConfig.from_pretrained`` and asserts the
# model path is a real directory. Each stage needs its own ``Config``
# because ``gpu_memory_utilization`` is per-stage (thinker 0.6, talker
# 0.2 in the deploy YAML). Caching by stage name so repeat factory
# invocations reuse the same ``Config`` object.

_stage_configs: dict[str, Any] = {}


def get_stage_config(stage_name: str, model_path: str, **kwargs: Any) -> Any:
    """Return the fork ``Config`` for one stage (lazy, cached).

    Fork ``Config.__post_init__`` calls ``AutoConfig.from_pretrained``
    without ``trust_remote_code``. MiniMind's config.json has ``auto_map``,
    so we patch that one call. After load, coerce ``hf_config.dtype`` to
    a ``torch.dtype`` — ModelRunner does ``torch.set_default_dtype(hf_config.dtype)``.
    """
    if stage_name not in _stage_configs:
        import torch
        from nanovllm.config import Config
        from transformers import AutoConfig

        orig = AutoConfig.from_pretrained

        def _trusted(*args, **kw):
            kw.setdefault("trust_remote_code", True)
            return orig(*args, **kw)

        AutoConfig.from_pretrained = _trusted  # type: ignore[method-assign]
        try:
            cfg = Config(model=model_path, **kwargs)
        finally:
            AutoConfig.from_pretrained = orig  # type: ignore[method-assign]

        dt = getattr(cfg.hf_config, "dtype", None)
        if isinstance(dt, str):
            cfg.hf_config.dtype = getattr(torch, dt, torch.float16)
        elif dt is None:
            cfg.hf_config.dtype = getattr(cfg.hf_config, "torch_dtype", None) or torch.float16
        _stage_configs[stage_name] = cfg
    return _stage_configs[stage_name]


def reset_stage_configs() -> None:
    """Drop the cached stage ``Config`` objects (test hook)."""
    _stage_configs.clear()


# ---------------------------------------------------------------------------
# Fork helpers: dist init + Triton KV store patch.
# ---------------------------------------------------------------------------


_DIST_PATCHED = False


def _ensure_dist() -> None:
    """Initialize the fork's NCCL process group once, monkey-patch subsequent calls.

    The fork's ``ModelRunner.__init__`` unconditionally calls
    ``dist.init_process_group("nccl", ...)``.  Creating two
    ``ModelRunner`` instances (thinker + talker) would crash on the
    second call.  We monkey-patch ``dist.init_process_group`` to a
    no-op when ``dist.is_initialized()``, so the first ``ModelRunner``
    goes through and the second's call is harmless.
    """
    global _DIST_PATCHED
    if _DIST_PATCHED:
        return
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # Monkey-patch dist.init_process_group so fork ModelRunner.__init__
    # does not crash on the second instance.
    _orig_init = dist.init_process_group

    def _safe_init(*args, **kwargs):
        if dist.is_initialized():
            return None
        return _orig_init(*args, **kwargs)

    dist.init_process_group = _safe_init  # type: ignore[assignment]
    _DIST_PATCHED = True


def _patch_store_kvcache(kv_dim: int) -> None:
    """Patch ``store_kvcache`` when D (num_kv_heads * head_dim) is not a power of 2.

    Fork Triton kernel uses ``tl.arange(0, D)`` which requires D = 2^n.
    MiniMind talker has D = 2 * 96 = 192 (not power of 2); thinker
    has D = 2 * 64 = 128 (fine).  Only call for stages that need it.

    ponytail: PyTorch scatter fallback.  Upgrade: pad head_dim in fork kernel.
    """
    if kv_dim & (kv_dim - 1) == 0:
        return  # already power of 2, no patch needed

    import nanovllm.layers.attention as attn

    orig = attn.store_kvcache

    def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
        n, num_heads, head_dim = key.shape
        d = num_heads * head_dim
        if d & (d - 1) == 0:
            return orig(key, value, k_cache, v_cache, slot_mapping)
        # clamp(min=0) is CUDA graph safe — no data-dependent branch.
        slots = slot_mapping.clamp(min=0)
        k_cache.view(-1, d)[slots] = key.reshape(n, d).to(k_cache.dtype)
        v_cache.view(-1, d)[slots] = value.reshape(n, d).to(v_cache.dtype)

    attn.store_kvcache = store_kvcache


# ---------------------------------------------------------------------------
# StageRunner.
# ---------------------------------------------------------------------------


class StageRunner:
    """One fork ``ModelRunner`` + the multi-method API the fork exposes.

    ADR-002: ``set_context`` / ``forward`` / ``sample`` / ``reset_context``
    are separate methods so the call site shows the fork's
    prefill/decode lifecycle explicitly. The fork's ``run`` method
    collapses this into one call; we keep the components visible for
    learning.

    Each ``StageRunner`` owns:
      - its own ``ModelRunner`` (and therefore its own KV tensors),
      - its own ``Sampler`` (vocab differs between text thinker and
        audio talker heads),
      - a reference to the shared ``SharedBlockManager``.

    ponytail: this class does not share state across stages. KV
    tensors, sampler, CUDA graphs are per-stage.
    """

    def __init__(
        self,
        model_class: type,
        config: Any,
        shared_block_manager: SharedBlockManager,
        rank: int = 0,
    ) -> None:
        # Lazy: importing ``ModelRunner`` pulls in ``nanovllm.layers.attention``
        # which requires ``triton``. Defer to the call site.
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.layers.sampler import Sampler

        self.model_class = model_class
        self.config = config
        self.shared_block_manager = shared_block_manager
        # ``ModelRunner.__init__`` is heavy: ``dist.init_process_group``,
        # model load, weight loader, KV allocate, optional CUDA graph
        # capture. Side effects all live here.
        self.model_runner = ModelRunner(
            config,
            rank=rank,
            event=None,
            model_class=model_class,
        )
        # Per-stage vocab sampler; thinker (text) and talker (audio
        # codebook heads) cannot share.
        self.sampler = Sampler()

    # --- Lifecycle methods (mirror fork's prepare_* / run_model / sampler) ---

    def set_context(
        self,
        is_prefill: bool,
        cu_seqlens_q: Any = None,
        cu_seqlens_k: Any = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping: Any = None,
        context_lens: Any = None,
        block_tables: Any = None,
    ) -> None:
        """Install the fork module-level ``Context`` for one step.

        Must be paired with ``reset_context()`` after the forward call
        to clear stale tensors (the fork ``Attention`` reads the
        context globals at every forward).
        """
        from nanovllm.utils.context import set_context as _set_context

        _set_context(
            is_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
        )

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        is_prefill: bool,
    ) -> Any:
        """Run the model for one step; returns logits.

        Delegates to ``ModelRunner.run_model`` which knows how to pick
        between the eager path (prefill / oversize decode) and the
        CUDA-graph replay path (small decode).
        """
        return self.model_runner.run_model(input_ids, positions, is_prefill)

    def sample(self, hidden_states: Any, temperatures: Any) -> Any:
        """Draw one token per sequence from ``hidden_states`` logits.

        The fork's ``Sampler`` is ``@torch.compile``-d and expects
        ``[B, vocab]`` logits + ``[B]`` temperatures.
        """
        return self.sampler(hidden_states, temperatures)

    def reset_context(self) -> None:
        """Clear the fork module-level ``Context``."""
        from nanovllm.utils.context import reset_context as _reset_context

        _reset_context()


# ---------------------------------------------------------------------------
# ThinkerStage / TalkerStage.
# ---------------------------------------------------------------------------


def _resolve_model_path(args: Any) -> str:
    """Pull the model directory from ``OmniEngineArgs.model``.

    Fork ``Config.__post_init__`` asserts ``os.path.isdir(self.model)``.
    Resolves Hub IDs (``jingyaogong/minimind-3o``) to a local snapshot
    via ``_resolve_snapshot`` from ``bundle``. Local paths pass through.

    For the fork ``load_model`` to skip non-thinker weights, the
    directory needs only ``model.safetensors`` containing tensors
    MiniMindThinker actually consumes (with the ``model.`` prefix
    stripped). The full HF dump also ships ``audio_proj.*`` and other
    talker weights; ``load_model`` calls ``get_parameter`` for every
    key and crashes on unknowns.
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

    Mirrors ``tools/verify_phase4_wire.py::_thinker_only_dir``: strip
    ``model.`` prefix, drop tensors MiniMindThinker does not consume
    (``audio_proj.*``, ``audio_head.*``, ...), copy other config files.
    Cached by source path so repeat invocations reuse the same files.
    """
    import glob
    import hashlib
    import os
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = os.path.abspath(src)
    cache_key = hashlib.sha1(src.encode()).hexdigest()[:12]
    dst = os.path.join("/tmp", f"minimind-thinker-only-{cache_key}")
    target = os.path.join(dst, "model.safetensors")

    if os.path.isfile(target):
        return dst

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith((".json", ".py", ".jinja", ".md", ".txt")):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    keep_prefix = ("embed_tokens.", "layers.", "norm.", "lm_head.")
    tensors: dict = {}
    seen: set = set()
    src_files = glob.glob(os.path.join(src, "*.safetensors"))
    if not src_files:
        raise FileNotFoundError(f"no safetensors in {src}")
    with safe_open(src_files[0], framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
            name = k[6:] if k.startswith("model.") else k
            if not name.startswith(keep_prefix):
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


def _stage_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Map ``OmniEngineArgs`` fields onto fork ``Config`` kwargs.

    Only fields that fork ``Config`` accepts are forwarded; the rest
    stay on ``OmniEngineArgs`` for later decode-loop use.
    """
    gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    max_num_seqs = getattr(args, "max_num_seqs", None)
    tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
    return {
        "gpu_memory_utilization": (
            gpu_memory_utilization if gpu_memory_utilization is not None else 0.9
        ),
        "max_num_batched_tokens": (
            max_num_batched_tokens if max_num_batched_tokens is not None else 16384
        ),
        "max_num_seqs": max_num_seqs if max_num_seqs is not None else 512,
        "tensor_parallel_size": tensor_parallel_size,
    }


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
        self.config = get_stage_config(
            "thinker",
            model_path=_resolve_model_path(args),
            **_stage_kwargs_from_args(args),
        )
        # Patch store_kvcache if thinker head_dim * num_kv_heads is not power of 2.
        hf = self.config.hf_config
        thinker_num_kv_heads = getattr(hf, "num_key_value_heads", None) or 4
        thinker_head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        _patch_store_kvcache(thinker_num_kv_heads * thinker_head_dim)
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
        tokenizer = _load_tokenizer(_resolve_model_path(self.args))
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
    """Simple wrapper for talker audio codes consumed by ``talker2code2wav``."""

    audio_codes: Any  # torch.Tensor [frames, 8]


class TalkerStage:
    """Stage 1 — MiniMind talker MTP decode (custom, not via ModelRunner).

    ``MiniMindTalker.forward(hidden_states, text_codes, positions)`` has
    a different signature from ``ModelRunner.run_model(input_ids, positions)``,
    so we load the model directly and manage the forward loop ourselves.
    """

    def __init__(self, deploy: Any, args: Any) -> None:
        self.deploy = deploy
        self.args = args
        self._model = None
        self._device: str | None = None
        self._config: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the talker model + assign KV cache (once)."""
        if self._model is not None:
            return self._model

        import torch
        from nanovllm.config import Config
        from nanovllm.utils.loader import load_model
        from transformers import AutoConfig

        from .talker import MiniMindTalker

        model_path = _resolve_model_path(self.args)
        kwargs = _stage_kwargs_from_args(self.args)

        # Create talker config (may differ from thinker in memory util).
        orig = AutoConfig.from_pretrained

        def _trusted(*args, **kw):
            kw.setdefault("trust_remote_code", True)
            return orig(*args, **kw)

        AutoConfig.from_pretrained = _trusted  # type: ignore[method-assign]
        try:
            cfg = Config(model=model_path, **kwargs)
        finally:
            AutoConfig.from_pretrained = orig  # type: ignore[method-assign]

        dt = getattr(cfg.hf_config, "dtype", None)
        if isinstance(dt, str):
            cfg.hf_config.dtype = getattr(torch, dt, torch.float16)
        elif dt is None:
            cfg.hf_config.dtype = getattr(cfg.hf_config, "torch_dtype", None) or torch.float16

        self._config = cfg

        # Patch Triton KV store for non-power-of-2 D if needed.
        hf = cfg.hf_config
        talker_hidden = getattr(hf, "talker_hidden_size", None) or getattr(hf, "hidden_size", 768)
        talker_num_heads = getattr(hf, "talker_num_heads", None) or getattr(
            hf, "num_attention_heads", 8
        )
        talker_num_kv_heads = getattr(hf, "talker_num_kv_heads", None) or getattr(
            hf, "num_key_value_heads", 2
        )
        talker_head_dim = talker_hidden // talker_num_heads
        kv_dim = talker_num_kv_heads * talker_head_dim
        _patch_store_kvcache(kv_dim)

        # Ensure dist is ready (may already be from thinker's ModelRunner).
        _ensure_dist()

        # Load talker model (uses fork's weight loader for packed_modules_mapping).
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(cfg.hf_config.dtype)
        torch.set_default_device("cuda")
        model = MiniMindTalker(
            hidden_size=talker_hidden,
            num_layers=getattr(hf, "talker_num_layers", None)
            or getattr(hf, "num_hidden_layers", 4),
            num_heads=talker_num_heads,
            num_kv_heads=talker_num_kv_heads,
            intermediate_size=getattr(hf, "talker_intermediate_size", None),
            max_position=getattr(hf, "max_position_embeddings", 4096),
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf, "rope_theta", 10000),
            audio_vocab_size=getattr(hf, "audio_vocab_size", 2048),
            num_audio_heads=getattr(hf, "num_audio_heads", 8),
            talker_hidden_size=talker_hidden,
        )

        load_model(model, cfg.model)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Allocate KV cache and assign to attention layers.
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.current"]
        num_kv_heads = talker_num_kv_heads // cfg.tensor_parallel_size
        head_dim = talker_head_dim
        block_bytes = (
            2
            * getattr(hf, "num_hidden_layers", 4)
            * cfg.kvcache_block_size
            * num_kv_heads
            * head_dim
            * cfg.hf_config.dtype.itemsize
        )
        cfg.num_kvcache_blocks = (
            int(total * cfg.gpu_memory_utilization - used - peak + current) // block_bytes
        )
        assert (
            cfg.num_kvcache_blocks > 0
        ), f"Talker KV cache: not enough GPU memory ({cfg.num_kvcache_blocks} blocks)"
        kv_cache = torch.empty(
            2,
            getattr(hf, "num_hidden_layers", 4),
            cfg.num_kvcache_blocks,
            cfg.kvcache_block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

        self._model = model
        self._device = "cuda"
        return model

    def __call__(self, payload: Any, sampling: Any) -> Any:
        """Run the Talker: AR decode audio codes from bridge hidden states.

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

        # Extract inputs from the payload.
        if isinstance(payload, TalkerInputPayload):
            bridge = payload.bridge_states
            text_codes = list(payload.text_token_ids)
        else:
            raise TypeError(
                f"TalkerStage expected TalkerInputPayload, got {type(payload).__name__}"
            )

        if bridge is None or (isinstance(bridge, torch.Tensor) and bridge.numel() == 0):
            raise ValueError("TalkerStage received empty bridge hidden states")

        bridge = bridge.to(device=device, dtype=model.embed_proj.weight.dtype)

        # Talker max steps: number of text tokens (one audio frame per token).
        max_steps = len(text_codes)
        extra = getattr(sampling, "extra", None) or {}
        watchdog = extra.get("watchdog_limit")
        if watchdog is not None:
            max_steps = min(max_steps, int(watchdog))

        temperature = getattr(sampling, "temperature", None)
        if temperature is None:
            temperature = extra.get("temperature", 0.2)
        temperature = float(temperature)
        if temperature <= 1e-10:
            temperature = 1e-5

        num_audio_heads = model.num_audio_heads
        pad_id = 0  # initial audio code (BOS-like)

        # Generate audio codes autoregressively.
        # At each step: forward(bridge, audio_codes, positions) -> logits at last pos -> sample.
        generated_frames: list[torch.Tensor] = []
        with torch.no_grad():
            for step in range(max_steps):
                # Current audio codes: pad for positions not yet generated.
                num_generated = step
                audio_len = 1 + num_generated  # 1 pad + generated frames
                codes = torch.full((audio_len,), pad_id, dtype=torch.long, device=device)
                if num_generated > 0:
                    codes[1:] = torch.tensor(
                        [frame[step - 1] for frame in generated_frames],
                        dtype=torch.long,
                        device=device,
                    )

                positions = torch.arange(audio_len, dtype=torch.long, device=device)

                # Forward: bridge is always [T, H] (full text length).
                # The model projects bridge + adds audio embeddings for audio_len positions.
                logits = model(bridge, codes, positions)  # [audio_len, 8, vocab]
                # Sample from the last position.
                last_logits = logits[-1]  # [8, vocab]

                # Temperature sampling per codebook head.
                if temperature > 0 and temperature < 1.0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    frame = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [8]
                else:
                    frame = last_logits.argmax(dim=-1)  # [8]

                generated_frames.append(frame.cpu())

        if not generated_frames:
            # Degenerate case: return a single empty frame.
            audio_codes = torch.zeros(1, num_audio_heads, dtype=torch.long)
        else:
            audio_codes = torch.stack(generated_frames, dim=0)  # [frames, 8]

        return TalkerOutput(audio_codes=audio_codes)


__all__ = [
    "SharedBlockManager",
    "StageRunner",
    "ThinkerStage",
    "TalkerOutput",
    "TalkerStage",
    "get_shared_block_manager",
    "get_stage_config",
    "reset_shared_block_manager",
    "reset_stage_configs",
]
