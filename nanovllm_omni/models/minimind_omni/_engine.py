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

Why a module-level cache for ``SharedBlockManager``: ``ThinkerStage`` and
``TalkerStage`` are constructed by independent factory calls in
``PipelineRunner._ensure_stages``. Without a shared handle the second
stage would mint a fresh ``BlockManager`` and the two stages' block
IDs would collide.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Fork import plumbing (lazy).
# ---------------------------------------------------------------------------
#
# The fork submodule lives at ``third_party/nano-vllm`` and is *not*
# installed as a package (no ``pyproject.toml`` for it at the project
# root). Inserting its directory onto ``sys.path`` is the simplest way
# to make ``from nanovllm.engine.model_runner import ModelRunner``
# resolve. Only done once, on first fork import inside this module.

_FORK_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "third_party",
        "nano-vllm",
    )
)


def _ensure_fork_on_path() -> str:
    """Insert the fork submodule directory into ``sys.path`` once.

    Returns the resolved absolute path so callers can detect duplicates.
    Importing the fork eagerly here would pull in ``triton`` (via
    ``nanovllm.layers.attention``) — leave that to the call sites that
    actually need it.
    """
    if _FORK_PATH not in sys.path:
        sys.path.insert(0, _FORK_PATH)
    return _FORK_PATH


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
        # Lazy: only touch the fork when an instance is actually built.
        _ensure_fork_on_path()
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
    """Return the fork ``Config`` for one stage (lazy, cached)."""
    if stage_name not in _stage_configs:
        _ensure_fork_on_path()
        from nanovllm.config import Config

        _stage_configs[stage_name] = Config(model=model_path, **kwargs)
    return _stage_configs[stage_name]


def reset_stage_configs() -> None:
    """Drop the cached stage ``Config`` objects (test hook)."""
    _stage_configs.clear()


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
        _ensure_fork_on_path()
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
        _ensure_fork_on_path()
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
        _ensure_fork_on_path()
        from nanovllm.utils.context import reset_context as _reset_context

        _reset_context()


# ---------------------------------------------------------------------------
# ThinkerStage / TalkerStage.
# ---------------------------------------------------------------------------


def _resolve_model_path(args: Any) -> str:
    """Pull the model directory from ``OmniEngineArgs.model``.

    Fork ``Config.__post_init__`` asserts ``os.path.isdir(self.model)``
    so we surface the missing-arg case as a clearer error before
    hitting the fork assertion.
    """
    model = getattr(args, "model", None)
    if not model:
        raise ValueError(
            "OmniEngineArgs.model is required to construct a StageRunner "
            "(fork Config validates the model path is a real directory)."
        )
    return model


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


class ThinkerStage:
    """Stage 0 — MiniMind thinker AR decode via fork ``ModelRunner``.

    The factory ``_thinker_stage`` (in ``thinker.py``) constructs this
    class; ``PipelineRunner`` then drives decoding by calling the
    instance with ``(payload, sampling)``.

    Phase 3 stops at the wiring (ModelRunner constructed, KV allocated,
    CUDA graph captured). The actual decode loop + bridge extraction
    into ``ThinkerStageOutput`` lives in Phase 4.
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
        # The first ``get_shared_block_manager`` call wins; the
        # talker's later call reuses this same instance.
        self.shared_block_manager = get_shared_block_manager(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
        )
        self.stage_runner = StageRunner(
            model_class=MiniMindThinker,
            config=self.config,
            shared_block_manager=self.shared_block_manager,
        )

    def __call__(self, payload: Any, sampling: Any) -> Any:
        raise NotImplementedError(
            "ThinkerStage.__call__ (decode loop + bridge extraction) is "
            "Phase 4 territory. Phase 3 stops at ModelRunner construction "
            "and CUDA graph capture; see docs/dev/nanovllm-omni-rewrite.md §7."
        )


class TalkerStage:
    """Stage 1 — MiniMind talker MTP via fork ``ModelRunner``.

    Consumes ``TalkerInputPayload`` (bridge hidden states + text token
    alignment) produced by ``thinker2talker``. Phase 3 stops at
    ModelRunner construction; the MTP decode loop is Phase 4.
    """

    def __init__(self, deploy: Any, args: Any) -> None:
        from .talker import MiniMindTalker

        self.deploy = deploy
        self.args = args
        self.config = get_stage_config(
            "talker",
            model_path=_resolve_model_path(args),
            **_stage_kwargs_from_args(args),
        )
        # Reuses the ThinkerStage's SharedBlockManager.
        self.shared_block_manager = get_shared_block_manager(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
        )
        self.stage_runner = StageRunner(
            model_class=MiniMindTalker,
            config=self.config,
            shared_block_manager=self.shared_block_manager,
        )

    def __call__(self, payload: Any, sampling: Any) -> Any:
        raise NotImplementedError(
            "TalkerStage.__call__ (MTP decode loop + audio code emission) "
            "is Phase 4 territory. Phase 3 stops at ModelRunner construction; "
            "see docs/dev/nanovllm-omni-rewrite.md §7."
        )


__all__ = [
    "SharedBlockManager",
    "StageRunner",
    "ThinkerStage",
    "TalkerStage",
    "get_shared_block_manager",
    "get_stage_config",
    "reset_shared_block_manager",
    "reset_stage_configs",
]
