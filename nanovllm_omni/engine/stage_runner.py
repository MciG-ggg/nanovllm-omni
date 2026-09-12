"""Generic fork AR-Engine adapter layer for multi-stage pipelines.

Fork ``ModelRunner`` is built around a single text AR model. ``StageRunner``
exposes its lifecycle (``set_context`` / ``forward`` / ``sample`` /
``reset_context``) as discrete methods so the call site reads as the
fork's prefill/decode dance, and threads the right ``model_class``
through.

``SharedBlockManager`` is a thin wrapper over fork ``BlockManager`` that
gives every stage a single logical block table — the table says which
block ID holds which tokens and is layer-agnostic, so two stages can
share it. Each stage still allocates its own KV cache tensor because
the layer counts differ (thinker 12 layers, talker 4 layers → the
physical (2, num_layers, num_blocks, block_size, num_kv_heads,
head_dim) tensor has different num_layers for each stage).

Fork ``dist.init_process_group`` is monkey-patched by ``_ensure_dist``
so a single process can build multiple ``ModelRunner`` instances (one
per stage) without crashing on the second ``init_process_group`` call.

Fork imports are deliberately lazy: the fork submodule requires
``triton`` / ``flash-attn`` to import ``ModelRunner``, which is not
available on a CPU-only host. Smoke tests that only import this module
don't trigger the fork import.

ponytail: only MiniMind-O consumes this today (``SharedBlockManager``
couples thinker + talker stages). Belongs in
``models/minimind_omni/stage_runner.py`` but kept at ``engine/`` until a
second family needs it (YAGNI on the move; see review SUMMARY.md §2 #17).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# SharedBlockManager
# ---------------------------------------------------------------------------


class SharedBlockManager:
    """Thin wrapper over fork ``BlockManager``.

    The *logical* block table (which block ID holds which tokens) is
    shared across stages; the *physical* KV tensors are per-stage
    because layer counts differ (thinker 12 layers, talker 4 layers —
    incompatible tensor shapes). Fork ``BlockManager`` only tracks
    logical block IDs; the per-stage KV tensors are allocated by each
    stage's ``ModelRunner`` independently.

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
# they pass.


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

    Fork ``Config.__post_init__`` accepts ``trust_remote_code`` as a
    field, so we pass it as a kwarg instead of monkey-patching
    ``AutoConfig.from_pretrained``.
    """
    if stage_name not in _stage_configs:
        import torch
        from nanovllm.config import Config

        cfg = Config(model=model_path, **kwargs)
        dt = getattr(cfg.hf_config, "dtype", None)
        if isinstance(dt, str):
            cfg.hf_config.dtype = getattr(torch, dt, torch.float16)
        elif dt is None:
            cfg.hf_config.dtype = getattr(cfg.hf_config, "torch_dtype", None) or torch.float16
        _stage_configs[stage_name] = cfg
    return _stage_configs[stage_name]


# ---------------------------------------------------------------------------
# Fork helpers: dist init.
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

    _orig_init = dist.init_process_group

    def _safe_init(*args, **kwargs):
        if dist.is_initialized():
            return None
        return _orig_init(*args, **kwargs)

    dist.init_process_group = _safe_init  # type: ignore[assignment]
    _DIST_PATCHED = True


# ---------------------------------------------------------------------------
# StageRunner.
# ---------------------------------------------------------------------------


class StageRunner:
    """One fork ``ModelRunner`` + the multi-method API the fork exposes.

    ``set_context`` / ``forward`` / ``sample`` / ``reset_context`` are
    separate methods so the call site shows the fork's prefill/decode
    lifecycle explicitly. The fork's ``run`` method collapses this
    into one call; we keep the components visible for learning.

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
        inputs_embeds: Any | None = None,
    ) -> Any:
        """Run the model for one step; returns logits.

        Delegates to ``ModelRunner.run_model``, which accepts
        ``inputs_embeds`` so multimodal preps can pass a pre-computed
        hidden. For text AR ``inputs_embeds=None``; for talker-like
        multimodal preps the runner forces a non-graph forward because
        the captured graph binds to ``input_ids``.
        """
        return self.model_runner.run_model(
            input_ids, positions, is_prefill, inputs_embeds=inputs_embeds
        )

    def sample(
        self,
        hidden_states: Any,
        temperatures: Any,
        top_k: int = 0,
        top_p: float = 1.0,
        history: Any = None,
        repetition_penalty: float = 1.0,
    ) -> Any:
        """Draw one token per sequence from ``hidden_states`` logits.

        Wraps fork ``Sampler``, which accepts ``top_k`` / ``top_p`` /
        ``history`` / ``repetition_penalty`` as optional kwargs.
        Default path is Gumbel-max (bit-exact with the pre-extension
        behaviour); the vendor top-k / top-p + repetition-penalty
        recipe is opt-in per call.
        """
        return self.sampler(
            hidden_states,
            temperatures,
            top_k=top_k,
            top_p=top_p,
            history=history,
            repetition_penalty=repetition_penalty,
        )

    def reset_context(self) -> None:
        """Clear the fork module-level ``Context``."""
        from nanovllm.utils.context import reset_context as _reset_context

        _reset_context()

    def decode(
        self,
        token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float = 1.05,
        top_p: float = 1.0,
        eos_id: int | None = None,
        enter_token: int = 0,
        pad_token: int = 0,
    ) -> tuple[list[int], Any, Any]:
        """Run the vendor decode dance: prefill + AR loop + filler tail.

        Owns the sampler wrap/restore pair (the one place that touches
        ``model_runner.sampler``), the bridge-row accumulation, and the
        post-EOS filler clocking. Returns ``(generated, bridge, text_state)``.
        """
        import torch
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams as ForkSamplingParams

        fork_sp = ForkSamplingParams(
            temperature=temperature, max_tokens=max_tokens, ignore_eos=True
        )
        scheduler = Scheduler(self.config)
        sequence = Sequence(token_ids, fork_sp)
        scheduler.add(sequence)

        model = self.model_runner.model
        runner = self.model_runner
        _base_sampler = runner.sampler

        def _sampler_with_rp(logits, temperatures):
            # Match vendor stream_generate: divide the last-token logits,
            # apply the full-history penalty and top-p filter, then call
            # torch.multinomial directly. Gumbel-max is distributionally
            # equivalent but consumes a different RNG path; a different
            # text token changes every later bridge/audio-buffer row.
            logits_i = logits[0].clone() / (temperatures[0] + 1e-9)
            for token in set(sequence.token_ids):
                logits_i[token] /= repetition_penalty
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
                remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits_i[sorted_indices[remove]] = -float("inf")
            return torch.multinomial(torch.softmax(logits_i, dim=-1), 1).view(-1)

        runner.sampler = _sampler_with_rp
        generated: list[int] = []
        bridge_hidden: Any = None
        logits: Any = None
        # Vendor ``stream_generate`` does NOT end the loop at text EOS:
        # it sets ``text_finished`` and keeps clocking the forward pass
        # with throwaway filler so the talker tail can drain. The sampler
        # still runs on those steps so the RNG stream advances identically.
        text_finished = False
        first_finished = True
        if eos_id is None:
            eos_id = self.config.eos

        try:
            while not scheduler.is_finished():
                seqs, is_prefill = scheduler.schedule()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                input_ids, positions = (
                    runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
                )
                temperatures = runner.prepare_sample(seqs)
                logits = runner.run_model(input_ids, positions, is_prefill)

                # Decode steps only see the last token, so only the last
                # bridge row is new. Prefill covers the prompt; each decode
                # step appends exactly one row.
                bh = model.get_bridge_hidden()
                if bh is not None:
                    bh = bh.detach().clone()
                    if is_prefill:
                        bridge_hidden = bh
                    elif bridge_hidden is not None:
                        bridge_hidden = torch.cat([bridge_hidden, bh[-1:]], dim=0)

                token_id_list = runner.sampler(logits, temperatures).tolist()
                sampled_id = token_id_list[0]
                if text_finished:
                    effective_id = enter_token if first_finished else pad_token
                    first_finished = False
                    token_id_list = [effective_id]
                else:
                    effective_id = sampled_id
                scheduler.postprocess(seqs, token_id_list, is_prefill)
                self.reset_context()

                generated.append(effective_id)
                if not text_finished and sampled_id == eos_id:
                    text_finished = True

                if sequence.num_completion_tokens >= max_tokens:
                    break
        finally:
            # Restore so a second call doesn't wrap an already-wrapped
            # callable (which would explode on signature mismatch).
            runner.sampler = _base_sampler

        text_state = logits[0].detach() if logits is not None else None
        if bridge_hidden is not None:
            bridge_hidden = bridge_hidden.clone()
        return generated, bridge_hidden, text_state

    def recapture_bridge(self, full_ids: list[int], bridge_hidden: Any) -> Any:
        """One full-sequence prefill to replace the paged-KV bridge.

        The decode loop forwards one token per step (paged KV cache),
        but vendor ``MiniMindOmni`` runs full-sequence attention each
        step (``use_cache=False``). The rows are not bit-equal, and the
        talker cross-attends to them — so re-run the full sequence once
        and take that bridge instead.
        """
        import torch

        if not full_ids or bridge_hidden is None:
            return bridge_hidden
        model = self.model_runner.model
        n_total = len(full_ids)
        dev = bridge_hidden.device
        self.set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, n_total], dtype=torch.int32, device=dev),
            cu_seqlens_k=torch.tensor([0, n_total], dtype=torch.int32, device=dev),
            max_seqlen_q=n_total,
            max_seqlen_k=n_total,
            slot_mapping=torch.arange(n_total, dtype=torch.int32, device=dev),
            context_lens=None,
            block_tables=None,
        )
        try:
            with torch.no_grad():
                _in = torch.tensor(full_ids, dtype=torch.int64, device=dev)
                _pos = torch.arange(n_total, dtype=torch.int64, device=dev)
                _ = model(_in, _pos)
            _full_bridge = model.get_bridge_hidden()
            if _full_bridge is not None:
                bridge_hidden = _full_bridge.detach().to(dev).clone()
        finally:
            self.reset_context()
        return bridge_hidden


# ---------------------------------------------------------------------------
# Stage kwargs mapper.
# ---------------------------------------------------------------------------


def stage_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Map ``OmniEngineArgs`` fields onto fork ``Config`` kwargs.

    Only fields that fork ``Config`` accepts are forwarded; the rest
    stay on ``OmniEngineArgs`` for later decode-loop use.
    """
    gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    max_num_seqs = getattr(args, "max_num_seqs", None)
    tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
    # MiniMind-3o ships trust_remote_code modeling files; without this
    # AutoConfig.from_pretrained raises before fork Config is built.
    # Default True (matches minimind_omni/stage.py hardcoded path);
    # caller can override by passing trust_remote_code=False explicitly.
    trust_remote_code = getattr(args, "trust_remote_code", True)
    return {
        "gpu_memory_utilization": (
            gpu_memory_utilization if gpu_memory_utilization is not None else 0.9
        ),
        "max_num_batched_tokens": (
            max_num_batched_tokens if max_num_batched_tokens is not None else 16384
        ),
        "max_num_seqs": max_num_seqs if max_num_seqs is not None else 512,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }


__all__ = [
    "SharedBlockManager",
    "StageRunner",
    "_ensure_dist",
    "get_shared_block_manager",
    "get_stage_config",
    "stage_kwargs_from_args",
]
