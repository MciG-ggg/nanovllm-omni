"""Unified fork-AR engine adapter.

One file, one concept: the fork (``third_party/nano-vllm``)
attention is the only AR backend in this project, so all AR plumbing
and the per-stage runner live here. Model families that use fork
attention (currently only MiniMind-O) write their own decode recipe
on top — see ``models/minimind_omni/stage_runner.py``.

Contents:
- ``_ensure_dist`` — monkey-patches ``dist.init_process_group`` so
  two ``ModelRunner`` instances can co-exist in one process.
- ``get_stage_config`` — builds a fresh per-stage fork ``Config``. It
  is not cached: callers set ``enforce_eager`` on the returned object,
  so a shared instance would leak one stage's graph choice into another.
- ``stage_kwargs_from_args`` — maps ``OmniEngineArgs`` fields onto
  fork ``Config`` kwargs.
- ``StageRunner`` — the per-stage AR runner. ``__init__`` builds
  fork ``ModelRunner`` + ``Sampler``; the four lifecycle methods
  (``set_context`` / ``forward`` / ``sample`` / ``reset_context``)
  wrap the fork's module-level globals.

Each stage owns its own KV cache. Cross-stage state (thinker →
talker bridge hidden states) is passed through ``decode_minimind`` /
``recapture_bridge_minimind`` rather than shared memory or shared
block tables — same pattern as vllm-omni's ``StagePool``, where
every stage has independent KV.

Fork imports are lazy: ``nanovllm.engine.model_runner`` and
``nanovllm.layers.attention`` pull in ``triton``, which a CPU-only
host may lack. Smoke tests that import this module don't trigger
the fork import.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-stage Config construction.
# ---------------------------------------------------------------------------


def get_stage_config(model_path: str, **kwargs: Any) -> Any:
    """Build the fork ``Config`` for one stage.

    Fork ``Config.__post_init__`` accepts ``trust_remote_code`` as a
    field, so we pass it as a kwarg instead of monkey-patching
    ``AutoConfig.from_pretrained``.

    Every call returns a fresh object: both callers set ``enforce_eager``
    on it, so caching by stage name would let one instance's graph choice
    leak into another.
    """
    import torch
    from nanovllm.config import Config

    # The fork has no ``dtype`` field: it builds its model in
    # ``hf_config.dtype`` and sizes the KV cache with it. The
    # checkpoint dtype wins unless the caller asks for another one.
    # Without this the decode model runs in fp32 while the stage's
    # own copy of the weights runs in the requested dtype.
    dtype = kwargs.pop("dtype", None)
    config = Config(model_path, **kwargs)
    if dtype is not None:
        config.hf_config.dtype = getattr(torch, dtype)
    # Force a CUDA device probe on the config so a later
    # ``ModelRunner(config)`` sees a consistent device. The fork's
    # default is "cuda"; an explicit ``device`` kwarg wins.
    if "device" in kwargs:
        config.device = kwargs["device"]
    return config


# ---------------------------------------------------------------------------
# NCCL / dist init patch.
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
    dtype = getattr(args, "dtype", None)
    # The MiniMind-O stages ignore this flag: they register their own
    # config class, so the fork Config loads no remote code. Other
    # families pass it through to the fork Config.
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
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# StageRunner
# ---------------------------------------------------------------------------


class StageRunner:
    """Per-stage fork-AR runner.

    Builds the fork ``ModelRunner`` + ``Sampler`` pair (lazy import —
    both pull in ``triton``), then exposes the fork-attention
    lifecycle as discrete methods:

    - ``set_context`` / ``reset_context`` install and clear the
      fork module-level ``Context`` (the fork ``Attention`` layer
      reads those globals at every forward).
    - ``forward`` delegates to ``ModelRunner.run_model``.
    - ``sample`` wraps fork ``Sampler`` with the standard
      ``top_k`` / ``top_p`` / ``history`` / ``repetition_penalty``
      kwargs.

    Model families that need a custom decode loop (vendor recipe,
    filler clocking, bridge accumulation, …) compose this runner
    with their own ``decode_*`` function rather than subclassing —
    see ``models/minimind_omni/stage_runner.py``.

    ponytail: this class does not share state across stages. KV
    tensors, sampler, CUDA graphs are per-stage.
    """

    def __init__(
        self,
        model_class: type,
        config: Any,
        rank: int = 0,
    ) -> None:
        # Lazy: importing ``ModelRunner`` pulls in
        # ``nanovllm.layers.attention`` which requires ``triton``.
        # Defer to the call site.
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.layers.sampler import Sampler

        self.model_class = model_class
        self.config = config
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

        Delegates to ``model_runner.run_model``, which accepts
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


__all__ = [
    "StageRunner",
    "_ensure_dist",
    "get_stage_config",
    "stage_kwargs_from_args",
]
