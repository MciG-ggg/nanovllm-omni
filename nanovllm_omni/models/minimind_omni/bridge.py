"""Bridge capture for the MiniMind-O thinker.

``recapture_bridge_minimind`` re-runs the model over the full sequence to
get the bridge rows the talker cross-attends to. It is the CUDA track's
path; the MLX track gets its rows from the runner's own ``recapture_bridge``.
"""

from __future__ import annotations

from typing import Any


def recapture_bridge_minimind(
    stage_runner: Any,
    full_ids: list[int],
    bridge_hidden: Any,
) -> Any:
    """One full-sequence prefill to replace the paged-KV bridge.

    The decode loop forwards one token per step (paged KV cache),
    but vendor ``MiniMindOmni`` runs full-sequence attention each
    step (``use_cache=False``). The rows are not bit-equal, and the
    talker cross-attends to them — so re-run the full sequence once
    and take that bridge instead.

    The re-run detaches the paged KV tensors first. A request can finish
    while others are still decoding, and a prefill writes the KV it
    computes; with the paged cache attached that write would land on
    slots owned by the running sequences. With the cache detached the
    attention uses the freshly computed K/V and writes nothing.
    """
    import torch

    if not full_ids or bridge_hidden is None:
        return bridge_hidden
    model = stage_runner.model
    n_total = len(full_ids)
    dev = bridge_hidden.device
    caches = [
        (module, module.k_cache, module.v_cache)
        for module in model.modules()
        if hasattr(module, "k_cache") and hasattr(module, "v_cache")
    ]
    empty = torch.tensor([])
    stage_runner.set_context(
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
        for module, _, _ in caches:
            module.k_cache = module.v_cache = empty
        with torch.no_grad():
            _in = torch.tensor(full_ids, dtype=torch.int64, device=dev)
            _pos = torch.arange(n_total, dtype=torch.int64, device=dev)
            _ = model(_in, _pos)
        _full_bridge = stage_runner.bridge_hidden()
        if _full_bridge is not None:
            # The buffer holds ``max_position`` rows; only the first
            # ``n_total`` carry this sequence. The talker embeds every
            # row it gets, so the trailing zero rows must not travel.
            bridge_hidden = _full_bridge[:n_total].detach().to(dev).clone()
    finally:
        for module, k_cache, v_cache in caches:
            module.k_cache = k_cache
            module.v_cache = v_cache
        stage_runner.reset_context()
    return bridge_hidden


__all__ = ["recapture_bridge_minimind"]
