"""Per-request fixed-slot KV pool for MiniMind-O.

After the design split, the request lifecycle + group formation moved to
``runtime_scheduler.RuntimeScheduler``. This module keeps only the
KV-cache memory manager (``FixedKvSlotPool``) which is orthogonal to the
scheduler -- it owns the per-request fixed-size KV buffers that
``BatchedThinkerRunner`` writes into during ``prefill_group`` /
``decode_group``.

Scope notes:

- Q4b/Q6a  real concurrency with fixed-slot KV: each running request owns
  one preallocated [layers, 2, kv_heads, max_seq, head_dim] slot written in
  place (no per-step ``torch.cat`` growth); decode groups gather rows back
  into one [B, ...] tensor for a batched forward.
- No preemption / no paged blocks: admission is capped by ``max_num_seqs``
  on the scheduler; this pool just hands out slots sized to whatever
  ``max_seq`` the runner was constructed with (fixed-slot budgets;
  add preemption when a 4 GB card OOMs, and paged KV only if you outgrow
  this teaching shape).
"""

from __future__ import annotations

from typing import Any


class FixedKvSlotPool:
    """Per-request fixed-size KV slots with in-place writes.

    Each running request owns one ``[n_layers, 2, max_seq, kv_heads, head_dim]``
    buffer matching the model's KV layout ``[B, seq, kv_heads, d]`` (seq at
    dim 1 -- the vendored attention cats on ``dim=1`` and the omni forward
    reads ``start_pos = past_key_values[0][0].shape[1]``). The runner writes
    newly computed per-layer (key, value) rows into the slot; ``gather``
    re-stacks the alive rows of a decode group into one ``[B, seq, kv, d]``
    (key, value) pair per layer so a single forward processes them together.

    Tensor-free itself; all shapes are supplied by the runner at register time.
    """

    def __init__(self, max_sequence_len: int) -> None:
        self.max_sequence_len = max_sequence_len
        self._slots: dict[str, dict[str, Any]] = {}

    def register(
        self,
        req_id: str,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: Any,
        dtype: Any,
    ) -> None:
        """Allocate a per-request KV slot. ``max_seq`` is fixed at construction."""
        import torch  # deferred; only needed by callers running with torch

        if req_id in self._slots:
            return  # idempotent re-register
        buf = torch.empty(
            (num_layers, 2, self.max_sequence_len, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._slots[req_id] = {"buf": buf, "ptr": 0}

    def write(self, req_id: str, layer: int, key: Any, value: Any, *, row: int = 0) -> None:
        """Copy one layer's (key, value) rows [B, seq, kv, d] into the slot.

        ``key[row]`` / ``value[row]`` are ``[seq, kv_heads, d]`` and are stored
        into layer ``layer``; the write pointer advances to the sequence length.
        """
        slot = self._slots[req_id]
        k = key[row]
        v = value[row]
        length = k.shape[0]
        if length > self.max_sequence_len:
            raise ValueError(f"seq length {length} exceeds pool max_seq={self.max_sequence_len}")
        slot["buf"][layer, 0, :length, :, :] = k
        slot["buf"][layer, 1, :length, :, :] = v
        slot["ptr"] = length

    def gather(self, req_ids: list[str]) -> list[tuple[Any, Any]]:
        """Re-stack the alive rows of one decode group into a per-layer (k, v) pair.

        All ``req_ids`` must be at the same write pointer (``start_pos``); the
        caller (``BatchedThinkerRunner.decode_group``) guarantees this by only
        forming decode groups over sequences sharing ``num_tokens``.
        """
        import torch  # deferred; this module is otherwise tensor-free

        slots = [self._slots[r] for r in req_ids]
        if not slots:
            return []
        ptrs = {s["ptr"] for s in slots}
        if len(ptrs) > 1:
            raise ValueError(
                f"unequal write pointers in gather: {ptrs} -- " "decode groups must share start_pos"
            )
        length = next(iter(ptrs))
        if length == 0:
            raise ValueError(f"{[s['ptr'] for s in slots]} -- run prefill first")
        num_layers = slots[0]["buf"].shape[0]
        pairs: list[tuple[Any, Any]] = []
        for layer in range(num_layers):
            keys = torch.stack([s["buf"][layer, 0, :length, :, :] for s in slots])
            vals = torch.stack([s["buf"][layer, 1, :length, :, :] for s in slots])
            pairs.append((keys, vals))
        return pairs

    def release(self, req_id: str) -> None:
        self._slots.pop(req_id, None)

    def length(self, req_id: str) -> int:
        """Current write pointer for ``req_id`` (= number of tokens written)."""
        return self._slots[req_id]["ptr"]


__all__ = ["FixedKvSlotPool"]
