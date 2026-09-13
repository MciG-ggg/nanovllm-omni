"""MiniMindThinker bridge buffer is CUDA-graph safe (source-level contract).

The thinker's bridge hidden state must live in a registered buffer, not
a plain Python attribute. Plain attributes don't survive CUDA-graph
replay — Python attribute assignment isn't re-executed when the
captured graph re-runs its kernels — so a registered buffer with a
fixed memory address is required when fork ``ModelRunner`` captures
(when fork ``ModelRunner`` captures decode steps into a CUDA graph).

These tests are source-level rather than constructing
``MiniMindThinker`` because the fork's layers (Attention /
ModelRunner / triton) drag in the full GPU stack at __init__ time,
which is awkward to mock without shadowing real fork imports. The
behaviour we care about is captured by what the source code says:
a registered buffer in ``__init__``, a ``copy_`` write in ``forward``,
and ``get_bridge_hidden`` returning the buffer. If the source drifts,
these tests catch it before it reaches production and silently
breaks CUDA-graph capture.

Run alongside the engine wiring tests; the contract they lock is:
- bridge state is captured in a buffer with a fixed memory address,
  so it survives CUDA-graph replay;
- ``forward`` writes to ``_bridge_buffer`` at a fixed row (row 0 for
  decode, the seq_len prefix for prefill);
- ``get_bridge_hidden`` returns the buffer so the consumer
  (``decode_minimind``) can slice ``[:1]`` per step or
  ``[:seq_len]`` after prefill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_THINKER_PATH = (
    Path(__file__).resolve().parents[1] / "nanovllm_omni/models/minimind_omni/thinker.py"
)


def _read_source() -> str:
    return _THINKER_PATH.read_text(encoding="utf-8")


def test_bridge_buffer_registered_in_init() -> None:
    """``_bridge_buffer`` is registered (not just a Python attribute).

    A registered buffer has a fixed memory address that survives
    CUDA-graph replay; a plain ``self._bridge_buffer = ...`` Python
    attribute does not.
    """
    src = _read_source()
    assert "register_buffer" in src, "thinker.py missing register_buffer call"
    assert (
        '"_bridge_buffer"' in src or "'_bridge_buffer'" in src
    ), "thinker.py must register a buffer named _bridge_buffer"


def test_no_legacy_bridge_hidden_attribute() -> None:
    """The pre-CUDA-graph ``_bridge_hidden`` Python attribute is gone.

    Belt-and-braces guard for the rewrite. If a stale
    ``self._bridge_hidden = ...`` line creeps back into ``forward``,
    CUDA-graph replay will silently produce stale bridge data.
    """
    src = _read_source()
    assert "self._bridge_hidden" not in src, (
        "Legacy self._bridge_hidden Python attribute should be removed; "
        "use the registered _bridge_buffer instead"
    )


def test_forward_writes_to_bridge_buffer() -> None:
    """``forward`` writes the bridge row into ``_bridge_buffer``.

    The ``copy_`` writes are CUDA-graph safe because the buffer
    address is fixed at capture time. Writes that use a captured
    Python slice index (``_bridge_buffer[i]`` where ``i`` is a Python
    int varying per replay) would NOT survive replay; ``copy_`` to a
    fixed slice (``[0]`` for decode, ``[:seq_len]`` for prefill) does.
    """
    src = _read_source()
    assert (
        "_bridge_buffer[0]" in src
    ), "forward should write the decode bridge row to _bridge_buffer[0]"
    assert (
        "_bridge_buffer[:seq_len]" in src or "_bridge_buffer[:hidden_states.size" in src
    ), "forward should write the prefill bridge rows to _bridge_buffer[:seq_len]"


def test_get_bridge_hidden_returns_buffer() -> None:
    """``get_bridge_hidden`` returns ``_bridge_buffer`` (not None, not Python attr).

    The decode loop in ``models/minimind_omni/stage_runner.py`` calls
    ``model.get_bridge_hidden()`` and then slices the result. Returning
    the registered buffer — at a known fixed memory address — is
    what lets the Python consumer read the just-replayed bridge row
    between CUDA-graph replays.
    """
    src = _read_source()
    # Find the method body. ``def get_bridge_hidden`` should contain
    # ``return self._bridge_buffer`` (or equivalent).
    cls_start = src.find("class MiniMindThinker")
    assert cls_start != -1, "MiniMindThinker class missing"
    cls_src = src[cls_start:]
    method_start = cls_src.find("def get_bridge_hidden")
    assert method_start != -1, "get_bridge_hidden method missing"
    next_def = cls_src.find("\n    def ", method_start + 1)
    method_body = cls_src[method_start : next_def if next_def != -1 else None]
    assert "return" in method_body and "_bridge_buffer" in method_body, (
        "get_bridge_hidden must return the _bridge_buffer (not None, " "not a Python attribute)"
    )


def test_decode_minimind_slices_buffer_per_step() -> None:
    """The consumer slices the buffer ``[:1]`` per decode step.

    Each decode replay overwrites row 0 of the buffer; the consumer
    takes ``[:1]`` (or equivalent) to get the just-replayed row, then
    appends to the bridge list. ``bh[-1:]`` would not work because
    ``_bridge_buffer`` has ``max_position`` rows and ``[-1:]`` would
    grab the stale last row.
    """
    src_path = (
        Path(__file__).resolve().parents[1] / "nanovllm_omni/models/minimind_omni/stage_runner.py"
    )
    src = src_path.read_text(encoding="utf-8")
    assert "bh[:1]" in src, "decode_minimind should slice _bridge_buffer as bh[:1] per decode step"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
