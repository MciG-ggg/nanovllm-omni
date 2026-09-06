"""MiniMind-Omni bridge hidden-state capture tests.

The thinker's bridge-layer hidden state is the conditioning signal the
talker consumes to generate Mimi codec codes. This test pins the
end-to-end contract for capturing it inside ``BatchedThinkerRunner``:

  * ``capture_bridge_states=False`` (default) leaves ``state.bridge_states``
    empty and pays zero overhead.
  * ``capture_bridge_states=True`` populates ``state.bridge_states`` with
    one ``[hidden_size]`` tensor per runner step (prefill = 1 entry;
    decode = 1 entry per step).
  * ``extract_bridge_states(state)`` stacks them into
    ``[num_steps, hidden_size]``.
  * ``enable_bridge_capture`` is idempotent.

A tiny ``FakeMiniMindOmniWithBridge`` stands in for the joint model; its
bridge layer's ``forward`` is monkey-patched via
``enable_bridge_capture`` so we exercise the same code path the runner
uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.batched_generation import (  # noqa: E402
    BatchedThinkerRunner,
    _resolve_bridge_layer,
    enable_bridge_capture,
    extract_bridge_states,
)
from nanovllm_omni.models.minimind_omni.generation import stream_generate  # noqa: E402
from tests.test_batched_generation import (  # noqa: E402
    FakeMiniMindOmni,
)

# ---------------------------------------------------------------------------
# Fake model with a real MiniMindBlock-shaped bridge layer
# ---------------------------------------------------------------------------


class _BridgeLayer(torch.nn.Module):
    """Single-projection stand-in for ``MiniMindBlock``.

    Signature mirrors the vendored HF block:
    ``forward(hidden_states, position_embeddings, past_key_value, use_cache, attention_mask) -> (hidden_states, present)``.
    Returns a deterministic post-MLP hidden state so we can assert the
    capture path picks it up unchanged.
    """

    def __init__(self, hidden_size: int, marker_value: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.marker_value = float(marker_value)
        # Identity-ish projection; output is marker * hidden_states so the
        # captured tensor is recognisable.
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> tuple[torch.Tensor, None]:
        out = self.proj(hidden_states) * self.marker_value
        return out, None


class FakeMiniMindOmniWithBridge(FakeMiniMindOmni):
    """Extends ``FakeMiniMindOmni`` with a real bridge-layer block.

    The bridge-layer ``forward`` is monkey-patchable via
    ``enable_bridge_capture`` so the runner can stash
    ``block._bridge_capture`` after each joint forward.
    """

    def __init__(
        self,
        *,
        num_thinker_layers: int = 4,
        num_talker_layers: int = 2,
        hidden_size: int = 8,
        kv_heads: int = 2,
        head_dim: int = 4,
        vocab: int = 64,
        bridge_marker: float = 2.0,
    ) -> None:
        super().__init__(
            num_thinker_layers=num_thinker_layers,
            num_talker_layers=num_talker_layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            vocab=vocab,
        )
        # Override the parent fake's audio_pad_token (2049) which is larger
        # than vocab=64; without this, ``_note_audio``'s sampling step indexes
        # history[2049] out of bounds. Keep stop >= AUDIO_VOCAB_BOUNDARY (2048)
        # semantics by skipping the audio-sampling path in the bridge tests;
        # the runner's max_new_tokens=4 makes every test terminate via the
        # text-EOS path long before the audio-stop gate matters.
        self.audio_pad_token = 10
        self.audio_stop_token = max(vocab - 1, 64)
        self.config.bridge_layer = num_thinker_layers // 2 - 1
        self.config.num_hidden_layers = num_thinker_layers
        # Replace the placeholder thinker.layers list with real blocks so
        # ``enable_bridge_capture`` can monkey-patch the bridge layer's
        # ``forward``.
        bridge_idx = self.config.bridge_layer
        blocks = []
        for i in range(num_thinker_layers):
            if i == bridge_idx:
                blocks.append(_BridgeLayer(hidden_size, bridge_marker))
            else:
                # Non-bridge layers use the same _BridgeLayer shape with
                # marker=1.0 so the joint forward path looks uniform but
                # the captured tensor at the bridge index stands out.
                blocks.append(_BridgeLayer(hidden_size, 1.0))
        self.thinker.layers = torch.nn.ModuleList(blocks)
        # Joint forward still runs the same path but routes through the
        # real layers (instead of the SimpleNamespace placeholders used
        # by the parent fake). The parent's forward signature is
        # unchanged because the joint forward doesn't actually call
        # layer.forward -- it relies on the parent fake's behaviour.
        self._bridge_idx = bridge_idx
        self._hidden_size = hidden_size
        self._bridge_marker = float(bridge_marker)

        # Re-bind ``forward`` so it actually walks the thinker's layers in
        # order. The parent fake just returns zeros; the bridge-layer patch
        # never fires. By invoking each layer's ``forward`` here, the
        # bridge-layer patch sets ``_bridge_capture`` on the captured tensor,
        # which the runner then reads via ``_read_bridge_capture``.
        outer_forward = self.forward

        def _forward_with_layers(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            logits_to_keep=0,
            **kwargs,
        ):
            # Run the parent's forward for logits + past + audio_logits,
            # then ALSO walk the thinker layers so the bridge patch fires.
            bs, _, tlen = input_ids.shape
            hidden = torch.zeros(bs, tlen, self._hidden_size)
            for layer in self.thinker.layers:
                hidden, _ = layer(hidden, None)
            return outer_forward(
                input_ids, attention_mask, past_key_values, use_cache, logits_to_keep, **kwargs
            )

        import types

        self.forward = types.MethodType(_forward_with_layers, self)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_runner(model: Any, prompts: list[list[int]], *, capture_bridge: bool = False) -> tuple:
    sched = RuntimeScheduler(max_num_seqs=2)
    runner = BatchedThinkerRunner(
        SimpleNamespace(model=model),
        sched,
        temperature=0.75,
        top_p=1.0,
        rp=1.0,
        max_new_tokens=4,
        open_thinking=False,
        base_seed=7,
        capture_bridge_states=capture_bridge,
    )
    rids = []
    for p in prompts:
        rid = runner.add_request(p)
        rids.append(rid)
    return runner, sched, rids


def _drain(runner: BatchedThinkerRunner, sched) -> None:
    finished: set[str] = set()
    while sched.has_work():
        out = sched.schedule()
        if out.is_empty:
            break
        prefilled: set[str] = set()
        finished_now: set[str] = set()
        for g in out.prefill_groups:
            runner.prefill_group(g)
            prefilled.update(chunk.sequence.request_id for chunk in g.items)
        for g in out.decode_groups:
            runner.decode_group(g)
            for sequence in g.items:
                if runner.step_finished(sequence.request_id):
                    finished_now.add(sequence.request_id)
        sched.update_from_output(prefilled=prefilled, finished=finished_now)
        finished.update(finished_now)


# Lazy import so non-bridge tests can skip the runtime scheduler import.
from nanovllm_omni.engine.runtime_scheduler import RuntimeScheduler  # noqa: E402

# ---------------------------------------------------------------------------
# enable_bridge_capture + _resolve_bridge_layer
# ---------------------------------------------------------------------------


def test_resolve_bridge_layer_uses_explicit_config_field() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4)
    # explicit config.bridge_layer = 1 (set in fake's __init__).
    assert _resolve_bridge_layer(model) == 1


def test_resolve_bridge_layer_defaults_to_middle_layer() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=6)
    # Drop the explicit field; default should be num_hidden//2 - 1 = 2.
    delattr(model.config, "bridge_layer")
    assert _resolve_bridge_layer(model) == 2


def test_resolve_bridge_layer_returns_minus_one_when_model_lacks_layers() -> None:
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=0))
    assert _resolve_bridge_layer(model) == -1


def test_enable_bridge_capture_patches_bridge_layer() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    bridge_idx = model.config.bridge_layer
    block = model.thinker.layers[bridge_idx]
    # Pre-condition: no _bridge_capture attribute.
    assert not hasattr(block, "_bridge_capture")
    patched = enable_bridge_capture(model, bridge_idx)
    assert patched == bridge_idx
    assert getattr(block, "_nanovllm_bridge_patched", False) is True
    # Now invoke the patched forward and verify _bridge_capture is set.
    block.forward(torch.randn(2, 3, 8))
    assert hasattr(block, "_bridge_capture")


def test_enable_bridge_capture_is_idempotent() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    enable_bridge_capture(model, model.config.bridge_layer)
    first_block = model.thinker.layers[model.config.bridge_layer]
    first_forward = first_block.forward
    enable_bridge_capture(model, model.config.bridge_layer)
    second_forward = first_block.forward
    # Same bound method (re-patch is a no-op).
    assert first_forward.__func__ is second_forward.__func__


def test_enable_bridge_capture_returns_minus_one_for_missing_thinker() -> None:
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=8))
    assert enable_bridge_capture(model, 3) == -1


def test_enable_bridge_capture_returns_minus_one_for_out_of_range_layer() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4)
    assert enable_bridge_capture(model, 999) == -1


# ---------------------------------------------------------------------------
# Runner integration: capture_bridge_states=True populates state.bridge_states
# ---------------------------------------------------------------------------


def test_runner_with_capture_off_leaves_bridge_states_empty() -> None:
    """Default (capture_bridge_states=False) pays zero overhead."""
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    runner, sched, rids = _make_runner(model, [[1, 2, 3]], capture_bridge=False)
    _drain(runner, sched)
    for rid in rids:
        assert runner.states[rid].bridge_states == []
    # No patch was installed (capture was off).
    block = model.thinker.layers[model.config.bridge_layer]
    assert not getattr(block, "_nanovllm_bridge_patched", False)


def test_runner_with_capture_on_populates_prefill_and_decode_states() -> None:
    """capture_bridge_states=True installs the patch and populates the per-step list."""
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    runner, sched, rids = _make_runner(model, [[1, 2, 3]], capture_bridge=True)
    _drain(runner, sched)
    st = runner.states[rids[0]]
    # prefill adds 1 entry (last prompt position) + at least 1 decode step.
    assert len(st.bridge_states) >= 2
    # Every captured tensor is [hidden_size] (the per-request last position).
    for t in st.bridge_states:
        assert t.shape == (8,)


def test_runner_capture_is_per_request() -> None:
    """Two requests in one batched group: each gets its own row index."""
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    runner, sched, (a, b) = _make_runner(model, [[1, 2, 3], [4, 5, 6]], capture_bridge=True)
    _drain(runner, sched)
    # Both states should have the same number of entries (same group steps).
    assert len(runner.states[a].bridge_states) == len(runner.states[b].bridge_states)
    assert len(runner.states[a].bridge_states) >= 2
    for t in runner.states[a].bridge_states + runner.states[b].bridge_states:
        assert t.shape == (8,)


# ---------------------------------------------------------------------------
# extract_bridge_states
# ---------------------------------------------------------------------------


def test_extract_bridge_states_empty_state_returns_zero_row_tensor() -> None:
    runner, sched, (rid,) = _make_runner(FakeMiniMindOmniWithBridge(), [[1]], capture_bridge=False)
    st = runner.states[rid]
    out = extract_bridge_states(st)
    assert out.shape == (0, 0)


def test_extract_bridge_states_stacks_per_step_into_2d_tensor() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    runner, sched, (rid,) = _make_runner(model, [[1, 2, 3]], capture_bridge=True)
    _drain(runner, sched)
    out = extract_bridge_states(runner.states[rid])
    # 1 prefill entry + at least 1 decode entry.
    assert out.ndim == 2
    assert out.shape[0] >= 2
    assert out.shape[1] == 8
    # Extracted to CPU + fp32 (extract_bridge_states contract).
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32


def test_extract_bridge_states_returns_independent_copy() -> None:
    """Mutating the extracted tensor must not affect the runner's buffer."""
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    runner, sched, (rid,) = _make_runner(model, [[1, 2, 3]], capture_bridge=True)
    _drain(runner, sched)
    st = runner.states[rid]
    original = st.bridge_states[0].clone()
    out = extract_bridge_states(st)
    out[0, 0] = 999.0
    # The runner's buffer is untouched.
    assert torch.allclose(st.bridge_states[0], original)


def test_stream_generate_forwards_bridge_capture_callback() -> None:
    model = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    captured: list[torch.Tensor] = []
    list(
        stream_generate(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            max_new_tokens=4,
            top_p=1.0,
            capture_bridge_states=True,
            bridge_state_callback=captured.append,
        )
    )
    assert len(captured) == 1
    assert captured[0].ndim == 2
    assert captured[0].shape[1] == 8
    assert captured[0].device.type == "cpu"


# ---------------------------------------------------------------------------
# Backward compatibility: capture flag does not change runner behaviour
# ---------------------------------------------------------------------------


def test_capture_flag_does_not_change_text_or_audio_sampling() -> None:
    """Bridge capture is purely additive: text + audio outputs stay identical
    when the flag is toggled (Q10a determinism contract).
    """
    model_a = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    model_b = FakeMiniMindOmniWithBridge(num_thinker_layers=4, hidden_size=8)
    # Same seed / same prompts.
    runner_a, sched_a, (rid_a,) = _make_runner(model_a, [[1, 2, 3]], capture_bridge=False)
    runner_b, sched_b, (rid_b,) = _make_runner(model_b, [[1, 2, 3]], capture_bridge=True)
    _drain(runner_a, sched_a)
    _drain(runner_b, sched_b)
    # Same text tokens, same audio codes.
    assert runner_a.states[rid_a].text_tokens == runner_b.states[rid_b].text_tokens
    assert runner_a.states[rid_a].audio_codes == runner_b.states[rid_b].audio_codes
