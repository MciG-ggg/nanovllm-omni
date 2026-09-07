"""CPU tests for the thinker-CUDA-Graph + full-pipeline contract.

When ``bundle.use_thinker_cuda_graph=True`` is set on the full pipeline,
the thinker's stage factory (``_full_thinker_stage``) routes through
``run_generate`` instead of calling ``stream_generate`` directly. On
CPU (no CUDA), ``enable_cuda_graph`` returns ``None`` so the graph
attempt silently falls through to eager -- this file pins the
end-to-end ``ThinkerStageOutput`` so the seam is exercised without a
GPU.

A second test exercises ``run_generate`` with post-EOS padding directly:
a stub decoder flips ``_text_finished`` (defect B parity), and we verify
that graph output is returned without replaying eager generation. This
pins the production contract without touching a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni import generation as gen_module  # noqa: E402
from nanovllm_omni.models.minimind_omni.stage_processors import (  # noqa: E402
    ThinkerStageOutput,
)
from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402
from nanovllm_omni.models.minimind_omni.thinker import (  # noqa: E402
    _full_thinker_stage,
    run_generate,
)
from nanovllm_omni.optim import cuda_graph as cg  # noqa: E402
from tests._talker_fixtures import make_fake_bundle  # noqa: E402
from tests.test_full_pipeline import (  # noqa: E402
    FakeMimi,
    FakeMiniMindOmniWithBridge,
    _FakeTokenizer,
)


def _make_full_bundle(*, use_graph: bool) -> Any:
    model = FakeMiniMindOmniWithBridge()
    mimi = FakeMimi()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=_FakeTokenizer(),
        mimi=mimi,
        device="cpu",
        model_id="fake/minimind-3o",
        thinker=model,
        talker=None,
        code2wav=mimi,
        use_thinker_cuda_graph=use_graph,
    )
    talker = wrap_talker(make_fake_bundle())
    bundle.talker = talker
    return bundle


# ---------------------------------------------------------------------------
# End-to-end: _full_thinker_stage closes over the right code path
# ---------------------------------------------------------------------------


def test_full_thinker_stage_with_cuda_graph_flag_runs_and_emits_valid_output() -> None:
    """With ``use_thinker_cuda_graph=True``, ``_full_thinker_stage`` runs
    through ``run_generate`` (which falls through to eager on CPU). The
    returned ``ThinkerStageOutput`` must be structurally valid -- the
    bridge tensor has the prompt + decode-span shape, ``output_token_ids``
    is non-empty, and metadata carries the post-EOS knob.
    """
    bundle = _make_full_bundle(use_graph=True)
    deploy = SimpleNamespace(post_eos_padding_count=128, internal_stop_token_id=17)
    stage = _full_thinker_stage(bundle, deploy)
    sampling = SimpleNamespace(max_tokens=4, temperature=0.2, top_p=0.9)

    out = stage("hello", sampling)

    assert isinstance(out, ThinkerStageOutput)
    # Bridge shape: rank-2 [T_bridge, hidden_size] with hidden_size=8.
    assert out.bridge_states.ndim == 2
    assert out.bridge_states.shape[1] == 8
    assert out.bridge_states.shape[0] > 0  # at least prompt-position rows
    # Output tokens captured.
    assert isinstance(out.output_token_ids, list)
    assert len(out.output_token_ids) >= 1
    # Aligned text span covers prompt + output (talker consumes this).
    assert len(out.text_token_ids) == len(out.prompt_token_ids) + max(
        len(out.output_token_ids) - 1, 0
    )
    # Metadata carries the deploy knobs so downstream stages can audit them.
    assert out.metadata["pipeline_kind"] == "full"
    assert out.metadata["post_eos_padding_count"] == 128
    # request_id is set for tracing.
    assert out.request_id is not None


def test_full_thinker_stage_eager_path_matches_graph_path_shape() -> None:
    """Sanity: the eager-only and graph-flagged runs emit structurally
    equivalent outputs (same bridge columns, same metadata shape) so
    audio bytes stay bit-exact across the deploy flag toggle.
    """
    bundle_graph = _make_full_bundle(use_graph=True)
    bundle_eager = _make_full_bundle(use_graph=False)
    deploy = SimpleNamespace(post_eos_padding_count=128, internal_stop_token_id=17)
    sampling = SimpleNamespace(max_tokens=4, temperature=0.2, top_p=0.9)

    out_graph = _full_thinker_stage(bundle_graph, deploy)("hello", sampling)
    out_eager = _full_thinker_stage(bundle_eager, deploy)("hello", sampling)

    # Same hidden-size column, same prompt (prompt is deterministic from
    # the fake tokenizer).
    assert out_graph.bridge_states.shape[1] == out_eager.bridge_states.shape[1] == 8
    assert out_graph.prompt_token_ids == out_eager.prompt_token_ids
    assert out_graph.metadata == out_eager.metadata
    # The bridge column count is identical for both -- the post-EOS
    # padding happens in both paths (graph-flagged falls through to
    # eager on CPU, eager runs direct).
    assert out_graph.bridge_states.shape[0] > 0
    assert out_eager.bridge_states.shape[0] > 0


# ---------------------------------------------------------------------------
# Direct test of run_generate's post-EOS graph state machine
# ---------------------------------------------------------------------------


class _FakeGraphDecoder:
    """Stand-in for ``CudaGraphDecoder``.

    ``stop_early`` (default True) mirrors defect B parity: the first
    call flips ``_text_finished`` so the early-stop branch engages. Set
    ``stop_early=False`` to simulate a graph that ran to budget.

    Returned ``generate_tokens`` mimics the real decoder: text_codes,
    audio_codes (8 channels), and an optional bridge tensor of shape
    ``[prompt_len + decode_steps, hidden_size]``.
    """

    def __init__(
        self,
        *,
        text_codes: list[int],
        hidden_size: int = 8,
        stop_early: bool = True,
    ) -> None:
        self.temperature = None
        self.top_p = None
        self.rp = 1.0
        self.eos_token_id = text_codes[-1] if text_codes else 0
        self.audio_stop_token = 0
        self.audio_pad = 1
        self._text_finished = False
        self._text_codes = text_codes
        self._hidden_size = hidden_size
        self._stop_early = stop_early

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        seed: int | None = None,
        return_audio: bool = False,
        return_bridge: bool = False,
        **_kwargs: Any,
    ) -> Any:
        # Mirror CudaGraphDecoder.generate_tokens behaviour: flip the
        # flag the first time we sample EOS, then exit -- but only when
        # stop_early=True (the production stop-on-EOS path).
        decode_steps = len(self._text_codes)
        audio_codes = [[self.audio_pad] * decode_steps for _ in range(8)]
        if self._stop_early and not self._text_finished and self._text_codes:
            self._text_finished = True
        if return_bridge:
            prompt_len = input_ids.shape[1]
            bridge = torch.zeros(prompt_len + decode_steps, self._hidden_size)
            return self._text_codes, audio_codes, bridge
        return self._text_codes, audio_codes


class _FakeEagerStream:
    """Iterable wrapper that mimics ``stream_generate``'s contract.

    Yields one ``(text_chunk, audio_frame)`` then exits; if a
    ``bridge_state_callback`` is supplied, fires it at the end with a
    tensor spanning ``prompt_len + decode_steps + post_eos_steps``. This
    mirrors the production eager runner's bridge shape so the
    consolidation logic in ``run_generate`` can be verified.
    """

    def __init__(
        self,
        text_codes: list[int],
        audio_frame: list[int],
        bridge_tensor: torch.Tensor,
    ) -> None:
        self._text_codes = text_codes
        self._audio_frame = audio_frame
        self._bridge = bridge_tensor
        self._fired = False

    def __iter__(self) -> Any:
        self._fired = False
        return self

    def __next__(self) -> Any:
        if self._fired:
            raise StopIteration
        self._fired = True
        return torch.tensor([self._text_codes]), self._audio_frame


def test_run_generate_uses_graph_output_with_post_eos_padding() -> None:
    """Post-EOS padding stays on the graph path instead of replaying eager."""
    fake_decoder = _FakeGraphDecoder(text_codes=[42, 43, 44], hidden_size=8)
    original_enable = cg.enable_cuda_graph
    cg.enable_cuda_graph = lambda *a, **kw: fake_decoder  # type: ignore[assignment]
    try:

        def boom(*a, **kw):  # noqa: ARG001
            raise AssertionError("eager stream_generate must not replay graph work")

        original_stream = gen_module.stream_generate
        gen_module.stream_generate = boom  # type: ignore[assignment]
        try:
            model = SimpleNamespace(
                forward=object(),
                audio_pad_token=1,
                audio_stop_token=2,
                audio_spk_token=3,
                config=SimpleNamespace(audio_pad_token=1),
            )
            captured: list[torch.Tensor] = []
            frames = run_generate(
                model,
                torch.tensor([[1] * 5]),
                max_new_tokens=10,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=fake_decoder.eos_token_id,
                open_thinking=False,
                use_thinker_cuda_graph=True,
                post_eos_padding_count=128,
                capture_bridge_states=True,
                bridge_state_callback=captured.append,
            )
        finally:
            gen_module.stream_generate = original_stream  # type: ignore[assignment]
    finally:
        cg.enable_cuda_graph = original_enable  # type: ignore[assignment]

    assert len(frames) == 3
    assert all(len(frame) == 8 for frame in frames)
    assert len(captured) == 1
    assert captured[0].shape == (5 + 3, 8)


def test_run_generate_no_fallback_when_no_post_eos_padding() -> None:
    """When ``post_eos_padding_count == 0``, the graph decoder's output is
    returned directly (no fall-through to eager). Frames come from the
    graph, not the eager tail. Locks the unchanged eager-fast-path
    semantics for the historic ``post_eos_padding_count=0`` callers.
    """
    fake_decoder = _FakeGraphDecoder(text_codes=[42, 43, 44], hidden_size=8)
    original_enable = cg.enable_cuda_graph
    cg.enable_cuda_graph = lambda *a, **kw: fake_decoder  # type: ignore[assignment]
    try:
        # Eager stream_generate must NOT be invoked.
        def boom(*a, **kw):  # noqa: ARG001
            raise AssertionError("eager stream_generate must not run when graph returns")

        original_stream = gen_module.stream_generate
        gen_module.stream_generate = boom  # type: ignore[assignment]
        try:
            model = SimpleNamespace(
                forward=object(),
                audio_pad_token=1,
                audio_stop_token=2,
                audio_spk_token=3,
                config=SimpleNamespace(audio_pad_token=1),
            )
            captured: list[torch.Tensor] = []
            frames = run_generate(
                model,
                torch.tensor([[1] * 5]),
                max_new_tokens=4,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=fake_decoder.eos_token_id,
                open_thinking=False,
                use_thinker_cuda_graph=True,
                post_eos_padding_count=0,  # no post-EOS -> graph's output is final
                capture_bridge_states=True,
                bridge_state_callback=captured.append,
            )
            # Frames come from the graph decoder (3 decode steps = 3 frames).
            assert len(frames) == 3
            assert all(len(f) == 8 for f in frames)
            # Graph's bridge callback fired (shape: prompt + decode_steps).
            assert len(captured) == 1
            assert captured[0].shape == (5 + 3, 8)
        finally:
            gen_module.stream_generate = original_stream  # type: ignore[assignment]
    finally:
        cg.enable_cuda_graph = original_enable  # type: ignore[assignment]


def test_run_generate_no_fallback_when_graph_runs_to_budget() -> None:
    """With ``post_eos_padding_count > 0``, graph output remains final even
    when the decoder runs to budget. Frames come from the graph decoder.
    """
    fake_decoder = _FakeGraphDecoder(text_codes=[42, 43, 44], hidden_size=8, stop_early=False)

    original_enable = cg.enable_cuda_graph
    cg.enable_cuda_graph = lambda *a, **kw: fake_decoder  # type: ignore[assignment]
    try:

        def boom(*a, **kw):  # noqa: ARG001
            raise AssertionError("eager stream_generate must not run when graph hit budget")

        original_stream = gen_module.stream_generate
        gen_module.stream_generate = boom  # type: ignore[assignment]
        try:
            model = SimpleNamespace(
                forward=object(),
                audio_pad_token=1,
                audio_stop_token=2,
                audio_spk_token=3,
                config=SimpleNamespace(audio_pad_token=1),
            )
            frames = run_generate(
                model,
                torch.tensor([[1] * 5]),
                max_new_tokens=4,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=fake_decoder.eos_token_id,
                open_thinking=False,
                use_thinker_cuda_graph=True,
                post_eos_padding_count=128,
                capture_bridge_states=True,
                bridge_state_callback=lambda b: None,
            )
            assert len(frames) == 3
        finally:
            gen_module.stream_generate = original_stream  # type: ignore[assignment]
    finally:
        cg.enable_cuda_graph = original_enable  # type: ignore[assignment]
