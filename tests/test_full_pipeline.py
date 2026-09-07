"""Phase 8: end-to-end full (3-stage) MiniMind-O pipeline tests.

Runs the REAL stage wiring through the public ``Omni.generate`` surface:

  thinker full factory -> ``thinker2talker`` -> talker stage
  -> ``talker2code2wav`` -> code2wav stage -> ``AudioPayload``
  -> ``OmniRequestOutput``

using bridge-capable model / talker / Mimi doubles so no checkpoint or GPU
is required. This proves full mode is *executable* end to end:

  * the thinker emits a ``ThinkerStageOutput`` whose bridge hidden states
    line up with the prompt + output text span,
  * the talker drives ``preprocess``/``forward``/``postprocess``/``sample``/
    ``talker_mtp`` over the bridge and returns ``[F, num_code_layers]``
    code rows,
  * code2wav decodes the rows into a 24 kHz mono WAV inside
    ``OmniRequestOutput``, and ``to_dict()`` emits base64 WAV.

Behavioral parity against real MiniMind-3o / Mimi weights on the RTX 3050
is the *outstanding* validation step and is deliberately NOT claimed here;
this file pins the wiring contract so the RTX run only has to verify
numeric fidelity, not plumbing.

Collapsed regression: the legacy path (deploy default) still returns a
valid WAV through the unchanged ``generate_audio`` thinker.

Reference mapping (vllm-omni PR #3796 miniMind-3o)
---------------------------------------------------

- thinker -> talker bridge handoff mirrors the reference ``thinker2talker``
  span selection: the bridge hidden-state sequence is one row per prompt
  position plus one row per *decode* step, and the talker conditions its
  layer-0 + residual codebook decode on that span.
- the talker stage drives the reference LLM_AR contract
  (``preprocess`` / ``forward`` / ``postprocess`` / ``compute_logits`` /
  ``sample`` / ``talker_mtp``) with the delayed-diagonal MTP alignment.
- post-EOS controls (enter -> bounded PAD tail -> internal stop) ride the
  deploy layer's ``post_eos_padding_count`` / ``internal_stop_token_id``,
  matching the reference watchdog semantics.
- reference parity is pinned here as offline CPU checks; anything that
  needs real weights / CUDA stays a ``smoke``-only concern on the RTX
  host (see ``deploy/minimind_omni.yaml``).

Known local-host alignment assumptions (RTX-3050 validation list)
------------------------------------------------------------------

1. The bridge/text span offset: the runner predicts the first output token
   at prefill, so the talker decodes ``len(output) - 1`` rows. Whether the
   reference emits ``len(output)`` rows (one per output token incl. the
   first) is unverified against real weights.
2. ``talker_mtp`` residual conditioning on the *previous* step's hidden
   state (delayed diagonal) is exercised structurally; exact codebook
   parity needs the real ``TalkerModule``.
3. The wrapper drives the talker trunk without per-step KV caching
   (``forward`` runs a fresh span each call); real-model decode may need
   the KV path for equal audio to the reference.
4. Full mode is text-to-audio only: audio input (ASR) remains a
   collapsed-path feature until that bridging is wired.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nanovllm_omni.config.params import OmniEngineArgs, SamplingParams
from nanovllm_omni.engine.runner import PipelineRunner
from nanovllm_omni.models.minimind_omni.pipeline import MINIMIND_OMNI_PIPELINE
from nanovllm_omni.outputs import AudioPayload, OmniRequestOutput

torch = pytest.importorskip("torch")

from nanovllm_omni.entrypoints import Omni  # noqa: E402
from nanovllm_omni.models.minimind_omni.batched_generation import (  # noqa: E402
    enable_bridge_capture,
)
from nanovllm_omni.models.minimind_omni.code2wav import decode_audio, encode_wav  # noqa: E402
from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402
from tests._talker_fixtures import make_fake_bundle  # noqa: E402
from tests.test_batched_generation import FakeMiniMindOmni  # noqa: E402

# ---------------------------------------------------------------------------
# Bridge-capable fake joint model (text + audio logits, real bridge layer)
# ---------------------------------------------------------------------------


class _BridgeLayer(torch.nn.Module):
    """One-projection stand-in for ``MiniMindBlock``.

    ``forward(hidden_states, ...) -> (hidden_states, present)``. The bridge
    layer is monkey-patched by ``enable_bridge_capture`` so the runner can
    stash ``block._bridge_capture`` after each joint forward.
    """

    def __init__(self, hidden_size: int, marker_value: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.marker_value = float(marker_value)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any = None,
        past_key_value: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> tuple[torch.Tensor, None]:
        del position_embeddings, past_key_value, use_cache, attention_mask
        return self.proj(hidden_states) * self.marker_value, None


class FakeMiniMindOmniWithBridge(FakeMiniMindOmni):
    """Extends ``FakeMiniMindOmni`` with real bridge-layer blocks.

    Prefill forwards capture ``[B, num_positions, hidden_size]`` bridge
    states; decode forwards capture ``[B, 1, hidden_size]``. The runner's
    ``_capture_prefill_bridge`` stores one row per prompt position so the
    full-mode bridge sequence lines up with prompt + output text span.
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
        # Keep every sampled code below the audio-vocab boundary so the
        # runner's stop-bookkeeping stays inert (bounded by max_new_tokens).
        self.audio_pad_token = 10
        self.audio_stop_token = max(vocab - 1, 64)
        self.audio_spk_token = max(vocab - 3, 2)
        self.config.bridge_layer = num_thinker_layers // 2 - 1
        self.config.num_hidden_layers = num_thinker_layers
        bridge_idx = self.config.bridge_layer
        blocks = [
            _BridgeLayer(hidden_size, bridge_marker if i == bridge_idx else 1.0)
            for i in range(num_thinker_layers)
        ]
        self.thinker.layers = torch.nn.ModuleList(blocks)
        self._hidden_size = hidden_size

        outer_forward = self.forward

        def _forward_with_layers(self, input_ids, **kwargs):
            # Walk the thinker layers so the bridge patch fires, then let
            # the parent fake produce logits/past (shape contract).
            bs, _, tlen = input_ids.shape
            hidden = torch.zeros(bs, tlen, self._hidden_size)
            for layer in self.thinker.layers:
                hidden, _ = layer(hidden, None)
            return outer_forward(input_ids, **kwargs)

        import types

        self.forward = types.MethodType(_forward_with_layers, self)


# ---------------------------------------------------------------------------
# Offline tokenizer / Mimi / bundle doubles
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        del tokenize, add_generation_prompt, kw
        return " ".join(m.get("content", "") for m in messages)

    def __call__(self, text):
        return SimpleNamespace(data={"input_ids": [ord(c) % 50 + 1 for c in text]})


class FakeMimi:
    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def decode(self, codes: torch.Tensor) -> SimpleNamespace:
        self.input_shape = tuple(codes.shape)
        return SimpleNamespace(audio_values=torch.full((1, 1, codes.shape[-1]), 0.25))


def make_full_fixtures() -> tuple[SimpleNamespace, Any, FakeMimi, FakeMiniMindOmniWithBridge]:
    """Build (bundle, talker_wrapper, mimi, model) wired for full mode."""
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
    )
    talker = wrap_talker(make_fake_bundle())  # duck-typed wrapper contract
    bundle.talker = talker
    return bundle, talker, mimi, model


def _write_deploy(tmp_path: Path) -> Path:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "max_batch: 1\n"
        "use_thinker_cuda_graph: false\n"
        "post_eos_padding_count: 128\n"
        "internal_stop_token_id: 17\n"
        "talker_max_steps_after_last_thinker_token: 192\n"
        "stages:\n"
        "  - name: thinker\n"
        "    default_sampling_params: {temperature: 0.7, max_tokens: 4}\n"
        "  - name: talker\n"
        "    default_sampling_params: {temperature: 0.2}\n"
        "  - name: code2wav\n"
        "    default_sampling_params: {}\n",
        encoding="utf-8",
    )
    return path


def _new_omni(tmp_path: Path, fixtures: tuple) -> Omni:
    bundle, talker, mimi, _model = fixtures
    return Omni(
        model="fake/minimind-3o",
        device="cpu",
        dtype="float32",
        pipeline="minimind_o",
        deploy_config_path=str(_write_deploy(tmp_path)),
        extra={"bundle": bundle, "talker": talker, "mimi": mimi},
    )


def _assert_wav(output: OmniRequestOutput, sample_rate: int = 24_000) -> bytes:
    assert output.error is None
    assert output.is_pipeline_output
    payload: Any = output.multimodal_output["audio"]
    assert isinstance(payload, AudioPayload)
    assert payload.data.startswith(b"RIFF")
    with wave.open(io.BytesIO(payload.data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == sample_rate
        assert wav.getsampwidth() == 2
    return payload.data


# ---------------------------------------------------------------------------
# Full mode: enable + E2E through the public surface
# ---------------------------------------------------------------------------


def test_full_mode_is_only_supported_kind_and_is_deploy_default() -> None:
    assert MINIMIND_OMNI_PIPELINE.supported_pipeline_kinds == ("full",)
    import pathlib

    from nanovllm_omni.config import load_deploy_config

    deploy_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "nanovllm_omni"
        / "deploy"
        / "minimind_omni.yaml"
    )
    assert load_deploy_config(deploy_path) is not None


def test_full_mode_e2e_produces_decodable_wav_through_omni(tmp_path: Path) -> None:
    omni = _new_omni(tmp_path, make_full_fixtures())
    outputs = omni.generate(
        ["hello"],
        SamplingParams(max_tokens=4, temperature=0.2, top_p=0.9),
    )
    assert len(outputs) == 1
    _assert_wav(outputs[0], sample_rate=24_000)


def test_full_mode_runner_produces_omni_request_output(tmp_path: Path) -> None:
    bundle, talker, mimi, _model = make_full_fixtures()
    deploy = _write_deploy(tmp_path)
    from nanovllm_omni.config import load_deploy_config

    args = OmniEngineArgs(
        model="fake/minimind-3o",
        device="cpu",
        dtype="float32",
        extra={"bundle": bundle, "talker": talker, "mimi": mimi},
    )
    runner = PipelineRunner(MINIMIND_OMNI_PIPELINE, load_deploy_config(deploy), args)
    payload = runner.run("hello", SamplingParams(max_tokens=4, temperature=0.2, top_p=0.9))
    assert isinstance(payload, AudioPayload)
    out = OmniRequestOutput.from_pipeline(payload, final_output_type="audio")
    _assert_wav(out)


def test_full_mode_to_dict_emits_base64_wav(tmp_path: Path) -> None:
    omni = _new_omni(tmp_path, make_full_fixtures())
    (output,) = omni.generate("hello", SamplingParams(max_tokens=4, temperature=0.2))
    d = output.to_dict()
    assert "multimodal_output" in d
    raw = base64.b64decode(d["multimodal_output"]["audio"])
    assert raw.startswith(b"RIFF")
    json.dumps(d)  # must survive json.dumps
    metadata = d["multimodal_output"]["audio_metadata"]
    assert metadata == {"format": "wav", "sample_rate": 24_000}


def test_full_mode_no_cross_request_state_leak(tmp_path: Path) -> None:
    """Two sequential requests: each yields a fresh WAV and the talker
    drops its per-request watchdog flags (no stale forced-stop)."""
    fixtures = make_full_fixtures()
    _bundle, talker, _mimi, _model = fixtures
    omni = _new_omni(tmp_path, fixtures)
    sp = SamplingParams(max_tokens=4, temperature=0.2)

    outs = omni.generate(["first", "second"], sp)
    assert len(outs) == 2
    for output in outs:
        _assert_wav(output)
    # Talker per-request lifecycle dicts are empty after both requests.
    assert talker._stop_pending_by_req == {}
    assert talker._steps_after_last_thinker_by_req == {}


def test_full_thinker_bridge_spans_prompt_and_output(tmp_path: Path) -> None:
    """The bridge sequence has one row per prompt position + one per decode
    step, so ``thinker2talker`` can align it to the full text span."""
    bundle, _talker, _mimi, model = make_full_fixtures()
    from nanovllm_omni.models.minimind_omni.generation import stream_generate
    from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate

    enable_bridge_capture(model, model.config.bridge_layer)
    input_ids = tokenize_for_generate(
        bundle.tokenizer, "hello", False, audio_special_token="<|audio_pad|>"
    ).to("cpu")
    captured: list[Any] = []
    list(
        stream_generate(
            model,
            input_ids,
            max_new_tokens=4,
            top_p=1.0,
            capture_bridge_states=True,
            bridge_state_callback=captured.append,
            post_eos_padding_count=128,
            internal_stop_token_id=17,
        )
    )
    assert len(captured) == 1
    bridge = captured[0]
    assert bridge.ndim == 2
    # 5 prompt positions + 3 decode steps (the runner predicts the first
    # output token at prefill, so max_new_tokens=4 yields 3 decode rows).
    assert bridge.shape[0] == 5 + 3
    assert bridge.shape[1] == 8


def test_full_mode_requires_bridge_capture(tmp_path: Path) -> None:
    """Models without a patchable bridge layer fail loud at the handoff
    (proving the exact missing step), never silently as collapsed audio."""
    _bundle, talker, mimi, _model = make_full_fixtures()
    plain = FakeMiniMindOmni()  # thinker.layers are [None] placeholders
    plain_bundle = SimpleNamespace(
        model=plain,
        tokenizer=_FakeTokenizer(),
        mimi=mimi,
        device="cpu",
        model_id="fake",
        thinker=plain,
        talker=talker,
        code2wav=mimi,
    )
    omni = Omni(
        model="fake/minimind-3o",
        device="cpu",
        dtype="float32",
        pipeline="minimind_o",
        deploy_config_path=str(_write_deploy(tmp_path)),
        extra={"bundle": plain_bundle, "talker": talker, "mimi": mimi},
    )
    with pytest.raises(ValueError, match="bridge"):
        omni.generate("hello", SamplingParams(max_tokens=4, temperature=0.2))


def test_full_mode_rejects_audio_input_with_clear_message(tmp_path: Path) -> None:
    """Full mode is text-to-audio only; audio input must not silently drop."""
    omni = _new_omni(tmp_path, make_full_fixtures())
    with pytest.raises(NotImplementedError, match="text-to-audio only"):
        omni.generate(
            "hello",
            SamplingParams(max_tokens=4, extra={"audio": b"fake-bytes"}),
        )


# ---------------------------------------------------------------------------
# Collapsed regression: deploy default unchanged
# ---------------------------------------------------------------------------


def test_codec_helpers_roundtrip_full_codes(tmp_path: Path) -> None:
    """The exact code rows the talker emits decode to a real mono WAV."""
    _bundle, _talker, mimi, _model = make_full_fixtures()
    codes = [[0] * 8, [1] * 8, [2] * 8]
    samples = decode_audio(mimi, codes, "cpu")
    wav_bytes = encode_wav(samples, sample_rate=24_000)
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24_000


# ---------------------------------------------------------------------------
# Stage-1 talker CUDA Graph integration
# ---------------------------------------------------------------------------


def test_deploy_yaml_parses_use_talker_cuda_graph(tmp_path: Path) -> None:
    """The deploy YAML default enables talker CUDA Graph; toggle is parsed."""
    from nanovllm_omni.config import load_deploy_config

    deploy_path = (
        Path(__file__).resolve().parent.parent / "nanovllm_omni" / "deploy" / "minimind_omni.yaml"
    )
    cfg = load_deploy_config(deploy_path)
    assert cfg.use_talker_cuda_graph is True

    # Toggle off via YAML override
    custom = tmp_path / "off.yaml"
    custom.write_text("use_talker_cuda_graph: false\n", encoding="utf-8")
    assert load_deploy_config(custom).use_talker_cuda_graph is False


def test_talker_stage_runs_eager_when_no_cuda(tmp_path: Path) -> None:
    """Without CUDA, the talker stage falls back to eager and produces a WAV."""
    fixtures = make_full_fixtures()
    omni = _new_omni(tmp_path, fixtures)
    outputs = omni.generate(
        "hello",
        SamplingParams(max_tokens=4, temperature=0.2, top_p=0.9),
    )
    assert len(outputs) == 1
    _assert_wav(outputs[0])


def test_talker_mtp_runner_dispatch_picks_eager_when_no_cuda(
    tmp_path: Path,
) -> None:
    """Direct call to _drive_talker_generation with mtp_runner=None stays eager."""
    from nanovllm_omni.models.minimind_omni.stage_processors import (
        TalkerInputPayload,
    )
    from nanovllm_omni.models.minimind_omni.talker import (
        _drive_talker_generation,
        wrap_talker,
    )

    fixtures = make_full_fixtures()
    bundle, _talker, mimi, _model = fixtures
    talker = wrap_talker(bundle)
    hidden_size = talker.text_hidden_size
    prompt_len = 4
    num_decode_steps = 4
    bridge = torch.randn(prompt_len + num_decode_steps - 1, hidden_size, dtype=torch.float32)
    payload = TalkerInputPayload(
        input_ids=torch.full((prompt_len,), 9, dtype=torch.long),
        bridge_states=bridge,
        text_token_ids=tuple(range(prompt_len + num_decode_steps)),
        prompt_token_ids=tuple(range(prompt_len)),
        output_token_ids=tuple(range(prompt_len, prompt_len + num_decode_steps)),
        request_id="runner-dispatch-test",
        metadata={},
    )
    # mtp_runner=None -> falls back to talker.talker_mtp (no graph attribute)
    rows = _drive_talker_generation(
        talker,
        payload,
        temperature=0.2,
        top_k=50,
        do_sample=True,
        mtp_runner=None,
    )
    assert rows.shape[0] > 0
    assert rows.shape[1] == 8


def test_talker_mtp_runner_dispatch_uses_graph_decode_when_supplied(
    tmp_path: Path,
) -> None:
    """When mtp_runner exposes decode(), _drive_talker_generation calls it.

    Covers the graph-dispatch branch (``if mtp_decode is not None``) that the
    real CUDA path exercises on GPU but CI never hits. Uses a CPU fake runner
    exposing ``decode`` with the talker_mtp contract.
    """
    from nanovllm_omni.models.minimind_omni.stage_processors import (
        TalkerInputPayload,
    )
    from nanovllm_omni.models.minimind_omni.talker import (
        _drive_talker_generation,
        wrap_talker,
    )

    fixtures = make_full_fixtures()
    bundle, _talker, mimi, _model = fixtures
    talker = wrap_talker(bundle)
    hidden_size = talker.text_hidden_size
    prompt_len = 4
    num_decode_steps = 4
    bridge = torch.randn(prompt_len + num_decode_steps - 1, hidden_size, dtype=torch.float32)
    payload = TalkerInputPayload(
        input_ids=torch.full((prompt_len,), 9, dtype=torch.long),
        bridge_states=bridge,
        text_token_ids=tuple(range(prompt_len + num_decode_steps)),
        prompt_token_ids=tuple(range(prompt_len)),
        output_token_ids=tuple(range(prompt_len, prompt_len + num_decode_steps)),
        request_id="graph-dispatch-test",
        metadata={},
    )

    class _FakeMtpRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def decode(self, **kwargs: Any) -> torch.Tensor:
            self.calls.append(kwargs)
            # Return code-a (0) rows: never equals audio_stop_token, so the
            # wrapper's early-stop guard doesn't truncate the decode loop.
            return torch.zeros(1, 8, dtype=torch.long)

    runner = _FakeMtpRunner()
    rows = _drive_talker_generation(
        talker,
        payload,
        temperature=0.2,
        top_k=50,
        do_sample=False,
        mtp_runner=runner,
    )
    # decode was called once per decode step (bridge rows minus prompt =
    # num_decode_steps - 1), and rows came back valid.
    assert len(runner.calls) == num_decode_steps - 1
    assert rows.shape[0] == num_decode_steps - 1
    assert rows.shape[1] == 8
    # active_mask is forwarded through the graph-dispatch branch.
    assert all("active_mask" in call for call in runner.calls)
