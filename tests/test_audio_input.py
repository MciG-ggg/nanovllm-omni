"""Focused unit tests for MiniMind-O audio input + ASR (Q1/Q2/Q4).

No weights, no funasr: the SenseVoice encoder/ASR is stubbed. We pin the
engine plumbing — dict-prompt audio -> ``<|audio_pad|>`` markers ->
prefill ``forward(audio_inputs=...)`` kwargs -> ``custom_output[transcript]``.

Everything torch/numpy heavy is ``importorskip``-gated so the no-torch CI
job skips these files; nothing here imports funasr.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.thinker import tokenize_for_generate  # noqa: E402

# ---------------------------------------------------------------------------
# tokenize: <|audio_pad|> markers are inserted only when audio is present
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self) -> None:
        self.captured: str | None = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.captured = messages[0]["content"]
        return messages[0]["content"]

    def __call__(self, text):
        return SimpleNamespace(data={"input_ids": [7, 8, 9]})


def test_tokenize_inserts_audio_markers_before_prompt() -> None:
    tok = _FakeTokenizer()
    tokenize_for_generate(tok, "hi", open_thinking=False, audio_markers=3)
    assert tok.captured == "<|audio_pad|>" * 3 + "\nhi"


def test_tokenize_markers_without_prompt() -> None:
    tok = _FakeTokenizer()
    tokenize_for_generate(tok, "", open_thinking=False, audio_markers=2)
    assert tok.captured == "<|audio_pad|>" * 2


def test_tokenize_no_markers_keeps_plain_prompt() -> None:
    tok = _FakeTokenizer()
    tokenize_for_generate(tok, "hi", open_thinking=False)
    assert tok.captured == "hi"


# ---------------------------------------------------------------------------
# engine prefill: audio rides into model.forward as audio_inputs / audio_lens
# ---------------------------------------------------------------------------


class _FakeAudioModel:
    """MiniMind-omni-shaped fake whose forward records kwargs."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_key_value_heads=2, head_dim=4, max_position_embeddings=64, think_end_ids=[]
        )
        self.thinker = SimpleNamespace(layers=[None] * 2)
        self.talker = SimpleNamespace(layers=[None] * 2)
        self.audio_pad_token = 2049
        self.audio_stop_token = 3000
        self.enter_token_id = 3001
        self.pad_token_id = 1
        self._params = [torch.zeros(1, dtype=torch.float32)]
        self.calls: list[dict] = []  # forward kwargs snapshot

    def parameters(self):
        return iter(self._params)

    def forward(self, input_ids, **kwargs):
        self.calls.append(dict(kwargs))
        bs, _, tlen = input_ids.shape
        num_layers = len(self.thinker.layers) + len(self.talker.layers)
        presents = [
            (torch.zeros(bs, tlen, 2, 4), torch.zeros(bs, tlen, 2, 4)) for _ in range(num_layers)
        ]
        logits = torch.zeros(bs, tlen, 4096)
        audio_logits = [torch.zeros(bs, tlen, 4096) for _ in range(8)]
        return SimpleNamespace(logits=logits, audio_logits=audio_logits, past_key_values=presents)


def _make_runner(model):
    from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner
    from nanovllm_omni.models.minimind_omni.runtime_scheduler import RuntimeScheduler

    sched = RuntimeScheduler(max_num_seqs=4)
    runner = BatchedThinkerRunner(
        SimpleNamespace(model=model),
        sched,
        temperature=0.75,
        top_p=1.0,
        rp=1.0,
        max_new_tokens=4,
        open_thinking=False,
        base_seed=7,
    )
    return runner, sched


def test_audio_prefill_kwargs_pads_mixed_group() -> None:
    runner, _ = _make_runner(_FakeAudioModel())
    a = runner.add_request(
        [1, 2, 3], audio_inputs=torch.randn(1, 4, 80), audio_lens=torch.tensor([3])
    )
    b = runner.add_request([4, 5, 6])
    kw = runner._audio_prefill_kwargs([a, b])
    assert kw["audio_inputs"].shape == (2, 4, 80)
    assert kw["audio_lens"].tolist() == [3, 1]
    # audio-free row is zero-filled -> model's batch mask drops it
    assert kw["audio_inputs"][1].abs().sum().item() == 0


def test_audio_prefill_kwargs_empty_without_audio() -> None:
    runner, _ = _make_runner(_FakeAudioModel())
    a = runner.add_request([1, 2, 3])
    assert runner._audio_prefill_kwargs([a]) == {}


def test_prefill_group_forwards_audio_kwargs_to_forward() -> None:
    model = _FakeAudioModel()
    runner, _ = _make_runner(model)
    a = runner.add_request(
        [1, 2, 3], audio_inputs=torch.randn(1, 4, 80), audio_lens=torch.tensor([4])
    )
    fake_group = SimpleNamespace(
        items=[SimpleNamespace(sequence=SimpleNamespace(request_id=a), end=3)]
    )
    runner.prefill_group(fake_group)
    assert len(model.calls) == 1
    kw = model.calls[0]
    assert kw["audio_inputs"].shape == (1, 4, 80)
    assert kw["audio_lens"].tolist() == [4]


# ---------------------------------------------------------------------------
# stream_generate threads audio to the runner
# ---------------------------------------------------------------------------


def test_stream_generate_forwards_audio_to_add_request(monkeypatch) -> None:
    import nanovllm_omni.models.minimind_omni.generation as gen

    captured: dict = {}

    class _FakeRunner:
        def __init__(self, *a, **kw) -> None:
            pass

        def add_request(self, prompt_ids, request_id=None, **kw):
            captured["kw"] = kw
            return "r1"

    monkeypatch.setattr(gen, "BatchedThinkerRunner", _FakeRunner)
    list(gen.stream_generate(None, torch.tensor([[1, 2, 3]]), audio_inputs="AI", audio_lens="AL"))
    assert captured["kw"] == {"audio_inputs": "AI", "audio_lens": "AL"}


# ---------------------------------------------------------------------------
# outputs: transcript on the payload surfaces as custom_output["transcript"]
# ---------------------------------------------------------------------------


def test_from_pipeline_surfaces_transcript() -> None:
    from nanovllm_omni.outputs import AudioPayload, OmniRequestOutput

    # thinker_forward wraps the frozen AudioPayload when a transcript exists
    payload = SimpleNamespace(
        audio=AudioPayload(data=b"RIFF", sample_rate=24000), transcript="你好"
    )
    out = OmniRequestOutput.from_pipeline(payload, final_output_type="audio")
    assert out.custom_output == {"transcript": "你好"}
    assert "audio" in out.multimodal_output


def test_from_pipeline_without_transcript() -> None:
    from nanovllm_omni.outputs import AudioPayload, OmniRequestOutput

    out = OmniRequestOutput.from_pipeline(
        AudioPayload(data=b"RIFF", sample_rate=24000), final_output_type="audio"
    )
    assert out.custom_output == {}


# ---------------------------------------------------------------------------
# audio.py helpers that need no funasr
# ---------------------------------------------------------------------------


def test_audio_encoder_attach_requires_local_path() -> None:
    from nanovllm_omni.models.minimind_omni.audio import attach_audio_encoder

    bundle = SimpleNamespace(model=object(), device="cpu")
    for bad in (None, "/nonexistent/sensevoice"):
        with pytest.raises(ValueError, match="SenseVoice"):
            attach_audio_encoder(bundle, bad)


def test_load_audio_decodes_wav_bytes_and_resamples() -> None:
    np = pytest.importorskip("numpy")
    import io
    import wave

    from nanovllm_omni.models.minimind_omni.audio import SENSEVOICE_SAMPLE_RATE, load_audio

    def _wav_bytes(rate: int) -> bytes:
        t = np.arange(rate) / rate
        pcm = (np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        return buf.getvalue()

    out = load_audio(_wav_bytes(SENSEVOICE_SAMPLE_RATE))
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(out).max() <= 1.0
    resampled = load_audio(_wav_bytes(8000))
    # 1 s of 8 kHz -> 16 kHz == 16 k samples, same as the native-16 k file
    assert len(resampled) == len(out)
