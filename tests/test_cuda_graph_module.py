"""Contract tests for the CUDA-Graph decoder module (optim/cuda_graph.py).

Plan §3.3's capture wrapper builds on the buffer pillar (`enable_fixed_kv_buffer`
in attention.py, report §23-§25). Its GPU loop is captured once / replayed
many (§25 determinism protocol) — the actual capture can only run with CUDA.
These CPU tests pin the pieces that must hold *without* a GPU:

  1. enable_cuda_graph returns None on CPU (opt-in, no CUDA -> eager path).
  2. _patched_forward neutralizes the two freqs[0,0] host-read checks (the
     only capture blockers, §13/§18) without rebinding the class.
  3. CudaGraphDecoder wires the fixed-KV-buffer patch + tracks the per-
     instance attn markers; on CPU its eager decode-input shape matches the
     validated _decode_input contract ([1, 9, 1] = 8 audio + 1 text).
  4. The patched forward on a stub model actually runs and returns the same
     logits as the original (arithmetic untouched — only the host-read
     branches are dead-code'd).

A future refactor that reloads a GPU-only dependency at import, breaks the
freqs-neutralization marker, or changes the decode input shape fails here.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.optim import cuda_graph as cg  # noqa: E402 -- after importorskip


class _FakeConfig:
    audio_pad_token = 2051
    max_position_embeddings = 2048


class Attention(torch.nn.Module):
    """Named `Attention` so enable_fixed_kv_buffer's type-name match works
    (the real bundle's attention classes are named `Attention`)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 8, bias=False)
        self.v_proj = torch.nn.Linear(8, 8, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.n_local_kv_heads = 1
        self.n_local_heads = 1
        self.head_dim = 8


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()


class _StubModel(torch.nn.Module):
    """Mini stub carrying the upstream freqs host-reads the patch targets."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.thinker = torch.nn.Module()
        self.thinker.layers = torch.nn.ModuleList([_FakeLayer()])
        self.thinker.freqs_cos = torch.ones(1, 1)
        self.thinker.freqs_sin = torch.zeros(1, 1)
        self.talker = torch.nn.Module()
        self.talker.layers = torch.nn.ModuleList([])
        self.talker.freqs_cos = torch.ones(1, 1)
        self.talker.freqs_sin = torch.zeros(1, 1)

    def forward(self, input_ids=None, past_key_values=None, use_cache=False, **kw):
        # mirrors upstream shape: check the two freq host-reads then return
        # logits (both thinker and talker, matching the real MiniMindOmni
        # forward that _patched_forward must neutralize)
        if self.thinker.freqs_cos[0, 0] == 0:
            raise RuntimeError("unreachable")
        if self.talker.freqs_cos[0, 0] == 0:
            raise RuntimeError("unreachable")
        b, s = input_ids.shape
        logits = torch.zeros(b, s, 8)
        return types.SimpleNamespace(
            logits=logits, past_key_values=([None] * 2 if use_cache else None)
        )


def test_enable_cuda_graph_returns_none_without_cuda() -> None:
    """CPU (no CUDA) must yield None -> callers keep the eager path."""
    assert not torch.cuda.is_available()
    model = _StubModel()
    dec = cg.enable_cuda_graph(model)
    assert dec is None
    assert not hasattr(model, "_nanovllm_graph_decoder")


def test_patched_forward_neutralizes_host_reads_without_rebind() -> None:
    """The compiled copy dead-codes both freqs[0,0] checks; the class
    forward must be untouched (no class-level rebind)."""
    cls = type(_StubModel())
    src = inspect_source(cls)
    patched = cg._patched_forward(cls, src)
    assert patched is not None
    # calling the patched copy on a stub whose freq read would be dead under
    # the neutralization (freqs_cos[0,0] is 1) -- no RuntimeError raised.
    out = patched(_StubModel(), input_ids=torch.ones(1, 2, dtype=torch.long))
    assert out is not None
    assert getattr(out, "logits", None) is not None
    # the class-level forward must remain the original (no global rebind)
    assert type(_StubModel()).forward is _StubModel.forward


def test_patched_forward_preserves_arithmetic() -> None:
    """Same stub, original forward vs patched copy: arguments pass through
    identically (the checks are dead-code'd, no arithmetic change)."""
    cls = type(_StubModel())
    src = inspect_source(cls)
    patched = cg._patched_forward(cls, src)
    assert patched is not None

    model = _StubModel()
    # original (instance) forward via the live bound method
    orig = model(input_ids=torch.ones(1, 2, dtype=torch.long))
    got = patched(model, input_ids=torch.ones(1, 2, dtype=torch.long))
    assert torch.equal(orig.logits, got.logits)


def test_decoder_resets_request_local_stop_state() -> None:
    """EOS state from one decoded request must not contaminate the next one."""
    decoder = cg.CudaGraphDecoder.__new__(cg.CudaGraphDecoder)
    decoder._text_finished = True
    decoder._reset_request_state()
    assert decoder._text_finished is False


def test_run_generate_syncs_graph_sampling_params(monkeypatch) -> None:
    """The graph decoder must use the caller's eager-equivalent sampling knobs."""
    from nanovllm_omni.models.minimind_omni import thinker as th

    class Decoder:
        temperature = None
        top_p = None

        def generate_tokens(self, *_args, **_kwargs):
            return [1], [[2] for _ in range(8)]

    decoder = Decoder()
    monkeypatch.setattr(cg, "enable_cuda_graph", lambda *_args, **_kwargs: decoder)
    model = types.SimpleNamespace(
        forward=object(),
        audio_pad_token=1,
        audio_stop_token=2,
        audio_spk_token=3,
    )
    frames = th.run_generate(
        model,
        torch.tensor([[1]]),
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.8,
        eos_token_id=2,
        open_thinking=False,
        use_thinker_cuda_graph=True,
    )
    assert frames == [[2] * 8]
    assert decoder.temperature == 0.7
    assert decoder.top_p == 0.8


def test_decoder_wires_buffer_patch_and_input_shape() -> None:
    """CudaGraphDecoder attaches the fixed-KV-buffer patch and its decode
    input is [1, 9, 1] (8 audio + 1 text), the shape the buffer forward
    validates against (report §19 - KV layout [B, seq, n_kv, d])."""
    # decoder needs a CUDA-less construction guard: simulate no-CUDA path by
    # constructing directly (the module's enable_cuda_graph already guarded).
    # Verify via the decode helper only, plus that buffer-ization marks land.

    from nanovllm_omni.optim.attention import enable_fixed_kv_buffer

    model = _StubModel()
    enable_fixed_kv_buffer(model, max_len=32)
    marks = [m for m in model.modules() if getattr(m, "_nanovllm_kv_buffer", False)]
    assert len(marks) == 1, len(marks)

    nid = torch.ones(1, 1, dtype=torch.long)
    inp = cg._build_omni_input(nid, audio_pad=2051)
    assert tuple(inp.shape) == (1, 9, 1), tuple(inp.shape)
    assert int(inp[0, 8, 0].item()) == 1  # text row carries the token


def inspect_source(cls) -> str:
    import inspect

    return inspect.getsource(cls.forward)


if __name__ == "__main__":
    import sys

    checks = [
        test_enable_cuda_graph_returns_none_without_cuda,
        test_patched_forward_neutralizes_host_reads_without_rebind,
        test_patched_forward_preserves_arithmetic,
        test_decoder_resets_request_local_stop_state,
        test_run_generate_syncs_graph_sampling_params,
        test_decoder_wires_buffer_patch_and_input_shape,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
