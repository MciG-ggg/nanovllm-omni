"""Batched scheduler + runner tests with a fake MiniMind-O model (no weights).

Verifies the three properties that would be invisible in runtime smoke:

1. ``OmniScheduler`` groups prefill by identical prompt length and decode by
   identical KV length (Q8a -- the model only supports one scalar start_pos).
2. A decode group really runs ONE batched forward of shape [B, 9, 1] (Q5a),
   and KV is gathered/written back through the fixed slots (Q6a).
3. Q10a: per-request results are identical whether the request runs batched
   with neighbours or solo -- each request owns its RNG (generator), so the
   batch composition never changes its draw order.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanovllm_omni.engine.sched import (
    FixedKvSlotPool,
    OmniScheduler,
)

torch = pytest.importorskip("torch")


class _FakeOut:
    def __init__(self, logits, audio_logits, past_key_values):
        self.logits = logits
        self.audio_logits = audio_logits
        self.past_key_values = past_key_values


class FakeMiniMindOmni:
    """Mimics the vendored MiniMindOmni.forward contract.

    KV layout matches the real model: ``[B, seq, kv_heads, head_dim]`` with
    seq at dim 1 (``start_pos = past[0][0].shape[1]``). Logits are uniform so
    sampling consumes each request's RNG and needs a seeded generator.
    """

    def __init__(self, n_thinker=2, n_talker=2, kv_heads=2, head_dim=4, vocab=4096):
        self.config = SimpleNamespace(
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            max_position_embeddings=256,
            think_end_ids=[],
        )
        self.thinker = SimpleNamespace(layers=[None] * n_thinker)
        self.talker = SimpleNamespace(layers=[None] * n_talker)
        self.audio_pad_token = 2049  # real MiniMind-O: >= AUDIO_VOCAB_BOUNDARY (2048)
        self.audio_stop_token = vocab - 1
        self.enter_token_id = vocab - 2
        self.pad_token_id = 1
        self._params = [torch.zeros(1, dtype=torch.float32)]  # device/dtype anchor
        self.n_thinker = n_thinker
        self.n_talker = n_talker
        self.n_layers = n_thinker + n_talker
        self.vocab = vocab
        self.calls: list[tuple[int, int]] = []  # (batch, seq_len) per forward

    def parameters(self):
        return iter(self._params)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        **kwargs,
    ):
        bs, _, tlen = input_ids.shape
        is_decode = past_key_values is not None
        self.calls.append((bs, tlen, is_decode))
        past_len = past_key_values[0][0].shape[1] if is_decode else 0
        seq_out = tlen if not is_decode else past_len + tlen
        presents = []
        for _ in range(self.n_layers):
            k = torch.zeros(bs, seq_out, self.config.num_key_value_heads, self.config.head_dim)
            v = torch.zeros(bs, seq_out, self.config.num_key_value_heads, self.config.head_dim)
            presents.append((k, v))
        logits = torch.zeros(bs, tlen, self.vocab)  # uniform -> RNG-driven sampling
        audio_logits = [torch.zeros(bs, tlen, self.vocab) for _ in range(8)]
        return _FakeOut(logits=logits, audio_logits=audio_logits, past_key_values=presents)


def _make_runner(model, prompts, **kwargs):
    from nanovllm_omni.models.minimind_omni.batched_generation import BatchedThinkerRunner

    sched = OmniScheduler(max_batch=kwargs.pop("max_batch", 2), max_seq=256)
    runner = BatchedThinkerRunner(
        SimpleNamespace(model=model),
        sched,
        temperature=0.75,
        top_p=1.0,
        rp=1.0,
        max_new_tokens=6,
        open_thinking=False,
        base_seed=7,
    )
    rids = []  # noqa
    for p in prompts:
        rid = runner.add_request(p)
        rids.append(rid)
    return runner, sched, rids


def _drain(runner, sched):
    """Run the engine loop (schedule -> prefill/decode -> update) to completion."""
    finished = set()
    while sched.has_requests():
        out = sched.schedule()
        if out.is_empty:
            break
        prefilled, generated = set(), {}
        for g in out.prefill_groups:
            runner.prefill_group(g)
            prefilled.update(g.req_ids)
        for g in out.decode_groups:
            runner.decode_group(g)
            for rid in g.req_ids:
                generated[rid] = runner.states[rid].step
                if runner.step_finished(rid):
                    finished.add(rid)
        sched.update_from_output(prefilled=prefilled, generated=generated, finished=finished)
    return finished


# -- scheduler grouping ------------------------------------------------------


def test_prefill_groups_by_prompt_length_and_decode_by_kv_length():
    model = FakeMiniMindOmni()
    runner, sched, (a, b) = _make_runner(model, [[1, 2, 3], [4, 5, 3]])

    out = sched.schedule()
    assert len(out.prefill_groups) == 1  # same length -> one batch
    assert out.prefill_groups[0].req_ids == [a, b]
    assert not out.decode_groups  # not ready yet

    # mark both prefilled -> they become a single decode group (same KV length)
    sched.update_from_output(prefilled={a, b})
    out = sched.schedule()
    assert not out.prefill_groups
    assert len(out.decode_groups) == 1
    assert out.decode_groups[0].req_ids == [a, b]


def test_mixed_prompt_lengths_never_mix_in_one_forward():
    model = FakeMiniMindOmni()
    runner, sched, (a, b) = _make_runner(model, [[1, 2, 3], [3, 4]])

    out = sched.schedule()
    assert len(out.prefill_groups) == 2  # different lengths -> separate prefills
    lengths = sorted(len(g.req_ids) for g in out.prefill_groups)
    assert lengths == [1, 1]


def test_max_batch_caps_concurrent_running():
    model = FakeMiniMindOmni()
    runner, sched, rids = _make_runner(model, [[1, 2, 3], [4, 5, 6]], max_batch=1)

    out = sched.schedule()
    assert len(out.prefill_groups[0].req_ids) == 1  # only one admitted


# -- batched forward really happens -----------------------------------------


def test_decode_group_runs_one_batched_forward():
    model = FakeMiniMindOmni()
    runner, sched, (a, b) = _make_runner(model, [[1, 2, 3], [4, 5, 6]])
    sched.update_from_output(prefilled=set(runner.sched._running)) if False else None

    out = sched.schedule()
    for g in out.prefill_groups:
        runner.prefill_group(g)
    prefill_calls = [c for c in model.calls if not c[2]]
    # one batched prefill of bs=2
    assert prefill_calls == [(2, 3, False)]

    sched.update_from_output(prefilled={a, b})
    out = sched.schedule()
    assert len(out.decode_groups) == 1
    runner.decode_group(out.decode_groups[0])
    decode_calls = [c for c in model.calls if c[2]]
    # one batched decode of bs=2, T=1
    assert decode_calls == [(2, 1, True)]


# -- Q10a: batch composition never changes per-request draws ------------------


def test_batch_layout_does_not_change_per_request_sampling():
    p_a, p_b = [1, 2, 3], [4, 5, 6]

    def run(prompts):
        model = FakeMiniMindOmni()
        runner, sched, _ = _make_runner(model, prompts, max_batch=4)
        _drain(runner, sched)
        return model, {
            rid: (list(s.text_tokens), [tuple(c) for c in s.audio_codes])
            for rid, s in runner.states.items()
        }

    # request A runs with B as neighbour, and alone -> identical draws
    model_batched, batched = run([p_a, p_b])
    model_solo, solo = run([p_a])

    assert batched["req-0"] == solo["req-0"]
    # and the batched-with-neighbour A used a real bs=2 decode group
    assert any(b == 2 for b, _, _ in model_batched.calls)


def test_fixed_slot_pool_layout_and_gather():
    pool = FixedKvSlotPool(max_seq=32)
    pool.register("x", n_layers=3, n_heads=2, head_dim=4, device="cpu", dtype=torch.float32)

    # write per-layer with seq at dim 0 of each row ([B, seq, kv, d])
    k = torch.randn(1, 5, 2, 4)
    v = torch.randn(1, 5, 2, 4)
    pool.write("x", layer=0, key=k, value=v, row=0)
    k2 = torch.randn(1, 5, 2, 4)
    pool.write("x", layer=1, key=k2, value=v, row=0)

    assert pool.length("x") == 5
    pairs = pool.gather(["x"])
    assert len(pairs) == 3  # n_layers
    assert pairs[0][0].shape == (1, 5, 2, 4)  # [B, seq, kv, d]

    # unequal-length gather must refuse (the model's rectangular constraint)
    pool2 = FixedKvSlotPool(max_seq=32)
    pool2.register("a", n_layers=1, n_heads=2, head_dim=4, device="cpu", dtype=torch.float32)
    pool2.register("b", n_layers=1, n_heads=2, head_dim=4, device="cpu", dtype=torch.float32)
    pool2.write("a", layer=0, key=torch.randn(1, 3, 2, 4), value=torch.randn(1, 3, 2, 4), row=0)
    pool2.write("b", layer=0, key=torch.randn(1, 4, 2, 4), value=torch.randn(1, 4, 2, 4), row=0)
    try:
        pool2.gather(["a", "b"])
        raise AssertionError("unequal-length gather should raise")
    except ValueError:
        pass


# --------------------------------------------------------------------------
# engine-level: run_batched_generate end-to-end with a fake bundle
# (tokenizer + mimi + model, no GPU) -- exercises the whole loop included
# tokenization and the serial codec chain.


class _FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return " ".join(m.get("content", "") for m in messages)

    def __call__(self, text):
        return SimpleNamespace(data={"input_ids": [ord(c) % 50 + 1 for c in text]})


class _FakeMimi:
    def decode(self, codes):
        return SimpleNamespace(audio_values=codes.float())


def test_run_batched_generate_engine_loop():
    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.outputs import AudioPayload

    model = FakeMiniMindOmni()
    bundle = SimpleNamespace(
        model=model, tokenizer=_FakeTokenizer(), mimi=_FakeMimi(), device="cpu"
    )
    out = run_batched_generate(
        bundle,
        ["hello world", "another prompt here"],
        max_batch=2,
        max_new_tokens=12,
        base_seed=42,
    )
    assert len(out) == 2
    for payload in out:
        assert isinstance(payload, AudioPayload)
        assert len(payload.data) > 44  # non-empty WAV (RIFF header + payload)
        assert payload.data[:4] == b"RIFF"  # real wave bytes


# -- deploy wiring (C): max_batch knob lives in deploy/*.yaml -------------


def test_deploy_config_parses_max_batch(tmp_path):
    from nanovllm_omni.config.registry import DeployConfig, load_deploy_config

    p = tmp_path / "deploy.yaml"
    p.write_text(
        "max_batch: 3\nstages:\n  - name: thinker\n    default_sampling_params: {max_tokens: 16}\n"
    )
    deploy = load_deploy_config(p)
    assert deploy.max_batch == 3
    assert DeployConfig().max_batch == 2  # default without yaml


def test_run_batched_generate_reads_max_batch_from_deploy():
    from nanovllm_omni.config.registry import DeployConfig
    from nanovllm_omni.engine.batched_runner import run_batched_generate
    from nanovllm_omni.outputs import AudioPayload

    model = FakeMiniMindOmni()
    bundle = SimpleNamespace(
        model=model, tokenizer=_FakeTokenizer(), mimi=_FakeMimi(), device="cpu"
    )
    deploy = DeployConfig(max_batch=2)  # knob from deploy yaml
    out = run_batched_generate(
        bundle,
        ["hello world", "another prompt here"],
        max_new_tokens=12,
        base_seed=42,
        deploy=deploy,
    )
    assert len(out) == 2
    assert all(isinstance(p, AudioPayload) for p in out)
