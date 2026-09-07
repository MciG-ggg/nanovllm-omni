"""Contract tests for the bundle kwargs plumbed from ``OmniEngineArgs``.

Pins the OmniEngineArgs effective-field matrix: ``trust_remote_code`` and
``dtype`` flow from ``OmniEngineArgs`` through the per-stage factory into
``load_minimind_omni_bundle`` and the resulting ``from_pretrained`` calls /
dtype cast.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# dtype cast helper
# ---------------------------------------------------------------------------


class _FakeCastModel:
    """Stand-in for a torch module that records which dtype method ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def half(self):
        self.calls.append("half")
        return self

    def bfloat16(self):
        self.calls.append("bfloat16")
        return self

    def float(self):
        self.calls.append("float")
        return self


def test_cast_model_dtype_default_is_half_on_gpu():
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    m = _FakeCastModel()
    assert _cast_model_dtype(m, None, "cuda") is m
    assert m.calls == ["half"]


def test_cast_model_dtype_explicit_float16_matches_default():
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    m = _FakeCastModel()
    _cast_model_dtype(m, "float16", "cuda")
    assert m.calls == ["half"]


def test_cast_model_dtype_bfloat16():
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    m = _FakeCastModel()
    _cast_model_dtype(m, "bfloat16", "cuda")
    assert m.calls == ["bfloat16"]


def test_cast_model_dtype_float32():
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    m = _FakeCastModel()
    _cast_model_dtype(m, "float32", "cuda")
    assert m.calls == ["float"]


def test_cast_model_dtype_cpu_is_noop_for_any_value():
    """cpu always skips the dtype cast regardless of dtype — keeps parity
    with the legacy ``if device != "cpu"`` gate that the helper replaces."""
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    class Strict:
        def half(self):
            raise AssertionError("called on cpu")

        def bfloat16(self):
            raise AssertionError("called on cpu")

        def float(self):
            raise AssertionError("called on cpu")

    for dtype in (None, "float16", "bfloat16", "float32"):
        m = Strict()
        assert _cast_model_dtype(m, dtype, "cpu") is m


def test_cast_model_dtype_rejects_unknown_value():
    """SmolVLA's int8/qint8 path does NOT flow through this helper;
    any other dtype string raises so we fail loud at load time instead
    of silently applying the wrong precision."""
    from nanovllm_omni.models.minimind_omni.bundle import _cast_model_dtype

    class Ok:
        def half(self):
            return self

        def bfloat16(self):
            return self

        def float(self):
            return self

    with pytest.raises(ValueError, match="unsupported dtype"):
        _cast_model_dtype(Ok(), "int8", "cuda")


# ---------------------------------------------------------------------------
# thinker's bundle kwarg plumbing
# ---------------------------------------------------------------------------


def test_thinker_stage_passes_trust_remote_code_and_dtype_to_bundle(monkeypatch):
    """``OmniEngineArgs.trust_remote_code`` and ``.dtype`` flow from
    ``_thinker_stage`` into ``create_bundle`` as kwargs."""
    from nanovllm_omni.config.params import OmniEngineArgs
    from nanovllm_omni.models.minimind_omni import thinker as thinker_mod

    captured: dict[str, object] = {}

    def fake_create_bundle(model_id, device, **kwargs):
        captured["model_id"] = model_id
        captured["device"] = device
        captured.update(kwargs)
        return None  # body of the factory doesn't use the bundle

    monkeypatch.setattr(thinker_mod, "create_bundle", fake_create_bundle)

    args = OmniEngineArgs(
        model="any",
        device="cpu",
        trust_remote_code=False,
        dtype="bfloat16",
    )
    thinker_mod._thinker_stage(deploy=None, args=args)

    assert captured["model_id"] == "any"
    assert captured["device"] == "cpu"
    assert captured["trust_remote_code"] is False
    assert captured["dtype"] == "bfloat16"


def test_thinker_stage_defaults_keep_legacy_bundle_kwargs(monkeypatch):
    """Without explicit ``trust_remote_code`` / ``dtype``, the thinker
    forwards the field defaults (True / None) so the bundle's behavior
    is bit-for-bit equivalent to the previous hard-coded path."""
    from nanovllm_omni.config.params import OmniEngineArgs
    from nanovllm_omni.models.minimind_omni import thinker as thinker_mod

    captured: dict[str, object] = {}

    def fake_create_bundle(model_id, device, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(thinker_mod, "create_bundle", fake_create_bundle)

    args = OmniEngineArgs(model="any", device="cpu")
    thinker_mod._thinker_stage(deploy=None, args=args)

    assert captured["trust_remote_code"] is True
    assert captured["dtype"] is None
    assert "mimi_model_id" not in captured


# ---------------------------------------------------------------------------
# bundle-level trust_remote_code plumbing (skipped when transformers is
# unavailable — the cast + thinker tests above already pin the dtype path).
# ---------------------------------------------------------------------------


def test_bundle_from_pretrained_receives_trust_remote_code_false(monkeypatch):
    """``OmniEngineArgs.trust_remote_code=False`` flows through
    ``load_minimind_omni_bundle`` into the tokenizer and model
    ``from_pretrained`` calls."""
    transformers = pytest.importorskip("transformers")
    try:
        # The loader does `from transformers import ... MimiModel` at load
        # time; some transformers builds ship MimiModel only behind a scoped
        # audio dep. Probe it so the trust_remote_code plumbing test skips
        # cleanly instead of dying on the missing codec class.
        from transformers import MimiModel  # noqa: F401
    except ImportError:
        pytest.skip("this transformers build has no MimiModel (audio dep missing)")
    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    captured: list[dict[str, object]] = []

    class _FakeModel:
        def eval(self):
            return self

        def half(self):
            return self

        def bfloat16(self):
            return self

        def float(self):
            return self

        def to(self, _device):
            return self

    def _fake_from_pretrained(path, **kwargs):
        captured.append(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        _fake_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        _fake_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.MimiModel,
        "from_pretrained",
        lambda *a, **kw: _FakeModel(),
    )
    monkeypatch.setattr(bundle_mod, "_resolve_snapshot", lambda x: x)
    # No-op the dtype cast so the fake model doesn't need real torch tensors.
    monkeypatch.setattr(bundle_mod, "_cast_model_dtype", lambda m, _d, _dev: m)
    # No-op the optimization imports that bundle.py does inside the function.
    monkeypatch.setattr("nanovllm_omni.optim.attention.enable_sdpa_decode", lambda m: None)
    monkeypatch.setattr(
        "nanovllm_omni.optim.attention.enable_fused_projections",
        lambda m: None,
    )
    monkeypatch.setattr("nanovllm_omni.optim.attention.enable_fused_rmsnorm", lambda m: None)
    monkeypatch.setattr("nanovllm_omni.optim.attention.enable_fused_rope", lambda m: None)

    bundle_mod.load_minimind_omni_bundle(
        model_id="any",
        device="cpu",
        trust_remote_code=False,
    )

    # Tokenizer + causal LM from_pretrained both honored trust_remote_code=False.
    # MimiModel's call doesn't carry trust_remote_code in the legacy path either.
    relevant = [c for c in captured if "trust_remote_code" in c]
    assert relevant, "expected trust_remote_code to be passed to from_pretrained"
    assert all(c["trust_remote_code"] is False for c in relevant)


__all__ = [
    "test_cast_model_dtype_default_is_half_on_gpu",
    "test_cast_model_dtype_explicit_float16_matches_default",
    "test_cast_model_dtype_bfloat16",
    "test_cast_model_dtype_float32",
    "test_cast_model_dtype_cpu_is_noop_for_any_value",
    "test_cast_model_dtype_rejects_unknown_value",
    "test_thinker_stage_passes_trust_remote_code_and_dtype_to_bundle",
    "test_thinker_stage_defaults_keep_legacy_bundle_kwargs",
    "test_bundle_from_pretrained_receives_trust_remote_code_false",
    "test_thinker_stage_passes_enforce_eager_to_bundle",
    "test_load_minimind_omni_bundle_skips_fusion_when_enforce_eager_true",
    "test_load_minimind_omni_bundle_runs_fusion_by_default",
    "test_bench_cli_accepts_enforce_eager_flag",
]


# ---------------------------------------------------------------------------
# enforce_eager plumbing
# ---------------------------------------------------------------------------


def test_thinker_stage_passes_enforce_eager_to_bundle(monkeypatch):
    """``OmniEngineArgs.enforce_eager`` flows through the thinker stage
    factory into ``create_bundle`` so the bench harness can measure a
    true apples-to-apples baseline (no fusion monkey-patches applied)."""
    from nanovllm_omni.config.params import OmniEngineArgs
    from nanovllm_omni.models.minimind_omni import thinker as thinker_mod

    captured: dict[str, object] = {}

    def fake_create_bundle(model_id, device, **kwargs):
        captured["enforce_eager"] = kwargs.get("enforce_eager")
        return None

    monkeypatch.setattr(thinker_mod, "create_bundle", fake_create_bundle)

    args = OmniEngineArgs(model="any", device="cpu", enforce_eager=True)
    thinker_mod._thinker_stage(deploy=None, args=args)

    assert captured["enforce_eager"] is True


def test_load_minimind_omni_bundle_skips_fusion_when_enforce_eager_true(monkeypatch):
    """``enforce_eager=True`` keeps the four attention-fusion monkey-patches
    from running. The bench harness relies on this to measure the unfused
    baseline; the patches would otherwise mask the launch overhead the
    fusion stack is supposed to remove."""
    pytest.importorskip("transformers")
    from transformers import MimiModel  # noqa: F401  -- probe audio dep

    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    fusion_calls: list[str] = []

    class _FakeModel:
        def eval(self):
            return self

        def half(self):
            return self

        def bfloat16(self):
            return self

        def float(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path, **kw: object(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda path, **kw: _FakeModel(),
    )
    monkeypatch.setattr(
        "transformers.MimiModel.from_pretrained",
        lambda *a, **kw: _FakeModel(),
    )
    monkeypatch.setattr(bundle_mod, "_resolve_snapshot", lambda x: x)
    monkeypatch.setattr(bundle_mod, "_cast_model_dtype", lambda m, _d, _dev: m)

    for name in (
        "enable_sdpa_decode",
        "enable_fused_rmsnorm",
        "enable_fused_projections",
        "enable_fused_rope",
    ):
        monkeypatch.setattr(
            f"nanovllm_omni.optim.attention.{name}",
            lambda _m, _n=name: fusion_calls.append(_n),
        )

    bundle_mod.load_minimind_omni_bundle(
        model_id="any",
        device="cpu",
        enforce_eager=True,
    )

    assert (
        fusion_calls == []
    ), f"enforce_eager=True must skip fusion patches, but {fusion_calls} ran"


def test_load_minimind_omni_bundle_runs_fusion_by_default(monkeypatch):
    """The legacy default (enforce_eager absent or False) keeps applying all
    four fusion patches -- this is the existing production behavior and
    reverting it silently would regress every downstream user."""
    pytest.importorskip("transformers")
    from transformers import MimiModel  # noqa: F401

    from nanovllm_omni.models.minimind_omni import bundle as bundle_mod

    fusion_calls: list[str] = []

    class _FakeModel:
        def eval(self):
            return self

        def half(self):
            return self

        def bfloat16(self):
            return self

        def float(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path, **kw: object(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda path, **kw: _FakeModel(),
    )
    monkeypatch.setattr(
        "transformers.MimiModel.from_pretrained",
        lambda *a, **kw: _FakeModel(),
    )
    monkeypatch.setattr(bundle_mod, "_resolve_snapshot", lambda x: x)
    monkeypatch.setattr(bundle_mod, "_cast_model_dtype", lambda m, _d, _dev: m)

    for name in (
        "enable_sdpa_decode",
        "enable_fused_rmsnorm",
        "enable_fused_projections",
        "enable_fused_rope",
    ):
        monkeypatch.setattr(
            f"nanovllm_omni.optim.attention.{name}",
            lambda _m, _n=name: fusion_calls.append(_n),
        )

    bundle_mod.load_minimind_omni_bundle(model_id="any", device="cpu")

    assert sorted(fusion_calls) == sorted(
        [
            "enable_sdpa_decode",
            "enable_fused_rmsnorm",
            "enable_fused_projections",
            "enable_fused_rope",
        ]
    )


def test_bench_cli_accepts_enforce_eager_flag():
    """The bench CLI must surface --enforce-eager on every subcommand
    (time / matrix / trace-torch / profile-detail / trace-nsys) so the
    apples-to-apples measurement matrix can sweep the flag uniformly."""
    from nanovllm_omni.optim.bench.__main__ import build_parser

    parser = build_parser()
    # Subcommands that take --out get a dummy path; the test only checks
    # --enforce-eager is accepted on each subcommand, not that the run succeeds.
    out_required = {"trace-torch", "profile-detail", "trace-nsys", "_nsys-inner"}
    for cmd in ("time", "matrix", "trace-torch", "profile-detail", "trace-nsys"):
        argv = [cmd, "--enforce-eager", "--pipeline", "full"]
        if cmd in out_required:
            argv.extend(["--out", "/tmp/bench-flag-test"])
        ns = parser.parse_args(argv)
        assert ns.enforce_eager is True, f"--enforce-eager not parsed for {cmd}"
        assert ns.pipeline == "full"


def test_bench_cli_can_force_each_graph_off() -> None:
    """The full E2E sweep must override YAML defaults in either direction."""
    from nanovllm_omni.optim.bench.__main__ import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "time",
            "--pipeline",
            "full",
            "--no-use-thinker-cuda-graph",
            "--use-talker-cuda-graph",
        ]
    )
    assert ns.use_thinker_cuda_graph is False
    assert ns.use_talker_cuda_graph is True


def test_full_bench_rejects_non_equivalent_thinker_graph() -> None:
    """The full path must fail loudly until its post-EOS bridge parity exists."""
    from nanovllm_omni.optim.bench.__main__ import build_parser, cmd_time

    args = build_parser().parse_args(["time", "--pipeline", "full", "--use-thinker-cuda-graph"])
    with pytest.raises(SystemExit, match="post-EOS bridge-state contract"):
        cmd_time(args)
