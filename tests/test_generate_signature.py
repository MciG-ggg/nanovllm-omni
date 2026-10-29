"""Test: Omni/AsyncOmni generate() signature parity with vllm-omni.

``generate`` keeps its locked public shape (``sampling_params`` single value,
returns ``list``) and gains a backward-compatible ``py_generator`` keyword
(vllm-omni parity) and per-request ``sampling_params_list`` input.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.config.params import SamplingParams  # noqa: E402
from nanovllm_omni.outputs import OmniRequestOutput  # noqa: E402


class _RunnerStub:
    """Stand-in for PipelineRunner: records the calls generate() makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    def run(self, prompt: str, sampling: SamplingParams | None) -> tuple:
        self.calls.append((prompt, sampling))
        return SimpleNamespace(audio=b"RIFF....", sample_rate=24000), {"s": 1.0}


def _make_omni(runner: _RunnerStub) -> object:
    from nanovllm_omni.entrypoints.omni import Omni

    omni = Omni("minimind_o")
    omni._executor = SimpleNamespace(run=runner.run)  # type: ignore[attr-defined]
    return omni


class _AsyncSubmitStub:
    """Executor with both ``run_in_executor``-style submit and sync run."""

    def __init__(self, runner: _RunnerStub) -> None:
        self._runner = runner
        self.submit_calls: list[tuple[str, object | None]] = []

    async def submit(self, prompt: str, sampling: SamplingParams | None) -> tuple:
        self.submit_calls.append((prompt, sampling))
        return self._runner.run(prompt, sampling)


def _make_async_omni(runner: _RunnerStub) -> tuple[object, _AsyncSubmitStub]:
    from nanovllm_omni.entrypoints.async_omni import AsyncOmni

    executor = _AsyncSubmitStub(runner)
    omni = AsyncOmni("minimind_o")
    omni._executor = executor  # type: ignore[attr-defined]
    return omni, executor


def _make_omni_output(out: SimpleNamespace) -> OmniRequestOutput:
    return OmniRequestOutput.from_pipeline(out, final_output_type="audio")


def test_generate_default_returns_list() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    out = omni.generate("hello")
    assert isinstance(out, list)
    assert len(out) == 1
    assert runner.calls == [("hello", None)]


def test_generate_accepts_list_and_single_sampling() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    sp = SamplingParams(temperature=0.5)
    out = omni.generate(["a", "b"], sampling_params=sp)
    assert len(out) == 2
    assert [c[0] for c in runner.calls] == ["a", "b"]
    assert all(c[1] is sp for c in runner.calls)


def test_generate_py_generator_yields_in_order() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    gen = omni.generate(["x", "y"], py_generator=True)
    import types as _types

    assert isinstance(gen, _types.GeneratorType)
    items = list(gen)
    assert len(items) == 2
    assert all(isinstance(i, OmniRequestOutput) for i in items)


def test_generate_py_generator_lazy() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    gen = omni.generate(["x", "y"], py_generator=True)
    assert runner.calls == []  # nothing executed until consumed
    next(gen)  # first prompt only
    assert [c[0] for c in runner.calls] == ["x"]


def test_generate_sampling_params_list_per_request() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    sps = [SamplingParams(temperature=0.5), SamplingParams(temperature=0.9)]
    out = omni.generate(["a", "b"], sampling_params_list=sps)
    assert len(out) == 2
    assert runner.calls[0][1] is sps[0]
    assert runner.calls[1][1] is sps[1]


def test_generate_positional_list_is_per_request() -> None:
    """The second positional argument is ``sampling_params_list``.

    Upstream ``vllm-omni`` takes one SamplingParams per prompt in that
    slot, so a list passed positionally must not be applied to every
    prompt.
    """
    runner = _RunnerStub()
    omni = _make_omni(runner)
    sps = [SamplingParams(temperature=0.5), SamplingParams(temperature=0.9)]
    omni.generate(["a", "b"], sps)
    assert runner.calls[0][1] is sps[0]
    assert runner.calls[1][1] is sps[1]


def test_generate_sampling_params_list_accepts_one_object() -> None:
    """``sampling_params_list`` also takes a bare object (upstream shape)."""
    runner = _RunnerStub()
    omni = _make_omni(runner)
    sp = SamplingParams(temperature=0.5)
    omni.generate(["a", "b"], sampling_params_list=sp)
    assert runner.calls[0][1] is sp
    assert runner.calls[1][1] is sp


def test_generate_rejects_both_param_arguments() -> None:
    runner = _RunnerStub()
    omni = _make_omni(runner)
    with pytest.raises(ValueError):
        omni.generate(
            ["a"],
            sampling_params=SamplingParams(),
            sampling_params_list=[SamplingParams()],
        )


def test_async_generate_unchanged_shape() -> None:
    """AsyncOmni still yields an async iterator."""
    omni, _ = _make_async_omni(_RunnerStub())

    async def collect() -> object:
        items = [o async for o in omni.generate(["a", "b"])]  # type: ignore[union-attr]
        return [type(o) for o in items]  # type: ignore[return-value]

    import asyncio

    result = asyncio.run(collect())
    assert result == [OmniRequestOutput, OmniRequestOutput]


def test_async_omni_is_a_sibling_not_a_subclass() -> None:
    """The reference keeps Omni and AsyncOmni as siblings.

    Their ``generate`` signatures differ, so inheriting one from the other
    would claim a substitutability that does not hold.
    """
    from nanovllm_omni.entrypoints.async_omni import AsyncOmni
    from nanovllm_omni.entrypoints.base import OmniBase
    from nanovllm_omni.entrypoints.omni import Omni

    assert issubclass(AsyncOmni, OmniBase)
    assert not issubclass(AsyncOmni, Omni)


def test_async_generate_dict_prompt_is_one_request() -> None:
    """A dict prompt must not be iterated into its keys."""
    import asyncio

    omni, executor = _make_async_omni(_RunnerStub())

    async def collect() -> None:
        async for _ in omni.generate({"prompt": "hi", "image": b"x"}):  # type: ignore[arg-type,union-attr]
            pass

    asyncio.run(collect())
    assert len(executor.submit_calls) == 1
    text, sampling = executor.submit_calls[0]
    assert text == "hi"
    assert sampling is not None and sampling.extra["image"] == b"x"


def test_async_output_carries_stage_ms() -> None:
    """Both entrypoints expose the same per-stage timings."""
    import asyncio

    omni, _ = _make_async_omni(_RunnerStub())

    async def collect() -> list[OmniRequestOutput]:
        return [o async for o in omni.generate(["a"])]  # type: ignore[union-attr]

    outputs = asyncio.run(collect())
    assert outputs[0].custom_output["stage_ms"] == {"s": 1.0}


def test_generate_accepts_dict_prompt_with_image() -> None:
    """TK-017: a dict prompt's ``image`` rides into sampling.extra['image']."""
    runner = _RunnerStub()
    omni = _make_omni(runner)
    img = b"fake-png-bytes"
    omni.generate([{"prompt": "pick up the block", "image": img}])  # type: ignore[arg-type]
    text, sampling = runner.calls[0]
    assert text == "pick up the block"
    assert sampling is not None and sampling.extra["image"] == img


def test_generate_dict_prompt_without_modal_is_text_only() -> None:
    """TK-017: a dict without modal fields behaves like a plain str."""
    runner = _RunnerStub()
    omni = _make_omni(runner)
    omni.generate([{"prompt": "hello"}])  # type: ignore[arg-type]
    text, sampling = runner.calls[0]
    assert text == "hello"
    assert sampling is None


def test_split_prompt_dict_preserves_prompt_and_drops_none() -> None:
    from nanovllm_omni.entrypoints.omni import _split_prompt

    assert _split_prompt("hi") == ("hi", None)
    assert _split_prompt({"prompt": "do it", "image": b"x"}) == ("do it", {"image": b"x"})
    assert _split_prompt({"prompt": "do it", "image": None}) == ("do it", None)
