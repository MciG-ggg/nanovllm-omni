"""CPU contracts for the opt-in Talker MTP CUDA Graph seam."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.engine.talker_cuda_graph import (  # noqa: E402
    CudaGraphTalkerDecoder,
    TalkerMtpCudaGraph,
    enable_talker_mtp_cuda_graph,
)


class _FakeTalker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
        temperature: float = 0.2,
        top_k: int = 50,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "input_ids": input_ids,
                "input_embeds": input_embeds,
                "last_talker_hidden": last_talker_hidden,
                "text_step": text_step,
                "active_mask": active_mask,
                "temperature": temperature,
                "top_k": top_k,
                "do_sample": do_sample,
                "generator": generator,
            }
        )
        return torch.where(
            active_mask.bool(),
            input_ids.reshape(-1, 1).expand(-1, active_mask.shape[-1]),
            torch.full_like(active_mask, 99, dtype=torch.long),
        )


def _inputs(batch_size: int = 1, hidden: int = 4) -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(batch_size, dtype=torch.long),
        torch.zeros(batch_size, hidden),
        torch.ones(batch_size, hidden),
        torch.zeros(batch_size, hidden),
        torch.ones(batch_size, 8, dtype=torch.bool),
    )


def test_helper_returns_none_without_cuda() -> None:
    assert not torch.cuda.is_available()
    assert enable_talker_mtp_cuda_graph(_FakeTalker()) is None


def test_unsupported_talker_returns_none_and_direct_constructor_is_clear() -> None:
    assert enable_talker_mtp_cuda_graph(object()) is None
    with pytest.raises(TypeError, match="talker_mtp"):
        TalkerMtpCudaGraph(object())


def test_cache_key_separates_batch_and_fixed_shapes() -> None:
    decoder = TalkerMtpCudaGraph(_FakeTalker(), batch_sizes=(1, 2))
    one = decoder._cache_key(*_inputs(1))
    two = decoder._cache_key(*_inputs(2))
    wider = decoder._cache_key(*_inputs(1, hidden=8))
    assert one != two
    assert one != wider
    assert one.batch_size == 1
    assert two.batch_size == 2


def test_invalidation_drops_only_requested_entry() -> None:
    decoder = TalkerMtpCudaGraph(_FakeTalker())
    key = decoder._cache_key(*_inputs())
    other = decoder._cache_key(*_inputs(hidden=8))
    decoder._cache[key] = object()  # type: ignore[assignment]
    decoder._cache[other] = object()  # type: ignore[assignment]
    decoder.invalidate(key)
    assert key not in decoder._cache
    assert other in decoder._cache
    decoder.invalidate()
    assert decoder.cache_size == 0


def test_cpu_decode_delegates_exact_active_mask_and_generator() -> None:
    talker = _FakeTalker()
    decoder = CudaGraphTalkerDecoder(talker)
    input_ids, embeds, hidden, text_step, active_mask = _inputs()
    generator = torch.Generator().manual_seed(7)
    output = decoder.decode(
        input_ids,
        embeds,
        hidden,
        text_step,
        active_mask=active_mask,
        do_sample=True,
        generator=generator,
    )
    assert torch.equal(output, input_ids.reshape(1, 1).expand(1, 8))
    assert talker.calls[-1]["active_mask"] is active_mask
    assert talker.calls[-1]["generator"] is generator
    assert talker.calls[-1]["do_sample"] is True
    assert decoder.cache_size == 0


def test_capture_failure_invalidates_entry_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    talker = _FakeTalker()
    decoder = TalkerMtpCudaGraph(talker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(decoder, "_graph_safe", lambda key, tensors: True)

    def fail_capture(key, inputs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(decoder, "_capture", fail_capture)
    input_ids, embeds, hidden, text_step, active_mask = _inputs()
    output = decoder.decode(
        input_ids,
        embeds,
        hidden,
        text_step,
        active_mask=active_mask,
        do_sample=False,
    )
    assert output.shape == (1, 8)
    assert decoder.cache_size == 0
    assert talker.calls[-1]["do_sample"] is False


def test_no_seed_is_created_and_generator_reaches_eager_boundary() -> None:
    source = inspect.getsource(TalkerMtpCudaGraph)
    assert "manual_seed" not in source
    talker = _FakeTalker()
    decoder = TalkerMtpCudaGraph(talker)
    generator = torch.Generator().manual_seed(11)
    input_ids, embeds, hidden, text_step, active_mask = _inputs()
    decoder.decode(
        input_ids,
        embeds,
        hidden,
        text_step,
        active_mask=active_mask,
        do_sample=True,
        generator=generator,
    )
    assert talker.calls[-1]["generator"] is generator


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_capture_requires_real_cuda_runtime() -> None:
    """The real capture path is intentionally not claimed by CPU CI."""
    talker = _FakeTalker()
    decoder = TalkerMtpCudaGraph(talker)
    input_ids, embeds, hidden, text_step, active_mask = (value.cuda() for value in _inputs())
    output = decoder.decode(
        input_ids,
        embeds,
        hidden,
        text_step,
        active_mask=active_mask,
        do_sample=False,
    )
    assert output.is_cuda
    assert decoder.cache_size == 1


if __name__ == "__main__":
    pytest.main([__file__])
