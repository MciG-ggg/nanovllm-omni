from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_omni.models.minimind_omni.batched_generation import (  # noqa: E402
    BatchedThinkerRunner,
)
from nanovllm_omni.models.minimind_omni.runtime_scheduler import RuntimeScheduler  # noqa: E402
from nanovllm_omni.models.minimind_omni.talker import wrap_talker  # noqa: E402
from tests._talker_fixtures import make_fake_bundle, span_bridge  # noqa: E402
from tests.test_batched_generation import FakeMiniMindOmni  # noqa: E402


class _EosFirstModel(FakeMiniMindOmni):
    """Fake thinker that emits visible EOS from the prefill prediction."""

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = super().forward(*args, **kwargs)
        if len(self.calls) == 1:
            out.logits[..., 2] = 100.0
        else:
            out.logits[..., 0] = 100.0
        return out


def _run_runner(runner: BatchedThinkerRunner, scheduler: RuntimeScheduler) -> None:
    while scheduler.has_work():
        scheduled = scheduler.schedule()
        if scheduled.is_empty:
            break
        prefilled: set[str] = set()
        finished: set[str] = set()
        for group in scheduled.prefill_groups:
            runner.prefill_group(group)
            request_ids = {chunk.sequence.request_id for chunk in group.items}
            prefilled.update(request_ids)
            finished.update(rid for rid in request_ids if runner.step_finished(rid))
        for group in scheduled.decode_groups:
            runner.decode_group(group)
            finished.update(
                sequence.request_id
                for sequence in group.items
                if runner.step_finished(sequence.request_id)
            )
        scheduler.update_from_output(prefilled=prefilled, finished=finished)


def _make_eos_runner(*, max_new_tokens: int = 20, padding_count: int = 3):
    model = _EosFirstModel()
    scheduler = RuntimeScheduler(max_num_seqs=1)
    runner = BatchedThinkerRunner(
        SimpleNamespace(model=model),
        scheduler,
        max_new_tokens=max_new_tokens,
        eos_token_id=2,
        post_eos_padding_count=padding_count,
        internal_stop_token_id=13,
    )
    request_id = runner.add_request([1, 2, 3])
    return model, scheduler, runner, request_id


def test_opt_in_post_eos_sequence_executes_forced_steps() -> None:
    model, scheduler, runner, request_id = _make_eos_runner()
    _run_runner(runner, scheduler)

    state = runner.states[request_id]
    assert state.text_tokens[0] == 2
    assert state.text_tokens[1:] == [
        model.enter_token_id,
        model.pad_token_id,
        model.pad_token_id,
        model.pad_token_id,
        13,
    ]
    assert state.internal_stop_emitted is True
    # One prefill plus one forward for every token after EOS.
    assert len(model.calls) == len(state.text_tokens)


def test_post_eos_state_machine_respects_hard_token_budget() -> None:
    _model, scheduler, runner, request_id = _make_eos_runner(max_new_tokens=3, padding_count=20)
    _run_runner(runner, scheduler)

    state = runner.states[request_id]
    assert len(state.text_tokens) == 3
    assert state.text_tokens == [2, runner.sampling["enter"], runner.sampling["pad"]]
    assert state.internal_stop_emitted is False


def _watchdog_preprocess(talker, request_id: str, num_computed: int) -> None:
    bridge = span_bridge(hidden_size=8, sequence_len=3)
    talker.preprocess(
        torch.tensor([1], dtype=torch.long),
        None,
        hidden_states={"bridge": bridge},
        _omni_prompt_len=2,
        _omni_num_computed_tokens=num_computed,
        _omni_is_prefill=False,
        request_id=request_id,
    )


def test_talker_watchdog_counts_only_steps_after_final_bridge() -> None:
    talker = wrap_talker(make_fake_bundle(max_steps_after_last_thinker_token=2))
    _watchdog_preprocess(talker, "req", 2)  # final bridge row, not after it
    assert talker._steps_after_last_thinker_by_req == {}
    assert talker._stop_pending_by_req == {}

    _watchdog_preprocess(talker, "req", 3)
    assert talker._steps_after_last_thinker_by_req["req"] == 1
    assert talker._stop_pending_by_req == {}
    _watchdog_preprocess(talker, "req", 4)
    assert talker._steps_after_last_thinker_by_req["req"] == 2
    assert talker._stop_pending_by_req["req"] is True


def test_talker_watchdog_negative_value_is_disabled() -> None:
    talker = wrap_talker(make_fake_bundle(max_steps_after_last_thinker_token=-1))
    for num_computed in range(3, 8):
        _watchdog_preprocess(talker, "req", num_computed)
    assert talker._steps_after_last_thinker_by_req["req"] == 5
    assert talker._stop_pending_by_req == {}
