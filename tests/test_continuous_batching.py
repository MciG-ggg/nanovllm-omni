"""Continuous batching: a request added between steps joins the live batch.

The stage step is incremental, so the fork ``Scheduler`` is driven one
``schedule()`` round at a time. A request added after the decode started
waits in the scheduler and is prefilled on the next step, ahead of the
running decode; the step after that decodes both together. The token
streams must match a serial run (ADR 0011 P1: same engine, same input,
same seed, same per-request RNG).

The fake runner here does not touch the real paged KV cache; it records
the batch shape every step so the join is observable, and produces peaked
logits so the sampler's choice does not depend on batch composition.
"""

from __future__ import annotations

import importlib.util
import types

import pytest

from nanovllm_omni.config.params import OmniEngineArgs
from nanovllm_omni.config.registry import (
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
)
from nanovllm_omni.engine.runner import PipelineRunner
from nanovllm_omni.models.minimind_omni import stage_runner as stage_runner_mod
from nanovllm_omni.models.minimind_omni.stage_runner import (
    ThinkerDecode,
    ThinkerDecodeSession,
)

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)
torch = importlib.import_module("torch")


class _TorchRunner:
    """Fake StageRunner: records batch shapes and returns peaked logits."""

    def __init__(self, vocab: int = 16) -> None:
        self.config = types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            eos=15,
            kvcache_block_size=256,
            num_kvcache_blocks=32,
        )
        self.vocab = vocab
        self.prefill_sizes: list[int] = []
        self.decode_sizes: list[int] = []
        self._seqs: list = []
        self._is_prefill = True

    def set_context(self, is_prefill: bool, **kwargs: object) -> None:
        pass

    def reset_context(self) -> None:
        pass

    def prepare_prefill(self, seqs: list) -> tuple:
        self._seqs = list(seqs)
        self._is_prefill = True
        self.prefill_sizes.append(len(seqs))
        return torch.zeros(1, dtype=torch.long), None

    def prepare_decode(self, seqs: list) -> tuple:
        self._seqs = list(seqs)
        self._is_prefill = False
        self.decode_sizes.append(len(seqs))
        return torch.zeros(len(seqs), dtype=torch.long), None

    def prepare_sample(self, seqs: list) -> object:
        return torch.full((len(seqs),), 1e-5)

    def forward(self, input_ids: object, positions: object, is_prefill: bool) -> object:
        logits = torch.zeros((len(self._seqs), self.vocab))
        for row, sequence in enumerate(self._seqs):
            logits[row, (sequence.token_ids[-1] + 1) % self.vocab] = 50.0
        return logits

    def sample(self, logits: object, temperatures: object, sampler: object = None) -> object:
        if sampler is None:
            raise AssertionError("the torch thinker must pass its vendor sampler")
        return sampler(logits, temperatures)

    def bridge_hidden(self) -> object:
        if not self._is_prefill:
            return torch.zeros((len(self._seqs), 2))
        rows = sum(sequence.num_scheduled_tokens for sequence in self._seqs)
        return torch.zeros((rows, 2))


def _session(monkeypatch: pytest.MonkeyPatch, runner: _TorchRunner) -> ThinkerDecodeSession:
    monkeypatch.setattr(
        stage_runner_mod,
        "recapture_bridge_minimind",
        lambda stage_runner, full_ids, bridge: torch.zeros((len(full_ids), 2)),
    )
    return ThinkerDecodeSession(runner, backend="torch")


def _decode(monkeypatch: pytest.MonkeyPatch, requests: list[ThinkerDecode]) -> list[list[int]]:
    runner = _TorchRunner()
    session = _session(monkeypatch, runner)
    request_ids = [str(index) for index in range(len(requests))]
    for request, request_id in zip(requests, request_ids, strict=True):
        session.add(request, request_id)
    results: dict[str, list[int]] = {}
    while session.has_work():
        for request_id, _, decoded in session.step():
            results[request_id] = decoded.generated
    return [results[request_id] for request_id in request_ids]


def test_a_mid_stream_request_decodes_with_the_running_one(monkeypatch):
    request_a = ThinkerDecode([1, 2, 3], max_tokens=6, temperature=1e-5, seed=1)
    request_b = ThinkerDecode([7, 8], max_tokens=4, temperature=1e-5, seed=2)
    serial_a = _decode(monkeypatch, [request_a])[0]
    serial_b = _decode(monkeypatch, [request_b])[0]

    runner = _TorchRunner()
    session = _session(monkeypatch, runner)
    session.add(request_a, "a")

    session.step()
    session.add(request_b, "b")

    results: dict[str, list[int]] = {}
    while session.has_work():
        for request_id, _, decoded in session.step():
            results[request_id] = decoded.generated

    assert runner.prefill_sizes == [1, 1]
    assert 2 in runner.decode_sizes
    assert results["a"] == serial_a
    assert results["b"] == serial_b


def test_a_finished_request_leaves_before_the_slow_one(monkeypatch):
    fast = ThinkerDecode([1, 2, 3], max_tokens=2, temperature=1e-5, seed=1)
    slow = ThinkerDecode([7, 8], max_tokens=5, temperature=1e-5, seed=2)

    runner = _TorchRunner()
    session = _session(monkeypatch, runner)
    session.add(fast, "fast")
    session.add(slow, "slow")

    session.step()
    finished = session.step()

    assert [request_id for request_id, _, _ in finished] == ["fast"]
    assert session.has_work()

    while session.has_work():
        session.step()

    assert runner.decode_sizes == [2, 1, 1, 1]


def test_runner_serves_a_stage_whose_queue_is_empty_but_batch_is_running():
    pipeline = PipelineConfig(
        name="incremental",
        stages=(
            StageConfig(
                0,
                "s",
                StageExecutionType.LLM_AR,
                ("tests._stage_factories", "incremental_simple"),
                is_terminal=True,
            ),
        ),
        default_deploy_config_name="incremental.yaml",
    )
    runner = PipelineRunner(
        pipeline=pipeline,
        deploy=DeployConfig(),
        args=OmniEngineArgs(model="m", device="cpu"),
    )

    id_a = runner.add_request("a")
    runner.step()
    id_b = runner.add_request("b")

    finished = []
    while runner.has_unfinished():
        finished.extend(runner.step())

    stage = runner._ensure_stages()[0]
    assert stage.batch_sizes == [1, 2, 2, 1]
    assert [entry.request_id for entry in finished] == [id_a, id_b]
