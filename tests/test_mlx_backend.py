"""MLX backend contracts that hold without mlx or a model on disk.

``mlx`` / ``mlx-lm`` are optional Mac-track dependencies and are not in the
project venv, so the runner is built against stub modules here. The real
decode loop runs in ``tools/bench_mlx_thinker.py``.
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from unittest import mock

import numpy as np
import pytest

from nanovllm_omni.backend import StageRunner as StageRunnerProtocol
from nanovllm_omni.backend import mlx_backend
from nanovllm_omni.backend.mlx_backend import MlxBackend, MlxConfig, MlxStageRunner
from nanovllm_omni.models.minimind_omni.stage_runner import ThinkerDecode, decode_minimind_mlx


class _FakeBatchKVCache:
    def __init__(self, left_padding: list[int]) -> None:
        self.left_padding = list(left_padding)
        self.offset: list[int] = []
        self._idx = 0

    @property
    def row_count(self) -> int:
        return len(self.left_padding)

    def record(self, input_ids: np.ndarray) -> None:
        if self._idx == 0:
            self._idx = input_ids.shape[1]
            self.offset = [self._idx - pad for pad in self.left_padding]
        else:
            self._idx += input_ids.shape[1]
            self.offset = [offset + input_ids.shape[1] for offset in self.offset]

    def extend(self, cache: _FakeBatchKVCache) -> None:
        max_idx = max(self._idx, cache._idx)
        self.left_padding.extend(max_idx - cache._idx + pad for pad in cache.left_padding)
        self.offset.extend(cache.offset)
        self._idx = max_idx

    def filter(self, batch_indices: list[int]) -> None:
        self.left_padding = [self.left_padding[index] for index in batch_indices]
        self.offset = [self.offset[index] for index in batch_indices]


class _FakeModel:
    def __init__(self) -> None:
        self.layers = [object(), object()]
        self.forward_rows: list[int] = []

    def __call__(self, input_ids: np.ndarray, cache: list[_FakeBatchKVCache]) -> np.ndarray:
        for layer_cache in cache:
            assert layer_cache.row_count == input_ids.shape[0]
            layer_cache.record(input_ids)
        self.forward_rows.append(input_ids.shape[0])
        return np.zeros((input_ids.shape[0], input_ids.shape[1], 4), dtype=np.float32)


def _stub_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the MLX modules used by the runner and decoder."""
    mx = types.ModuleType("mlx.core")
    mx.__spec__ = ModuleSpec("mlx.core", loader=None)
    mx.float16 = "float16"
    mx.float32 = np.float32
    mx.int32 = np.int32
    mx.gpu = "gpu"
    mx.array = np.array
    mx.arange = np.arange
    mx.random = types.SimpleNamespace(key=lambda seed: seed)
    mx.set_default_device = mock.MagicMock()
    mx.eval = mock.MagicMock()
    mlx = types.ModuleType("mlx")
    mlx.__spec__ = ModuleSpec("mlx", loader=None)
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    tokenizer = mock.MagicMock()
    tokenizer.eos_token_ids = {7, 3}
    utils = types.ModuleType("mlx_lm.utils")
    utils.load = mock.MagicMock(return_value=(mock.MagicMock(), tokenizer))
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    models = types.ModuleType("mlx_lm.models")
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.BatchKVCache = _FakeBatchKVCache
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)


def _fake_runner(monkeypatch: pytest.MonkeyPatch, config: MlxConfig) -> MlxStageRunner:
    _stub_mlx(monkeypatch)
    runner = MlxStageRunner.__new__(MlxStageRunner)
    runner._mx = sys.modules["mlx.core"]
    runner.config = config
    runner._model = _FakeModel()
    runner.tokenizer = types.SimpleNamespace(eos_token_ids={7, 3})
    runner._cache = None
    runner._rng = None
    runner._left_padding = [0]
    runner._bridge = None
    runner._bridge_layer = 0
    return runner


def test_mlx_runner_satisfies_the_stage_runner_protocol(monkeypatch) -> None:
    _stub_mlx(monkeypatch)

    runner = MlxStageRunner(model_class=object, config=MlxConfig(model_path="unused"))

    assert isinstance(runner, StageRunnerProtocol)
    assert runner.config.eos == 3


def test_mlx_backend_builds_a_protocol_runner(monkeypatch) -> None:
    _stub_mlx(monkeypatch)
    backend = MlxBackend()

    runner = backend.make_stage_runner(object, backend.make_config("unused"))

    assert isinstance(runner, StageRunnerProtocol)


def test_mlx_backend_config_forwards_scheduler_fields() -> None:
    backend = MlxBackend()
    options = {
        "kvcache_block_size": 128,
        "num_kvcache_blocks": 32,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "eos": 9,
    }

    defaults = backend.make_config("unused")
    config = backend.make_config("unused", **options)

    assert (
        defaults.kvcache_block_size,
        defaults.num_kvcache_blocks,
        defaults.max_num_seqs,
        defaults.max_num_batched_tokens,
        defaults.eos,
    ) == (256, 64, 512, 16384, None)
    assert (
        config.kvcache_block_size,
        config.num_kvcache_blocks,
        config.max_num_seqs,
        config.max_num_batched_tokens,
        config.eos,
    ) == (128, 32, 8, 4096, 9)


def test_mlx_prefill_join_extends_cache_rows_with_offsets(monkeypatch) -> None:
    runner = _fake_runner(monkeypatch, MlxConfig(model_path="unused"))
    old_sequences = [
        types.SimpleNamespace(token_ids=[1, 2, 3, 4, 5]),
        types.SimpleNamespace(token_ids=[6, 7, 8, 9, 10]),
    ]
    new_sequence = types.SimpleNamespace(token_ids=[11, 12, 13])

    input_ids, positions = runner.prepare_prefill(old_sequences)
    runner.forward(input_ids, positions, True)
    input_ids, positions = runner.prepare_prefill([new_sequence])
    runner.forward(input_ids, positions, True)

    cache = runner._cache[0]
    assert cache.row_count == 3
    assert cache._idx == 5
    assert cache.offset == [5, 5, 3]
    assert cache.left_padding == [0, 0, 2]


def test_mlx_decode_filters_finished_cache_rows_before_next_step(monkeypatch) -> None:
    runner = _fake_runner(
        monkeypatch, MlxConfig(model_path="unused", max_num_batched_tokens=5, eos=99)
    )

    def make_sampler(mx, sequences, samplings, keys):
        return lambda logits, temperatures: np.array(
            [sequence.token_ids[0] for sequence in sequences], dtype=np.int32
        )

    monkeypatch.setattr(mlx_backend, "_make_sampler", make_sampler)
    requests = [
        ThinkerDecode(ids, limit, 1.0)
        for ids, limit in (([1, 2, 3, 4, 5], 2), ([6, 7, 8], 3), ([9, 10, 11, 12, 13], 4))
    ]

    results = decode_minimind_mlx(runner, requests)

    assert runner._model.forward_rows == [1, 1, 1, 3, 2, 1]
    assert [result.generated for result in results] == [[1, 1], [6, 6, 6], [9, 9, 9, 9]]
    assert all(result.text_state is not None for result in results)
    assert runner._cache is None
