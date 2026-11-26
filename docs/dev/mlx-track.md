# Mac 轨：mlx / mlx-lm 的装法与运行方式

本文件是 MLX 轨的操作说明。设计理由见 ADR 0013（backend 接缝）、
ADR 0021（kernel 路线）与 ADR 0023（执行顺序）。

## 现状

- 能跑：MiniMind-O thinker stage（`backend="mlx"`），包含文本生成、bridge hidden
  与 `thinker2talker` 交接；单序列与 dense batch（左补齐）都通。
- 能跑：MiniMind-O talker 的 MLX 路径（`MlxTalkerRunner`），支持 dense 和
  `talker_kv="windowed"`；需先把权重转换到 `<checkpoint>/mlx-talker/`。
- 已接入：SmolVLM 的 `backend="mlx-vlm"` 路径由 `mlx-vlm` 提供视觉塔、处理器
  与文本模型，当前 `max_batch=1`。
- 能跑但不属于 MLX：Code2Wav 使用 torch Mimi；Mac 上走 MPS，见 ADR 0024。
- 暂不支持：SD-Turbo / diffusers 与 SmolVLA / lerobot 的 MLX 路径。
- `MlxStageRunner` 仍走 dense KV；`backend/mlx_kernel/` 中已有 Metal paged
  decode attention / KV store kernel 和数值测试，但尚未接入 runner。

## 为什么 mlx 不进项目依赖

`mlx` 只有 macOS arm64 wheel。`pyproject.toml` 是跨平台的，把 mlx 写进去会让
Linux / CUDA 的安装失败。所以 MLX 轨用 `uv run --no-project --with ...`
临时建环境，项目依赖保持不动。

## 一次性准备

### 1. 下载权重（不进 git）

```bash
export HTTP_PROXY=http://127.0.0.1:29659
export HTTPS_PROXY=http://127.0.0.1:29659
.venv/bin/hf download jingyaogong/minimind-3o --local-dir ~/models/minimind-3o
```

代理地址是这台机器的本地事实；不带代理时 HF 直连会停在 0 字节。
权重约 216 MB，落在 `~/models/minimind-3o/`。

### 2. 转换成 mlx_lm 模型目录

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
.venv/bin/python scripts/convert_minimind_thinker_mlx.py
```

只需要项目 venv 里的 torch + safetensors，不需要 mlx。输出
`~/models/minimind-3o/mlx-thinker/`。默认拒绝覆盖已有结果，重建用 `--force`。

可重复性检查：

```bash
.venv/bin/python scripts/convert_minimind_thinker_mlx.py --output ~/models/minimind-3o/mlx-thinker-repro --force
shasum -a 256 ~/models/minimind-3o/mlx-thinker/model.safetensors \
              ~/models/minimind-3o/mlx-thinker-repro/model.safetensors
```

两个 sha256 必须相同；当前值是
`f0525e44619dad670a67a1f97447aab2196404b7b682708a4a0ccd2f30deb760`。

### 3. 验证 MLX 环境

```bash
uv run --no-project --with mlx --with mlx-lm --python 3.12 python -c \
  "import mlx.core as mx; print(mx.default_device(), mx.device_info()['max_recommended_working_set_size'])"
```

已验证版本：mlx 0.32.4、mlx-lm 0.32.2、Python 3.12。输出应包含
`Device(gpu, 0)` 与 `12713115648`（11.84 GiB）。

## 跑 thinker

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=$PWD:$PWD/third_party/nano-vllm

# 单序列
uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/bench_mlx_thinker.py --runs 5 --warmup 2

# dense batch（4 条，长度 19/45/22/53，补齐到 53）
uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python tools/bench_mlx_thinker.py --runs 5 --warmup 2 \
    --batch-prompts short_01,medium_01,short_02,medium_02 --max-tokens 32
```

`PYTHONPATH` 里的 `third_party/nano-vllm` 提供 fork 的 `Sequence` 与
`SamplingParams`；MLX 后端只借用这两个纯 Python 对象。

数字与限制记录在 `docs/perf/minimind-thinker-mlx-2026-11-09.md`。

## 跑 thinker stage

`ThinkerStage` 走 `extra["backend"]` 选驱动：`"mlx"` 时加载
`<checkpoint>/mlx-thinker`（不是 torch checkpoint），并用
`decode_minimind_mlx` 跑 vendor 的 decode 舞步。

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=$PWD:$PWD/third_party/nano-vllm

uv run --no-project --with mlx --with mlx-lm --with torch --with transformers \
    --python 3.12 python -m pytest tests/test_minimind_thinker_mlx.py -q
```

MLX 轨不需要 triton / NCCL：`ThinkerStage` 在 `backend="mlx"` 时不 import
torch 的 thinker 模块，也不建进程组。

## 转换 MLX talker 权重

转换脚本只需要项目 venv 里的 torch + safetensors，不需要安装 MLX。默认读取
`~/models/minimind-3o`，写到 `~/models/minimind-3o/mlx-talker/`：

```bash
.venv/bin/python scripts/convert_minimind_talker_mlx.py
```

要比对转换后的 logits，再加 `--check`；这一步需要 MLX：

```bash
uv run --no-project --with mlx --with mlx-lm --with torch --python 3.12 \
    python scripts/convert_minimind_talker_mlx.py --check
```

MLX `TalkerStage` 从 `<checkpoint>/mlx-talker` 加载转换结果。选择 `backend="mlx"`
时使用 `MlxTalkerRunner`；dense 与 `talker_kv="windowed"` 均可用。

## 验证 Metal kernel

Metal kernel 目前是独立实现，尚未接入 `MlxStageRunner`。在 Apple Silicon 上可用
`build.py` 让 `xcrun metal` 编译 MLX 实际生成的 kernel source，再运行数值测试：

```bash
uv run --no-project --with mlx --with mlx-lm --python 3.12 \
    python -m nanovllm_omni.backend.mlx_kernel.build
uv run --no-project --with mlx --with mlx-lm --with pytest --python 3.12 \
    python -m pytest tests/test_mlx_kernel.py -q
```

编译产物只写入临时目录并删除。测试比较 paged decode attention 与 dense `BatchKVCache`
参考；这不会让 `MlxStageRunner` 改用 paged kernel。

## 测试与 lint

```bash
.venv/bin/python -m pytest tests/test_mlx_backend.py tests/test_backend_wiring.py -q
.venv/bin/ruff check nanovllm_omni/ tests/ tools/
.venv/bin/black --check nanovllm_omni/ tests/ tools/
```

`tests/test_mlx_backend.py` 用 stub 的 `mlx` / `mlx_lm` 断言
`MlxStageRunner` 满足 `StageRunner` 协议，不需要装 mlx。真实模型路径由
`tools/bench_mlx_thinker.py` 覆盖。

## mlx-lm 0.32.2 的实际 API（introspect 结果）

```text
mlx_lm.utils.load(
    path_or_hf_repo: str, tokenizer_config=None, model_config=None,
    adapter_path=None, lazy=False, return_config=False,
    revision=None, trust_remote_code=False,
) -> (model, tokenizer) | (model, tokenizer, config)

qwen3.Model.__init__(self, args: qwen3.ModelArgs)
qwen3.Model.__call__(self, inputs: mx.array, cache=None, input_embeddings=None)
qwen3.Qwen3Model.__call__(self, inputs: mx.array, cache=None, input_embeddings=None)
qwen3.Attention.__call__(self, x, mask=None, cache=None) -> mx.array

mlx_lm.models.cache.make_prompt_cache(model, max_kv_size=None) -> List[Any]
mlx_lm.models.cache.KVCache()
mlx_lm.models.cache.BatchKVCache(left_padding: List[int])
BatchKVCache.make_mask(N: int, return_array=False, **kwargs)

mlx_lm.generate.BatchGenerator.__init__(
    self, model, *, max_tokens=128, stop_tokens=None, sampler=None,
    logits_processors=None, completion_batch_size=32,
    prefill_batch_size=8, prefill_step_size=2048,
    max_kv_size=None, stream=None,
)
BatchGenerator.insert(prompts, max_tokens=None, caches=None, all_tokens=None,
                      samplers=None, logits_processors=None, stop_sequences=None)
BatchGenerator.next(self)

mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask=...)
# 有 mask，没有 cu_seqlens
```

## Gotchas

- **`mlx_lm.generate` 是函数，不是模块。** `from mlx_lm import generate` 拿到的是
  `generate()`；`import mlx_lm.generate as gen` 也会解析成那个函数。要拿
  `BatchGenerator`，用 `importlib.import_module("mlx_lm.generate")`。本 backend
  不用 `BatchGenerator`（nanovllm-omni 自己的调度器才是被讲的东西）。
- **`utils.load` 不展开 `~`。** 传绝对路径；`~/models/...` 会被当成 HF repo id，
  报 `HFValidationError`。
- **左补齐必须配 `BatchKVCache`。** 它的 `offset` 是逐行数组（初值
  `-left_padding`），真实 token 因此拿到 0..len-1 的 RoPE 位置，pad 列由
  `make_mask` 屏蔽。换成普通 `KVCache` 会让每条序列的位置都从补齐后的长度起算。
- **`mlx_lm.models.base` 在 0.32.2 里没有 `Model` 类。** 模型类在各
  `mlx_lm.models.<arch>` 模块里；缓存类在 `mlx_lm.models.cache`。
- **没有 varlen。** `mx.fast.scaled_dot_product_attention` 不接受 `cu_seqlens`。
  真 CB 需要自研 Metal kernel（ADR 0021），接口设计见
  `docs/dev/mlx-varlen-kernel-interface.md`。
- **RNG 流与 torch 不同。** MLX 用 `mx.random.key` / `mx.random.categorical`，
  CUDA 侧用 `torch.Generator` / `torch.multinomial`。相同 seed 只对齐参数，
  不对齐随机流；ADR 0021 的 token 级一致不能靠 seed 单独达成。
- **`tie_word_embeddings`。** 转换脚本判定 lm_head 与 embed_tokens 逐位相同后
  不写入 `lm_head.weight`，config 里置 `tie_word_embeddings=true`。手工拷权重时
  不要把这个键加回去。
- **MLX 的 bridge recorder 只留最后一次 forward 的行。** talker 要的是整条序列的
  bridge，所以 MLX 轨的 stage 用一次全序列 forward 重跑（fresh cache）取行，
  跟 CUDA 轨的 `recapture_bridge_minimind` 同一个理由。
- **`OmniEngineArgs.device` 是 torch 的旋钮。** 没有 CUDA 的机器上它读 `"cpu"`，
  MLX backend 不能拿它——MLX 走统一内存的 GPU。
- **内存上限 11.84 GiB。** 当前 thinker 峰值 214.5 MiB（4 行 dense batch）。
  加模型前先看 `mx.get_peak_memory()`；Air 无风扇，长时间跑会热降频。
