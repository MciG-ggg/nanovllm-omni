#!/usr/bin/env python3
"""Compile the MLX Metal kernels with ``xcrun metal`` as a syntax gate.

``mx.fast.metal_kernel`` only accepts a source *string* -- there is no API in
MLX 0.32.4 to load a ``.metallib`` (``mx.fast.precompiled_cuda_kernel`` has a
CUDA sibling, nothing for Metal). So the shipped path is always JIT from the
``.metal`` source, and this script exists to fail *before* that JIT does.

It compiles the translation unit MLX actually generates: ``verbose=True`` makes
``metal_kernel`` print the assembled source, and that exact text goes through
``xcrun metal``. Gating on the post-generation source rather than on
``paged_attention.metal`` alone is what catches a body that upsets the
signature MLX builds around it.

Usable as a script (``python -m ...build``) and as a check function the tests
import. The ``.air`` / ``.metallib`` files are written to a temp dir and
deleted: the compile artifact is not a deliverable, the gate is.

    uv run --no-project --with mlx --with mlx-lm --python 3.12 \\
        python nanovllm_omni/backend/mlx_kernel/build.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

from . import kernels as kernels_module
from .kernels import DecodeShape

_GENERATED = re.compile(r"Generated source code for `(?P<name>\w+)`:\n```\n(?P<src>.*?)\n```", re.S)

# MLX's ``verbose`` printout is the body it assembles, but not the whole
# translation unit: at JIT time it also prepends its own prelude, which is where
# ``float16_t``, ``simd_sum`` and ``metal::max`` come from. ``xcrun metal``
# needs that prelude spelled out or it rejects a body MLX compiles fine.
PRELUDE = """#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef half float16_t;
typedef bfloat bfloat16_t;
"""


def generated_source(kernel: object, launch: dict) -> str:
    """The Metal translation unit ``metal_kernel`` assembles for one launch."""
    # ``verbose`` prints from C, so it goes to file descriptor 1 and bypasses
    # ``sys.stdout`` entirely -- including pytest's replacement of it.
    captured = ""
    saved_stdout = os.dup(1)
    try:
        with tempfile.TemporaryFile(mode="w+") as sink:
            os.dup2(sink.fileno(), 1)
            try:
                kernel(verbose=True, **launch)
            finally:
                os.dup2(saved_stdout, 1)
            sink.seek(0)
            captured = sink.read()
    finally:
        os.close(saved_stdout)
    match = _GENERATED.search(captured)
    if match is None:
        raise RuntimeError(f"metal_kernel printed no source for {kernel!r}")
    return match.group("src")


def _launch_shapes(shape: DecodeShape, dtype: object) -> tuple[dict, dict]:
    """Launch kwargs for both kernels, sized for a one-sequence probe."""
    batch, pool_rows, tokens = 1, 8, 1
    heads, kv_heads, head_dim = shape.num_heads, shape.num_kv_heads, shape.head_dim
    decode = dict(  # noqa: C408 -- kwargs mirror mx.fast.metal_kernel's signature
        inputs=[
            mx.zeros((batch, heads, head_dim), dtype),
            mx.zeros((pool_rows, kv_heads, head_dim), dtype),
            mx.zeros((pool_rows, kv_heads, head_dim), dtype),
            mx.zeros((batch, 1), mx.int32),
            mx.zeros((batch,), mx.int32),
            mx.zeros((1,), mx.float32),
            mx.zeros((1,), mx.int32),
        ],
        template=shape.template(dtype),
        grid=(batch * heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, heads, head_dim)],
        output_dtypes=[dtype],
    )
    store = dict(  # noqa: C408 -- kwargs mirror mx.fast.metal_kernel's signature
        inputs=[
            mx.zeros((tokens, kv_heads, head_dim), dtype),
            mx.zeros((tokens, kv_heads, head_dim), dtype),
            mx.zeros((pool_rows, kv_heads, head_dim), dtype),
            mx.zeros((pool_rows, kv_heads, head_dim), dtype),
            mx.zeros((tokens,), mx.int32),
            mx.zeros((2,), mx.int32),
        ],
        template=[("D", head_dim), ("NUM_KV_HEADS", kv_heads), ("T", dtype)],
        grid=(pool_rows * kv_heads * head_dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pool_rows, kv_heads, head_dim)] * 2,
        output_dtypes=[dtype, dtype],
    )
    return decode, store


def build(shape: DecodeShape | None = None, dtype: object | None = None) -> list[str]:
    """Compile every kernel with ``xcrun metal``; returns the checked names."""
    if shutil.which("xcrun") is None:
        raise RuntimeError("xcrun not found: install the Xcode command line tools")
    shape = shape or DecodeShape(num_heads=8, num_kv_heads=4, head_dim=96)
    dtype = dtype or mx.float16

    names = ("paged_decode_attention", "store_kv")
    launches = dict(zip(names, _launch_shapes(shape, dtype), strict=True))
    checked = []
    with tempfile.TemporaryDirectory() as workdir:
        # Fresh objects per call: ``metal_kernel`` prints its generated source
        # only on the run that builds it, so a module-level singleton is
        # already cached and silent by the second call.
        for name, kernel in zip(
            ("paged_decode_attention", "store_kv"), kernels_module._make_kernels(), strict=True
        ):
            source = generated_source(kernel, launches[name])
            metal = Path(workdir, f"{name}.metal")
            # The prelude above is deliberately the minimum a body needs, so
            # the gate rejects anything that leans on more than MLX supplies.
            metal.write_text(PRELUDE + source)
            air = Path(workdir, f"{name}.air")
            subprocess.run(
                ["xcrun", "metal", "-c", str(metal), "-o", str(air)],
                check=True,
                capture_output=True,
                text=True,
            )
            checked.append(name)
    return checked


def main() -> int:
    try:
        names = build()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        print(f"metal build failed:\n{detail}", file=sys.stderr)
        return 1
    print(f"compiled: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
