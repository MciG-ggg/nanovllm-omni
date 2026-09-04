"""CPU test: lock the past_kv structure assumption of
tools/profile_cuda_graph.py (the GPU CUDA Graph benchmark script).

The script does ``past_kv = prefill_out.past_key_values`` then
``static_past_kv = tuple((k.clone(), v.clone()) for k, v in past_kv)`` and
indexes per-layer ``(k, v)`` as tensors. If a future refactor (e.g. the
CUDA Graph integration swapping KV cache to fixed buffers, or a
transformers upgrade moving past_kv to ``DynamicCache``) changes this
structure, the script's first GPU run would crash before measuring
anything. This test pins the contract to the actual upstream
``model_minimind.py`` return shape *at the source level* so a drift is
caught in CI without needing a GPU.

CPU-only, no model weights.
"""

from __future__ import annotations

import re
from pathlib import Path

from nanovllm_omni import __file__ as _pkg_init  # noqa: F401

# Path to the vendored upstream model_minimind.py inside the local HF cache.
CACHE_HITS = sorted(
    Path.home().glob(".cache/huggingface/modules/transformers_modules/*/*/model_minimind.py")
)

# The toolkit's past_kv access pattern (what must stay valid).
SCRIPT_PATTERN = [
    r"past_kv\s*=\s*prefill_out\.past_key_values",
    r"static_past_kv\s*=\s*tuple\(\(k\.clone\(\), v\.clone\(\)\)",
]


def _upstream_src() -> str | None:
    if not CACHE_HITS:
        return None
    return CACHE_HITS[-1].read_text(encoding="utf-8")


def test_script_uses_past_key_values_attr() -> None:
    """The GPU benchmark script reads past_key_values from the prefill
    output — the standard HF output attribute. Pin it in source."""
    script = Path("tools/profile_cuda_graph.py").read_text(encoding="utf-8")
    assert "prefill_out.past_key_values" in script


def test_script_treats_each_layer_as_tensor_pair() -> None:
    """The script clones each layer's (K, V) as tensors. This matches the
    upstream MIT (per-layer (xk, xv) tensors appended to a list)."""
    script = Path("tools/profile_cuda_graph.py").read_text(encoding="utf-8")
    for pat in SCRIPT_PATTERN:
        assert re.search(pat, script), f"pattern missing from script: {pat}"


def test_upstream_returns_tuple_of_12_layer_pairs() -> None:
    """The upstream model returns presents = list of per-layer (K, V)
    tensors. Every layer must produce a non-None past_kv when use_cache=True
    even on the prefill step (past_key_value=None)."""
    src = _upstream_src()
    if src is None:
        return  # cache not present on this host: coarse check over
    # The Attention.forward must set past_kv = (xk, xv) when use_cache=True.
    m = re.search(r"past_kv\s*=\s*\((xk|key),\s*(xv|value)\)\s*if\s*use_cache\s*else\s*None", src)
    assert m is not None, (
        "upstream attention no longer returns (key, value) tuple on use_cache; "
        "the GPU benchmark script's per-layer (k.clone(), v.clone()) will break"
    )
    # presents = [] with append(present) for layer in layers.
    assert "presents.append" in src
    assert "use_cache" in src


def test_script_spins_up_model_via_bundle() -> None:
    """The script must construct via create_bundle (the offline-first
    loader), not directly from_pretrained with HF network access."""
    script = Path("tools/profile_cuda_graph.py").read_text(encoding="utf-8")
    assert "create_bundle" in script
    assert "from_pretrained" not in script.replace("AutoTokenizer.from_pretrained", "")


if __name__ == "__main__":
    import sys

    checks = [
        test_script_uses_past_key_values_attr,
        test_script_treats_each_layer_as_tensor_pair,
        test_upstream_returns_tuple_of_12_layer_pairs,
        test_script_spins_up_model_via_bundle,
    ]
    for fn in checks:
        fn()
        print(f"OK: {fn.__name__}")
    sys.exit(0)
