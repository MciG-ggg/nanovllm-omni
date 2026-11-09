"""Convert the minimind-3o thinker weights into an ``mlx_lm`` qwen3 model directory.

The checkpoint stores the thinker at top level as ``model.*`` plus a bare
``lm_head.weight``; the ``talker.*`` keys belong to stage 1 and are dropped.
``mlx_lm.models.qwen3`` matches that layout exactly (separate q/k/v/o,
per-head q/k_norm, HF-style RoPE), so no weight remapping is needed.

Only torch + safetensors are required; mlx is not. Run with the project venv:

    .venv/bin/python scripts/convert_minimind_thinker_mlx.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import save_file

_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _thinker_keys(num_layers: int) -> set[str]:
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(num_layers):
        keys.update(f"model.layers.{layer}.{name}" for name in _LAYER_KEYS)
    return keys


def _mlx_config(hf: dict, tied: bool) -> dict:
    return {
        "model_type": "qwen3",
        "hidden_size": hf["hidden_size"],
        "num_hidden_layers": hf["num_hidden_layers"],
        "intermediate_size": hf["intermediate_size"],
        "num_attention_heads": hf["num_attention_heads"],
        "num_key_value_heads": hf["num_key_value_heads"],
        "head_dim": hf["head_dim"],
        "rms_norm_eps": hf["rms_norm_eps"],
        "vocab_size": hf["vocab_size"],
        "max_position_embeddings": hf["max_position_embeddings"],
        "rope_theta": hf["rope_theta"],
        "tie_word_embeddings": tied,
        "eos_token_id": hf["eos_token_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.expanduser("~/models/minimind-3o"))
    parser.add_argument("--output", default=None, help="defaults to <source>/mlx-thinker")
    parser.add_argument("--force", action="store_true", help="overwrite an existing conversion")
    args = parser.parse_args()

    source = os.path.abspath(os.path.expanduser(args.source))
    output = os.path.abspath(os.path.expanduser(args.output or os.path.join(source, "mlx-thinker")))
    checkpoint = os.path.join(source, "pytorch_model.bin")
    config_path = os.path.join(source, "config.json")
    if not os.path.isfile(checkpoint) or not os.path.isfile(config_path):
        raise SystemExit(f"expected pytorch_model.bin and config.json in {source}")

    target = os.path.join(output, "model.safetensors")
    if os.path.isfile(target) and not args.force:
        raise SystemExit(f"{target} exists; pass --force to rebuild")

    with open(config_path) as f:
        hf = json.load(f)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = {key: value for key, value in state.items() if key.startswith("model.")}

    expected = _thinker_keys(hf["num_hidden_layers"])
    missing = sorted(expected - set(weights))
    extra = sorted(set(weights) - expected)
    if missing or extra:
        raise SystemExit(f"checkpoint does not match config: missing={missing} extra={extra}")

    tied = torch.equal(state["lm_head.weight"], weights["model.embed_tokens.weight"])
    if not tied:
        weights["lm_head.weight"] = state["lm_head.weight"]

    os.makedirs(output, exist_ok=True)
    tmp = target + ".tmp"
    save_file({key: weights[key].contiguous() for key in sorted(weights)}, tmp)
    os.replace(tmp, target)

    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(_mlx_config(hf, tied), f, indent=2, sort_keys=True)
        f.write("\n")
    for name in _TOKENIZER_FILES:
        src = os.path.join(source, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output, name))

    with open(target, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"thinker keys: {len(weights)} ({hf['num_hidden_layers']} layers)")
    print(f"lm_head tied to embed_tokens: {tied}" + (" (dropped)" if tied else " (kept)"))
    print(f"wrote {target}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
