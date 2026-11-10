"""Convert the minimind-3o talker weights into an MLX talker model directory.

The checkpoint stores the talker under the ``talker.`` prefix; the prefix is
stripped and the remaining keys are written to ``model.safetensors`` next to
a ``config.json`` holding the talker architecture fields. The weights stay in
the checkpoint's bfloat16; the MLX runner casts them to its own dtype.

Only torch + safetensors are required; mlx is not. Run with the project venv:

    .venv/bin/python scripts/convert_minimind_talker_mlx.py

``--check`` compares the converted model against the torch one, so it needs
mlx as well:

    uv run --no-project --with mlx --with mlx-lm --with torch --python 3.12 \\
        python scripts/convert_minimind_talker_mlx.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import save_file

_WEIGHT_PREFIX = "talker."

_CONFIG_FIELDS = (
    "hidden_size",
    "talker_hidden_size",
    "num_talker_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "audio_vocab_size",
    "spk_emb_size",
    "audio_spk_token",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
)


def _talker_config(hf: dict) -> dict:
    """The talker fields of the checkpoint config, plus the model type."""
    config = {"model_type": "minimind-talker"}
    config.update({field: hf[field] for field in _CONFIG_FIELDS})
    return config


def convert(source: str, output: str, *, force: bool = False) -> str:
    """Write the converted talker directory and return the weights path."""
    checkpoint = os.path.join(source, "pytorch_model.bin")
    config_path = os.path.join(source, "config.json")
    if not os.path.isfile(checkpoint) or not os.path.isfile(config_path):
        raise SystemExit(f"expected pytorch_model.bin and config.json in {source}")

    target = os.path.join(output, "model.safetensors")
    if os.path.isfile(target) and not force:
        raise SystemExit(f"{target} exists; pass --force to rebuild")

    with open(config_path) as f:
        hf = json.load(f)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = {
        key[len(_WEIGHT_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(_WEIGHT_PREFIX)
    }
    if not weights:
        raise SystemExit(f"no talker.* weights in {checkpoint}")

    os.makedirs(output, exist_ok=True)
    tmp = target + ".tmp"
    save_file({key: weights[key].contiguous() for key in sorted(weights)}, tmp)
    os.replace(tmp, target)

    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(_talker_config(hf), f, indent=2, sort_keys=True)
        f.write("\n")
    return target


def check(source: str, output: str) -> float:
    """Compare the converted model's logits against the torch model's."""
    import mlx.core as mx

    from nanovllm_omni.backend.mlx_talker import load_mlx_talker
    from nanovllm_omni.models.minimind_omni.talker import load_talker

    torch_model = load_talker(source, device="cpu", dtype=torch.float16)
    mlx_model = load_mlx_talker(output, dtype="float16", device="gpu")

    generator = torch.Generator().manual_seed(0)
    bridge = torch.randn(1, 12, 768, generator=generator, dtype=torch.float32).half()
    codes = torch.randint(0, 2048, (1, 8, 12), generator=generator, dtype=torch.long)

    with torch.no_grad():
        torch_logits = torch_model(bridge, codes, None, None)
    mlx_logits = mlx_model(mx.array(bridge.numpy()), mx.array(codes.numpy()), None, None)

    worst = 0.0
    for index, (left, right) in enumerate(zip(torch_logits, mlx_logits, strict=True)):
        difference = float(mx.max(mx.abs(mx.array(left.numpy()) - right[0, -1, :])).item())
        worst = max(worst, difference)
        print(f"codebook {index}: max abs diff {difference:.6f}")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.expanduser("~/models/minimind-3o"))
    parser.add_argument("--output", default=None, help="defaults to <source>/mlx-talker")
    parser.add_argument("--force", action="store_true", help="overwrite an existing conversion")
    parser.add_argument(
        "--check", action="store_true", help="compare the converted model against the torch one"
    )
    args = parser.parse_args()

    source = os.path.abspath(os.path.expanduser(args.source))
    output = os.path.abspath(os.path.expanduser(args.output or os.path.join(source, "mlx-talker")))

    target = convert(source, output, force=args.force)
    with open(target, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wrote {target}")
    print(f"sha256 {digest}")

    if args.check:
        worst = check(source, output)
        print(f"max abs diff over the eight codebooks: {worst:.6f}")


if __name__ == "__main__":
    main()
