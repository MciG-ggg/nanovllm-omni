# SmolVLM-500M-Instruct: Offline inference

Single-stage VLM (image + text -> text). The example synthesises a
224x224 gradient PNG when `--image` is omitted, so the demo runs
end-to-end without any external asset.

## Setup (one-off)

```bash
# Cache the SmolVLM weights (~1 GB):
mkdir -p pretrained
hf download HuggingFaceTB/SmolVLM-500M-Instruct --local-dir pretrained/SmolVLM-500M-Instruct
# transformers + torchvision must be importable (torchvision is the
# image-processor backend for SmolVLM in transformers 4.56+).
pip install transformers torchvision
```

## Run

```bash
# Default: synthesise a 224x224 gradient image.
python examples/offline_inference/smolvlm/run.py \
    --model pretrained/SmolVLM-500M-Instruct \
    --prompt "Describe this image briefly." \
    --max-new-tokens 128 \
    --device mps
# Real image:
python examples/offline_inference/smolvlm/run.py \
    --image /path/to/photo.jpg \
    --prompt "What's in this photo?" \
    --device cuda
```

Output prints the prompt + generated text to stdout (no `--output`).

## Pipeline

| # | stage | kind             | input                   | output      |
|---|-------|------------------|-------------------------|-------------|
| 0 | `vlm` | `LLM_GENERATION` | text + PIL image list   | text (terminal) |

Heavy deps (`transformers`, `torch`, `torchvision`) stay inside the
factory and forward, so importing `nanovllm_omni` does not require
them. Image routing rides in `SamplingParams.extra["images"]`; the
runner shape is unchanged from every other family.

## VRAM

BF16 weights + image processor + KV cache: ~1-1.5 GB. Fits in 4 GB
(RTX 3050). The 3050 4 GB AGENTS.md budget is honoured for this
family — this is the first pure-VLM under that cap.

## Smoke (Mac M-series, MPS)

```text
$ python examples/offline_inference/smolvlm/run.py \
    --model pretrained/SmolVLM-500M-Instruct \
    --prompt "Describe this image briefly." --max-new-tokens 128 --device mps
=== output ===
 This is a colorful background with many colors.
```

Cold load + inference: ~7-12 s. Subsequent runs are <100 ms warm.

## Out of scope

This family is image+text -> text only. Streaming generation, multi-image
batched VQA, and HTTP `image_url` blocks on `/v1/chat/completions` are
separate tickets. `SmolVLM-256M` / `SmolVLM-2.2B` / `InternVL` /
`Qwen2.5-VL` are separate family tickets.

## Apple Silicon (MLX)

The torch track above rides the fork's triton attention kernel, which has
no macOS wheel, so it cannot start on a Mac. `run_mlx.py` is the same
pipeline on the `mlx-vlm` backend. It is a different checkpoint format:
`mlx-vlm convert` rewrites the HF weights.

### Setup

```bash
uv tool install mlx-vlm        # or just use the uv run below
mlx-vlm convert --hf-path HuggingFaceTB/SmolVLM-500M-Instruct \
    -o ~/models/smolvlm-500m-instruct
```

`torchvision` is required by transformers 5's Idefics3 image processor.
Without it the processor silently falls back to numpy output and the MLX
embedding lookup then fails with a shape error.

### Run

```bash
bash run_mlx.sh --prompt "What is in this image?" --image /path/to/cat.jpg
```

`run_mlx.sh` runs in a throwaway `uv` env with `mlx`, `mlx-vlm` and
`torchvision`; the project venv deliberately does not carry them
(`tests/test_backend_wiring.py` pins that the backend registry imports
neither framework).

### Notes

- Backend name is `mlx-vlm`, not `mlx`: SmolVLM-500M declares
  `model_type: idefics3`, and the vision tower lives in mlx-vlm's
  Idefics3 implementation.
- `device="cpu"` is what the example passes: MLX reads its own device, and
  `MlxVlmBackend` substitutes `gpu` for any torch device name.
- Measured on an M2 Air: cold load + one 512x512 image ≈ 10 s, greedy
  output ` The image is red in colour.` for a solid red square.
- `tests/test_mlx_vlm.py` pins the logits width (49280), the merged
  embedding shape (1139 prompt tokens, 1088 of them image slots, 960
  hidden) and the greedy text.
