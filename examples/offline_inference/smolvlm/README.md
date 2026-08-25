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
