# nanovllm-omni

A small, local reference implementation of the MiniMind-O audio pipeline. Alignment work with vllm-omni is tracked in `.scratch/aligned-interfaces/`; this project does not claim to implement vllm-omni's full feature set.

## Status

Only the **MiniMind-O three-stage audio pipeline** is wired up and supported. Image generation, video generation, vision LLMs, VLA, and full-duplex S2S are aspirational and are not implemented.

## Supported models

| Model | Stages | Output | Weights |
|---|---:|---|---|
| MiniMind-O (`minimind-3o`) | 3 (Thinker → Talker → Code2Wav) | audio | `jingyaogong/minimind-3o` |

## Quickstart

Install the package and its existing dependencies:

```bash
pip install -e ".[dev]"
```

Pull the MiniMind-O + Mimi checkpoints once into local directories
(the bundle loader is offline-first and never auto-fetches):

```bash
hf download jingyaogong/minimind-3o --local-dir /home/mcig/minimind-3o
hf download kyutai/mimi             --local-dir /home/mcig/mimi
```

Then run the single-prompt smoke against the local weights:

```bash
cd examples/offline_inference/minimind_o
HF_HUB_OFFLINE=1 bash run_end2end.sh \
    --model /home/mcig/minimind-3o --mimi /home/mcig/mimi --out audio.wav
```

`audio.wav` lands in the example folder. A CUDA-capable machine with
at least 4 GB of VRAM is recommended; CPU inference works but is
slow.

The aligned API is available as it is implemented:

```python
from nanovllm_omni import Omni, SamplingParams

engine = Omni("jingyaogong/minimind-3o")
outputs = engine.generate(["hi"], SamplingParams(max_tokens=8))
with open("audio.wav", "wb") as f:
    f.write(outputs[0].multimodal_output["audio"].wav_bytes())
```

## Configuration

Pipeline topology lives in code (`nanovllm_omni/config/registry.py`); per-stage sampling and resource defaults live in `deploy/*.yaml`, read from `Path.cwd()` at runtime. Consult `.scratch/aligned-interfaces/SPEC.md` for the locked contract.

## Repository layout

```
nanovllm_omni/  # package implementation (config layer in nanovllm_omni/config/)
deploy/         # per-family sampling/resource defaults
examples/
    offline_inference/
        minimind_o/   # Python-seam audio smoke (single + batched)
    online_serving/
        minimind_o/   # curl/stdlib client for /v1/chat/completions
tests/          # automated tests
docs/           # project notes
.scratch/       # alignment specification and ticket checklists
```

## Scope

This repository intentionally excludes diffusion, vision, VLA, multi-model pipelines, distributed execution, WebSockets, and FastAPI/uvicorn serving. The aligned HTTP seam, when available, uses the Python standard library HTTP server.

## License

Apache-2.0. See [LICENSE](LICENSE).
