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
python examples/audio.py
```

The MiniMind-O smoke test writes `audio.wav` in the repository root. Model weights are downloaded from Hugging Face on first use; a CUDA-capable machine with sufficient memory is recommended.

The aligned API is available as it is implemented:

```python
from nanovllm_omni import Omni, SamplingParams

engine = Omni("jingyaogong/minimind-3o")
outputs = engine.generate(["hi"], SamplingParams(max_tokens=8))
with open("audio.wav", "wb") as f:
    f.write(outputs[0].multimodal_output["audio"].wav_bytes())
```

## Configuration

The legacy smoke test uses `configs/minimind_omni.yaml`. The aligned interface work will add the separated pipeline and deployment configuration under `deploy/`; consult `.scratch/aligned-interfaces/SPEC.md` for the locked contract.

## Repository layout

```
nanovllm_omni/  # package implementation
configs/        # existing pipeline configuration
examples/       # runnable examples
tests/          # automated tests
docs/           # project notes
.scratch/       # alignment specification and ticket checklists
```

## Scope

This repository intentionally excludes diffusion, vision, VLA, multi-model pipelines, distributed execution, WebSockets, and FastAPI/uvicorn serving. The aligned HTTP seam, when available, uses the Python standard library HTTP server.

## License

Apache-2.0. See [LICENSE](LICENSE).
