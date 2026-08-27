# nanovllm-omni

- 🎯 **MiniMind-O pipeline** — smallest full Thinker → Talker → Code2Wav runtime that loads real `jingyaogong/minimind-3o` weights
- ⚡ **Sub-340ms p50 on RTX 3050 laptop GPU** — fused QKV/gate-up projections, fused RMSNorm, fused RoPE (in `nanovllm_omni/models/minimind_omni/attention.py`); pre-allocated KV buffer; SDPA decode with `is_causal=True`
- 🔁 **StagePool pattern demo** — `num_replicas ≥ 2`, RoundRobin LB, `(stage_id, replica_id)` per output
- 🌐 **Unified omni I/O contract** — same `OmniRequestOutput` envelope for MiniMind-O (audio) + SmolVLM (text) + SD-Turbo (image) + SmolVLA (action)

A small, local reference implementation that exercises vllm-omni's stage-based serving architecture on a single card. This project does not claim to implement vllm-omni's full feature set.

## Supported models

| Model | Stages | Output | Weights |
|---|---:|---|---|
| MiniMind-O (`minimind-3o`) | 3 (Thinker → Talker → Code2Wav) | audio (24 kHz mono WAV) | `jingyaogong/minimind-3o` + `kyutai/mimi` |
| SmolVLM-500M-Instruct | 1 (VLM) | text | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| SD-Turbo | 1 (DIFFUSION, 1-step) | image (512×512 PNG) | `stabilityai/sd-turbo` |
| SmolVLA | 1 (LLM_GENERATION) | action chunks | `HuggingFaceTB/SmolVLA-256M` + LIBERO datasets |

All four run on a single 4 GB consumer card (RTX 3050) in one process. Image generation, video generation, vision LLMs beyond SmolVLM, full VLA stacks, and full-duplex S2S are aspirational and are not implemented.

## Quickstart

Install the package and its existing dependencies:

```bash
pip install -e ".[dev]"
```

For MiniMind-O (audio), pull the weights once into local directories
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

For SD-Turbo (image), SmolVLM (text), and SmolVLA (action), see the
per-family examples under `examples/offline_inference/<family>/`.

The aligned API is available across all four families:

```python
from nanovllm_omni import Omni, SamplingParams

# Audio (MiniMind-O)
engine = Omni("jingyaogong/minimind-3o")
outputs = engine.generate(["hi"], SamplingParams(max_tokens=8))
with open("audio.wav", "wb") as f:
    f.write(outputs[0].multimodal_output["audio"].wav_bytes())

# Image (SD-Turbo)
engine = Omni("stabilityai/sd-turbo")
outputs = engine.generate(["a red apple"], SamplingParams(max_tokens=1))
outputs[0].multimodal_output["image"].save("apple.png")

# Text (SmolVLM)
engine = Omni("HuggingFaceTB/SmolVLM-500M-Instruct")
outputs = engine.generate(["What is in this image? <image>"], SamplingParams(max_tokens=64))

# Action (SmolVLA)
engine = Omni("HuggingFaceTB/SmolVLA-256M")
outputs = engine.generate([{"prompt": "do the task", "image": rgb_obs}],
                          SamplingParams(max_tokens=50))
outputs[0].multimodal_output["actions"].array  # np.ndarray [chunk, action_dim]
```

## Configuration

Pipeline topology lives in code (`nanovllm_omni/config/registry.py`); per-stage sampling and resource defaults live in `deploy/*.yaml`, read from `Path.cwd()` at runtime.

## Repository layout

```
nanovllm_omni/  # package implementation (config layer in nanovllm_omni/config/)
deploy/         # per-family sampling/resource defaults
examples/
    offline_inference/
        minimind_o/   # Python-seam audio smoke (single + batched)
        sd_turbo/     # SD-Turbo image generation
        smolvla/      # SmolVLA policy (synthetic L1 + LIBERO eval)
        smolvlm/      # SmolVLM text/VLM
    online_serving/
        minimind_o/   # curl/stdlib client for /v1/chat/completions
tests/          # automated tests
docs/           # project notes and performance archives
```

## Scope

This repository intentionally excludes video generation, multi-model pipelines beyond the four listed, distributed execution, WebSockets, and FastAPI/uvicorn serving. The aligned HTTP seam, when available, uses the Python standard library HTTP server.

### Single-card, single-process runtime

One `Omni(...)` constructs one `MinimindBundle`
(`nanovllm_omni/models/minimind_omni/bundle.py`) holding the full
MiniMind-O checkpoint, the Mimi codec, and the tokenizer in one
Python process. There is no per-stage subprocess pool, no
`StageRuntime`, no multi-GPU dispatch, and no tensor-parallel
worker — by design.

This is a deliberate divergence from vllm-omni, where each stage
runs in its own `StageEngineCoreProc` subprocess and stages can
be pinned to separate GPUs. vllm-omni can do that because each
stage is an independent HF checkpoint (thinker 30B, talker 2B,
code2wav 1B, etc.). MiniMind-O's `thinker` and `talker` are
submodules of one `AutoModelForCausalLM` that loads from a single
safetensors file, so per-stage subprocess isolation would force
every stage to re-load the full ~3 GB checkpoint — unaffordable on
the 4 GB GPU we target and wasteful on anything bigger.
`bundle.py` instead keeps everything in one process; inter-stage
traffic (thinker hidden states → talker → code2wav) is Python
tensor references with zero serialization and zero IPC.

For request-level parallelism on the same GPU, use the in-process
batched runner (`examples/offline_inference/minimind_o/batched.py`,
verified by `tests/test_batched_generation.py` — Q10a).
Multi-card, per-stage subprocess isolation, tensor-parallel, and
pipeline-parallel schedulers are intentionally out of scope; if
any are added later, the change must start by reworking
`bundle.py` (one bundle per replica) and `engine/runtime.py`
(per-replica inference path), not by retrofitting vllm-omni's
`StageRuntime` into a runtime that has no use for it today.

The four supported model families all run on the same single-process
runtime: per-stage continuous batching (TK-004) and per-stage replica
+ RoundRobin LB (TK-007) are wired as in-process data structures
(`engine/runtime_scheduler.py`, `engine/load_balancer.py`), not as
subprocess pools.

## License

Apache-2.0. See [LICENSE](LICENSE).