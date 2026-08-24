# SmolVLA: Offline inference

VLA policy inference against the aligned Python seam
(``nanovllm_omni.Omni``). Same layout convention as
``offline_inference/minimind_o/`` — one model per folder, README +
entrypoint(s) + thin shell wrappers. The folder carries two entries
because the model has two natural demo shapes:

- ``synthetic_obs.py`` — L1 sanity check on synthetic pixels / state,
  verifies action-shape contract without a real dataset.
- ``libero_eval.py`` — real LIBERO env evaluation; runs the official
  lerobot eval chain through the SmolVLA pipeline stage.

## Setup

Weights are **offline-first** (see ``offline_inference/minimind_o/README.md``).
Pull both checkpoints once with ``hf``:

```bash
hf download HuggingFaceVLA/smolvla_libero --local-dir pretrained/smolvla_libero
hf download HuggingFaceVLA/smolvlm2-500m --local-dir pretrained/smolvlm2-500m
```

The bundle loader is offline-first; no Hub fetches at runtime. Install
the model-specific extras so the SmolVLA stage's policy import works:

```bash
pip install -e '.[smolvla]'  # torch, transformers, lerobot, Pillow, ...
```

## Run examples

#### Synthetic L1 demo

```bash
cd examples/offline_inference/smolvla
HF_HUB_OFFLINE=1 bash run_synthetic.sh \
    --model pretrained/smolvla_libero --dtype int8 --device cuda
```

`synthetic_obs.py` constructs `Omni(...)`, feeds a random 256x256
image + zero state through `SamplingParams.extra`, and prints the
returned `ActionArtifact` shape / dtype / chunk size. Use this as
the smoke test on a fresh install.

#### Real LIBERO evaluation

```bash
cd examples/offline_inference/smolvla
HF_HUB_OFFLINE=1 bash run_libero.sh \
    --model pretrained/smolvla_libero --num-episodes 2 \
    --task-suite libero_object --task-id 5
```

`libero_eval.py` builds a `LiberoEnv`, collects raw lerobot obs, and
feeds each one through `Omni.generate(...)`. The SmolVLA stage owns
the official preprocessing chain (flip + quat->axis-angle, policy
pre/post, chunked select_action) — the script does NOT preprocess
itself, by design, since hand-rolled obs led to 0% SR.

## Notes

- Both entrypoints share the offline-first weight contract; the bundle
  loader logs a `WARNING` (not an exception) when the Hub id is missing
  locally, mirroring the behavior pinned by
  `tests/test_bundle_resolve_snapshot.py`.
- `--dtype int8` is wired for the 4 GB RTX 3050 used by the smoke
  path; it requires `bitsandbytes` to be installed for CUDA int8
  quant. On hosts with more VRAM, drop `--dtype` for fp16.
- The SmolVLA pipeline stage is registered as `smolvla` in
  `nanovllm_omni/config/registry.py`; per-stage defaults live in
  `deploy/smolvla.yaml`. Override with `--deploy-config`.