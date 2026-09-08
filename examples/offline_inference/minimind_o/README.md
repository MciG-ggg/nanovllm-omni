# MiniMind-O: Offline inference

Single-process Python inference against the aligned API
(`nanovllm_omni.Omni`). Mirrors vllm-omni's per-model example layout
(`examples/offline_inference/<model>/`); one folder per model, README
+ entrypoint + thin shell wrappers.

## Setup

Weights are **offline-first**: the bundle loader never triggers a
network download. Pull both checkpoints once with `hf` into
`pretrained/`:

```bash
hf download jingyaogong/minimind-3o --local-dir pretrained/minimind-3o
hf download kyutai/mimi             --local-dir pretrained/mimi
```

That matches the `--model` / `--mimi` defaults. If you prefer the
Hub id, pass `--model jingyaogong/minimind-3o` explicitly and make
sure `~/.cache/huggingface/hub/` already has the snapshot —
otherwise the loader logs a warning naming `--model /path/to/local/dir`.

## Run examples

### Single prompt

```bash
cd examples/offline_inference/minimind_o
HF_HUB_OFFLINE=1 bash run_end2end.sh \
    --mimi pretrained/mimi --out audio.wav
```

`end2end.py` constructs `Omni(...)` directly, calls
`generate([prompt], sampling_params)` with sampling defaults from
`deploy/minimind_omni.yaml`, and writes a WAV file.

### Batched continuous-batching

```bash
cd examples/offline_inference/minimind_o
HF_HUB_OFFLINE=1 bash run_batched.sh \
    --mimi pretrained/mimi --out batched_smoke
```

`batched.py` exercises `models/minimind_omni/kv_pool.py + models/minimind_omni/batched_runner.py`
on two prompts and asserts:

- **Q10a determinism** — `solo_a.wav` is byte-identical to
  `batched_a.wav`, regardless of whether request A ran alone or
  alongside request B.
- **Real B>1 decode** — at least one decode step has `B=2` in the
  recorded forward shapes.
- All outputs decode as valid WAV.

## Notes

- `Omni(...)` resolves the pipeline topology from
  `nanovllm_omni/config/registry.py`; per-stage defaults come from
  `deploy/minimind_omni.yaml`. Override the deploy file with
  `--deploy-config /path/to/your.yaml`.
- The shell wrappers just `cd` to the example folder and run the
  Python entrypoint; `deploy/minimind_omni.yaml` is resolved by the
  engine from the package's `deploy/` directory.
- See `examples/online_serving/minimind_o/` for the HTTP `Omni`
  entrypoint and a curl-based chat client.