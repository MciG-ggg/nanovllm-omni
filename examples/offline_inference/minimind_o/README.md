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

## Gotchas

- **Chat template.** MiniMind was trained on chat-formatted text. `ThinkerStage`
  wraps a raw prompt with `tokenizer.apply_chat_template(..., add_generation_prompt=True,
  open_thinking=False)`; text that already contains `<|im_start|>` is passed through.
  Feeding unformatted text makes the thinker emit off-distribution tokens and the
  talker renders noise.
- **dtype.** The official `eval_omni.py` runs the released weights in fp16
  (`.half()`); the checkpoint's `dtype: bfloat16` is the training dtype. The
  MiniMind stages default to fp16. Pass `dtype="bfloat16"` only for experiments.
- **Talker.** Audio uses the vendor `MiniMindOmni.talker` (fp16) fed by the
  thinker bridge. The thinker stops its post-EOS filler at `content_len + 64`
  so the vendor talker can reach its stop token (55 frames / 4.40 s for the
  reference prompt; the HF teacher-forced reference on the *same tokens* is
  56 frames / 4.48 s).
- **Frame count is not a parity signal.** It only reflects which step the
  audio stop token landed on, so it moves whenever the sampled codes move.
  The broken pre-fix output and the official full model both happen to be
  72 frames / 5.76 s while differing in every other respect. Compare logits
  or argmax given the same input, not durations. ADR 0011 §4.2 and §5.
- **Latency.** With the default `max_tokens=512`, the thinker's early stop cuts
  the post-EOS filler and the talker still dominates the request. Numbers below
  are bench prompt `short_01` ("你好。") on WSL / RTX 3050, 37 frames, from
  `python -m nanovllm_omni.bench time --pipeline full --prompts short_01
  --max-tokens 512 --runs 3 --warmup 1`: thinker ~0.47 s, talker ~1.04 s,
  code2wav ~0.03 s, wall ~1.54 s. Do not compare these with the frame counts
  above — the example prompt produces more frames. Breakdown and history in
  ADR 0007.
- **Bridge.** The fork's blocks keep the residual separate, so the talker
  condition (`bridge_states`) must be the full block output `mlp_out + residual`
  — what the vendor captures. Copying only `mlp_out` leaves the text correct
  but the audio wrong (half amplitude, wrong length).
