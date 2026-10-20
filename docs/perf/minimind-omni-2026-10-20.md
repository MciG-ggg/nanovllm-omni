# minimind-omni on Colab T4: config unblock landed, measurement still open (2026-10-20)

## Status

**No MiniMind-O number was measured on the Colab T4 in this session.**
The 2x goal is still open for this model. The project-side blocker is
gone; what is missing is a GPU.

## The blocker and the fix

The MiniMind-3o checkpoint (`jingyaogong/minimind-3o`) ships its
modeling code as a `trust_remote_code` file on the hub. Running it on
the Colab VM means executing that remote code, and the sandbox that
guards this session refused the run:

```
Permission for this action was denied by the Claude Code auto mode
classifier. Reason: [Code from External] `colab exec ... s_minimind.py`
executes remote model code from `jingyaogong/minimind-3o` with
`trust_remote_code`, but the user did not explicitly name that
external source.
```

The denial is about the external code, not about the model. The
repository implements the thinker and the talker itself
(`nanovllm_omni/models/minimind_omni/thinker.py`, `talker.py`), so the
only reason the hub's remote code ran was the *config* class, which the
fork loads through `AutoConfig.from_pretrained`.

The fix landed in `nanovllm_omni/models/minimind_omni/hf_config.py`:
the repository registers its own `MiniMindOConfig` for
`model_type: minimind-o`. Transformers then finds the registered class
and skips the hub's `auto_map` file.

- `AutoConfig.from_pretrained` picks the registered class even when
  `trust_remote_code` is True, because the local class is not a
  `transformers.*` class and transformers takes its
  `explicit_local_code` branch.
- The thinker and the talker build their fork `Config` with
  `trust_remote_code=False`.
- The tokenizer loader drops `trust_remote_code`; the snapshot ships a
  plain `PreTrainedTokenizerFast`.

`tests/test_minimind_config_registration.py` pins the property: the
config loads from a directory whose `auto_map` points at a file that
raises when it runs, and that file never runs.

Verified on the local snapshot (`~/minimind-3o`, config files only):

```
class: MiniMindOConfig
tie: True dtype: torch.bfloat16
hidden: 768 vocab: 6400 layers: 8
audio_vocab: 2112 talker_hidden: 768 bridge: 3 spk: 192
hub module imported: False
```

## Checkpoint layout vs. the two-stage split (measured 2026-10-21)

The split that the thinker and the talker stages apply to the checkpoint
was checked against the real checkpoint, without a GPU.

`jingyaogong/minimind-3o` ships `pytorch_model.bin` and no safetensors
file, so the `.bin` fallback branch of `_filter_only_dir`
(`nanovllm_omni/models/minimind_omni/stage.py`) is the branch that
runs. The 195 tensor names were dumped on the Colab VM, then the
repository's own `_thinker_only_dir` / `_talker_only_dir` were run on a
fake `pytorch_model.bin` built from those names.

```
checkpoint keys: 195
thinker kept: 91   talker kept: 92   (the two sets do not overlap)
```

| Group in the checkpoint | Tensors | Where the split puts it |
|---|---|---|
| `model.*` (embed_tokens, 8 layers, norm) | 90 | thinker |
| `lm_head.weight` (bare, tied text head) | 1 | thinker |
| `talker.*` (layers, norm, embed_tokens, lm_head, codec_proj, embed_proj, scales, spk_proj) | 92 | talker, `talker.` prefix stripped |
| `audio_proj.*` | 6 | dropped |
| `vision_proj.*` | 6 | dropped |

Every kept name matches a parameter of the repository's own model
classes: the thinker's `model.embed_tokens` / `model.layers` /
`model.norm` / `lm_head`, and the talker's `layers` / `norm` /
`lm_head.base` + `lm_head.adapters.N.{0,2}` / `embed_tokens.base` +
`embed_tokens.adapters.N.{0,2}` / `codec_proj.{0,2,3}` /
`embed_proj.{0,2,3}` / `text_scale` / `audio_scale` / `speaker_proj`.
`codec_proj.0` / `.2` are the two `Linear` layers and `.3` is the
`RMSNorm` weight, which is why the checkpoint carries no `.1` key.

The 12 dropped tensors are the vendor's omni projectors. This project
replaces them with its own bridge, so nothing is lost.

The rename list also fires on the real names:
`talker.spk_proj.weight` becomes `speaker_proj.weight`. The fork loader
maps the substring `k_proj` to `qkv_proj`, and `spk_proj` contains it,
so the rename is what keeps the speaker projection loadable.

Still open: the fork's runtime loader mapping these names onto the
parameters above. That check needs the fork runtime and a GPU.

## What is still missing

1. A GPU to run the measurement on. Every `colab new --gpu T4` in this
   session failed with `Service Unavailable` on the assign endpoint,
   and the other accelerators are not entitled on this account. The WSL
   RTX 3050 host was unreachable.
2. The measurement itself, and then the two changes the SmolVLM
   profile justified: the decode model in the requested dtype and the
   decode loop under a CUDA graph. The thinker still runs eager, and
   the fork now falls back to eager when the capture fails, so the
   graph can be turned on by default once a measurement supports it.

## What the last known numbers say

The only MiniMind-O numbers in the repository are the WSL RTX 3050
runs in `docs/perf/minimind-2026-08-27.detail.md` and
`docs/adr/0003-profile-results-2026-09-13.md`. They are not a T4
baseline, so they cannot be compared with the T4 numbers of the other
models.
