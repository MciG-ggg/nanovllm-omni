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

## Shapes vs. the repository's model classes (measured 2026-10-21)

The tensor shapes of the checkpoint match the constructors of
`MiniMindThinker` and `MiniMindTalker` in the repository:

| checkpoint key | shape | repository parameter |
|---|---|---|
| `model.embed_tokens.weight` | (6400, 768) | `VocabParallelEmbedding(vocab_size=6400, hidden_size=768)` |
| `model.layers.N.self_attn.q_proj.weight` | (768, 768) | 8 heads x head_dim 96 |
| `model.layers.N.self_attn.k_proj.weight` | (384, 768) | 4 kv heads x head_dim 96 |
| `model.layers.N.self_attn.{q,k}_norm.weight` | (96,) | head_dim 96 |
| `model.layers.N.mlp.{gate,up}_proj.weight` | (2432, 768) | `intermediate_size=2432` |
| `model.layers.N.mlp.down_proj.weight` | (768, 2432) | same width |
| `model.norm.weight` | (768,) | `RMSNorm(hidden_size)` |
| `lm_head.weight` | (6400, 768) | `ParallelLMHead(vocab_size=6400, hidden_size=768)`, tied to the embedding |
| `talker.layers.N.*` | same geometry | the talker reuses the thinker MLP width |
| `talker.lm_head.base.weight` | (2112, 768) | `Linear(talker_hidden=768, audio_vocab=2112)` |
| `talker.lm_head.adapters.N.{0,2}.weight` | (256, 768), (2112, 256) | `Linear(768, 256)` then `Linear(256, 2112)`, rank 256 |
| `talker.embed_tokens.base.weight` | (2112, 768) | `Embedding(audio_vocab=2112, talker_hidden=768)` |
| `talker.embed_tokens.adapters.N.{0,2}.weight` | (2112, 256), (768, 256) | `Embedding(2112, 256)` then `Linear(256, 768)` |
| `talker.codec_proj.{0,2}.{weight,bias}` | (768, 768), (768,) | `Linear(768, 768)` twice, with bias |
| `talker.codec_proj.3.weight` | (768,) | the `RMSNorm` at index 3 of the `Sequential` |
| `talker.embed_proj.{0,2}.{weight,bias}` | (768, 768), (768,) | `Linear(text_hidden, text_hidden)` then `Linear(text_hidden, talker_hidden)` |
| `talker.spk_proj.weight` | (768, 192) | `Linear(spk_emb=192, talker_hidden=768)`, renamed to `speaker_proj` |

The layer counts also agree: 88 thinker layer tensors are 11 per layer
x 8 layers, and 44 talker layer tensors are 11 per layer x 4 layers.

Still open: the fork's runtime loader mapping these names onto the
parameters above. That check needs the fork runtime and a GPU.

## One environment for all four families (verified 2026-10-21)

The SmolVLA run needs the transformers range that `lerobot[smolvla]`
pins (`>=5.4.0,<5.6.0`), and the MiniMind-O run needs the Mimi codec
from transformers. Both agree on transformers 5.5.4: the 5.5.4 wheel
ships `transformers/models/mimi/modeling_mimi.py` and
`configuration_mimi.py`, so `from transformers import MimiModel` works
on the version that the lerobot extra requires. One environment
therefore serves MiniMind-O, SmolVLM, SD-Turbo and SmolVLA.

The session setup is one command (`tools/colab_setup.py`):

1. clone the repository and the fork
2. remove the image transformers before the pinned install, because pip
   uninstalls the old version first and the old `RECORD` then deletes
   files that the new version already wrote
3. install the packages with a constraint file that pins the torch
   stack the image ships, so the CUDA build stays intact
4. uninstall the preinstalled torchao, whose missing `FqnToConfig` name
   breaks the diffusers import
5. install the nsys and ncu wheels
6. pre-download the five checkpoints
7. print the versions, the GPU name, and the import checks

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
