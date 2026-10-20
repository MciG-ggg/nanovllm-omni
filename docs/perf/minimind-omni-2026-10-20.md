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
