# Vendored modeling code — jingyaogong/minimind-3o

The Python source files in this directory are a verbatim copy of the
modeling code shipped with the [jingyaogong/minimind-3o](https://huggingface.co/jingyaogong/minimind-3o)
Hugging Face checkpoint:

- `model_omni.py` — `OmniConfig`, `MiniMindOmni` (the `AutoModelForCausalLM` target), and the audio / vision / talker modules wired around the base MiniMind LM.
- `model_minimind.py` — base `MiniMindConfig` and the underlying MiniMindForCausalLM transformer.

Both files are byte-for-byte identical to upstream at the snapshot whose
audio.wav parity MD5 was `536fad2aba93d7b5df76067195dca26c`, **except** the
single TK-016 phase 3.b patch below. If you need to upgrade the vendored
copy to a newer upstream commit, re-apply the patch (the LM weights live
in `pytorch_model.bin` and are loaded separately; only the modeling code
is vendored here).

## Local patches

- **TK-016 phase 3.b (cuda-graph capture-friendly)**: `model_omni.py`
  adds `MiniMindOmni.materialize_rope()`, and the forward's two
  lazy-init RoPE checks (`if self.thinker.freqs_cos[0, 0] == 0:`, a
  device->host scalar read) are replaced by a CPU-side
  `getattr(self, "_rope_materialized", False)` guard. `bundle.py` calls
  `materialize_rope(device)` eagerly after load. Semantically neutral --
  audio.wav parity MD5 `536fad2aba93d7b5df76067195dca26c` is preserved,
  and the forward now captures under `torch.cuda.graph` with bit-identical
  eager vs replay output.

## Why vendored

Originally, `bundle.py` loaded these files via
`transformers.AutoModelForCausalLM.from_pretrained(snapshot_dir, trust_remote_code=True)`.
That mechanism downloads the modeling code from the HF Hub on every cold
start, requires network access, and trusts whatever code is at the URL
at the moment of load. Vendoring pins the modeling code to this repo so
behavior is reproducible offline, the CUDA-graph work in TK-016 phase
3.c can patch the modeling code locally, and the trust_remote_code
escape hatch is no longer required at runtime.

## Upstream license

The upstream repository does not currently publish an explicit `LICENSE`
file alongside `model_omni.py` / `model_minimind.py`. Before redistributing
this vendored copy outside the nanovllm-omni project, confirm the license
terms with the upstream maintainer.
