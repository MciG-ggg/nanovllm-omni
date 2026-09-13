# ADR 0002: SmolVLM fork-AR integration

- Status: Accepted
- Date: 2026-09-12
- Commit: `c27745a feat(smolvlm): fork-style AR; SmolVLMStage via StageRunner`
  (fork loader prefix in submodule `65dd4b6`)

We port SmolVLM-500M-Instruct from a HuggingFace `model.generate()`
closure to the fork paged-attention AR engine. The text decoder
(SmolLM2-135M, Llama-style) gains paged KV + CUDA graph on the AR
decode loop; vision (HF SiglipVisionModel) and the vision→text
connector stay as plain `nn.Module` since one-shot encode has no paged
benefit.

## Decision

### ADR-016 (revised): Complete SmolVLM in fork-style; submodule names mirror HF checkpoint keys

`SmolVLMForConditionalGeneration` is one `nn.Module` holding:

| Attribute | Type | Why fork-style? |
|---|---|---|
| `self.model.vision_model` | HF `SiglipVisionModel` | no — one-shot encode, no KV cache |
| `self.model.connector.modality_projection` | `nn.Sequential(GELU, Linear)` | no — pure linear projection |
| `self.model.language_model` | `SmolLM2ForCausalLM` (fork building blocks) | yes — paged KV + CUDA graph |
| `self.lm_head` | fork `ParallelLMHead` | yes — needed by fork Sampler |

Submodule names mirror HF checkpoint keys (`model.vision_model.*` /
`model.connector.modality_projection.*` / `model.language_model.*` /
`lm_head.*`); the fork `load_model` with default `prefix=""` lands
HF weights directly. No `/tmp` weight slicing, no
`models/smolvlm/_weights.py` prefix-mapping hack.

This deviates from the originally proposed `self.vpm + self.llm
(prefix="llm") + self.resampler` (vllm-omni minicpmo shape) — that
shape requires the fork loader `prefix` kwarg plus a non-trivial name
mapping that's easy to misread. The HF-mirroring shape is what the
fork `load_model` was originally designed for: a checkpoint with
`model.X.Y` keys loads into a `nn.Module` with `model.X.Y` attributes.
We follow that convention rather than fight it.

### ADR-017: vision embeds via the existing `inputs_embeds` path; no fork forward-signature change

`SmolVLMStage._build_merged_embeddings` does prefill:

1. HF `vision_model(pixel_values=pixel_values).last_hidden_state`
2. `connector(image_features)` — projects 768-d vision to 576-d text
3. `embed_tokens(input_ids)` — text token embeddings
4. Merge at `image_token_id` positions (vision features replace text)
5. Flatten to `[T, t_dim]`, pass as `inputs_embeds` to fork `run_model`

`SmolVLMForConditionalGeneration.forward(input_ids, positions,
inputs_embeds=None)` follows the existing ADR-008 convention; no new
kwargs.

Cost: prefill via `inputs_embeds` skips CUDA graph (fork `run_model`
existing behaviour). Decode step keeps `input_ids` path, graph
replay still works.

### ADR-018: pipeline kind = `LLM_AR`

Replaces the prior `LLM_GENERATION` placeholder. SmolVLM is now
genuinely an AR paged decode, same execution shape as minimind thinker.
`StageExecutionType` is a closed `StrEnum`; `LLM_AR` already exists.

### ADR-019: `SmolVLMStage` class, tuple factory registration

`_vlm_stage(deploy, args)` returns a `SmolVLMStage` instance instead
of a closure. The factory's tuple registration stays stable so
`pipeline.py` remains declarative.

### Fork loader `prefix` kwarg (forward-compatible enabler)

`load_model(model, path, prefix: str = "")` — `prefix` prepended to
every weight name before lookup. Empty prefix = no change; existing
minimind / qwen3 / standalone consumers are unaffected. SmolVLM itself
uses `prefix=""` (HF-mirroring submodule names); future multimodal
shells following the vllm-omni shape (`self.llm = ...` /
`self.vpm = ...`) will use it.

## Files

| File | Change |
|---|---|
| `third_party/nano-vllm/nanovllm/utils/loader.py` | `prefix: str = ""` kwarg (submodule commit `65dd4b6`) |
| `nanovllm_omni/models/smolvlm/smolvlm.py` | new — full model class (~340 lines) |
| `nanovllm_omni/models/smolvlm/stage.py` | rewrite — `SmolVLMStage` class + prefill/decode loop |
| `nanovllm_omni/models/smolvlm/pipeline.py` | `kind` = `LLM_AR` (ADR-018) |
| `nanovllm_omni/deploy/smolvlm.yaml` | add `gpu_memory_utilization: 0.85`, `max_num_seqs: 1`, `max_num_batched_tokens: 8192` |
| `tests/test_smolvlm_fork_smoke.py` | new — submodule layout, packed_modules_mapping, connector shape, pipeline registration, stage factory export (CPU-only) |
| `tests/test_smolvlm.py` | update kind assertion to `LLM_AR` |

## Test gate (deferred to follow-up commit)

The CPU smoke covers model structure, not numerical correctness. The
GPU side still requires:

1. **WSL RTX 3050 end-to-end**: `Omni.generate(prompt, images, ...)`
   with the new `SmolVLMStage` returns coherent text.
2. **§6.2 fault-isolation gate** (per
   `docs/dev/nanovllm-omni-smolvlm-ar-migration.md`):
   - (a) prefill logits ≈ HF `model.forward(pixel_values=...).logits`
     at `atol=1e-5` (locates merge / RoPE / attention mask errors)
   - (b) decode single-step logits ≈ HF reference (locates paged KV /
       position embedding errors)
   - (c) full token sequence aligned under greedy / `temperature=0`
       (locates sampler / top_k errors)
3. **§6.3-bis profile gate** (was the prompt-question outcome): if
   `prefill_ms > decode_per_token * 1024`, fork-AR's paged-attention
   motivation holds; otherwise the migration is doc-only architectural
   value.

These gates are owned by the next session — the current commit covers
the implementation surface, not the hardware gate.

## Cross-references

- `docs/dev/nanovllm-omni-smolvlm-ar-migration.md` — long-form plan
  with §3 ADR-016/017/018/019 detail and §5 commit ordering
- `docs/dev/nanovllm-omni-smolvla-arflow-migration.md` — SmolVLA's
  vlm stage can reuse `SmolVLMForConditionalGeneration.language_model`
  if its lerobot backbone is verified compatible
  (currently `ADR-029 deferred`)
- `docs/adr/0001-profile-analysis-protocol.md` — Phase-1 baseline
  profile format for smolvlm
- `nanovllm_omni/models/minimind_omni/stage_runner.py::decode_minimind`
  — the AR-decode pattern `SmolVLMStage` mirrors