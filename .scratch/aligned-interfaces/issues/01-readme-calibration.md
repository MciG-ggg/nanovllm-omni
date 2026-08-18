# 01 — README calibration (Phase 1, docs only)

**What to build:** The README and any docs reflect what the code actually does. A reader can trust every file path, code example, and "supported models" row in the README. Aspirational features are explicitly labeled as such or removed.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `grep -rE "serving/app\.py|examples/chat\.py|examples/image_gen\.py|examples/image_edit\.py|examples/vla\.py|examples/multi_stage\.py|diffusion/scheduler\.py|diffusion/audio_codec\.py|models/audio\.py|models/ar\.py|models/vla\.py|models/diffusion\.py|docs/design_mapping\.md|docs/deployment\.md|docs/TODO\.md|notebooks/01_stage_pipeline_walkthrough\.ipynb" README.md docs/` returns 0 matches.
- [x] README "Supported models" table contains only MiniMind-O. Any other rows removed (not just relabeled).
- [x] README has a "Status" section (or equivalent top-level callout) stating: only the MiniMind-O audio pipeline is wired up; other modalities (image generation, video generation, vision LLM, VLA, full-duplex S2S) are aspirational and not implemented.
- [x] README no longer references a non-existent `python -m nanovllm_omni.serving.app` entry point, nor any other entry point that does not exist on disk.
- [x] Any doc files referenced from README that do not exist are either created with a one-line stub note or removed from the README.
- [x] `git diff --stat` against the base branch shows modifications only to `*.md`, `*.txt`, and non-deploy `*.yaml` files. No Python source code modified in this ticket.
- [ ] All existing tests still pass.
- [x] The README's stated "alignment with vllm-omni" claim (if any) is softened to "alignment work tracked in `.scratch/aligned-interfaces/`" so that the README does not pre-claim work not yet done.