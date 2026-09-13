# ADR 0005: Comparison code lives in bench, not in core

- Status: Accepted
- Date: 2026-09-13

## Context

`nanovllm-omni` used to expose two `DeployConfig` fields —
`use_thinker_cuda_graph` and `use_talker_cuda_graph` — that toggled
between CUDA Graph and eager dispatch per stage. They were inherited
from a prototyping phase where we were actively measuring the
speedup CUDA Graph could give on a single 4 GB RTX 3050, and they
made sense then: the production deploy config was the most
convenient place to flip a runtime knob.

The codebase has since settled. The three production stages have
made their graph-vs-eager decisions and they do not change at
runtime:

- **thinker** (`minimind_omni/stage.py:ThinkerStage`): graph
  capture fails on RTX 3050 with the current minimind-3o fork
  Config (`cudaErrorStreamCaptureInvalidated`). Same root cause as
  the historical `fix(smolvlm): force enforce_eager` series.
  Stage hardcodes `cfg.enforce_eager = True`.
- **talker** (`minimind_omni/stage.py:TalkerStage`): bypasses
  `ModelRunner` and uses fork `Context` directly — `enforce_eager`
  does not apply (see ADR-006).
- **smolvlm** (`smolvlm/stage.py:SmolVLMStage`): graph capture
  fails on this shell. Stage hardcodes `cfg.enforce_eager = True`.

`DeployConfig` carried a comparison knob that only the bench
harness ever flipped (`tools/profile_talker_graph_e2e.py`,
`tools/bench_paged_single_graph.py`, `nanovllm_omni/bench/__main__.py`).
Production never had a reason to toggle it; the public surface
was a leak.

The relay that wired the deploy flag through `stage_runner.py`
into fork `Config.enforce_eager` was dead infrastructure with a
misleading shape — it pretended to be a generic stage kwarg
while serving only the comparison block in
`ThinkerStage.__init__`.

## Decision

**Comparison code lives in `bench/` (and `tools/bench_*.py`,
`tools/profile_*.py`), not in core.**

Specifically:

1. `DeployConfig` no longer exposes `use_thinker_cuda_graph` /
   `use_talker_cuda_graph`. Stages pick their own default.
2. Each stage expresses its capability directly:
   `cfg.enforce_eager = True/False` on the cached `Config` after
   `get_stage_config` returns. No intermediate `use_*_cuda_graph`
   flag.
3. The fork `Config.enforce_eager` field itself is unchanged — it
   belongs to upstream `third_party/nano-vllm` and continues to
   gate graph capture inside fork's `ModelRunner`. Stages set it,
   the fork reads it.
4. `OmniEngineArgs` continues to reject `enforce_eager` as a kwarg
   (`config/params.py:69-71`). The public surface and the stage
   internals stay separate.

## Concrete application

The cleanup landed in two commits:

- `4a4502f` — drop `DeployConfig.use_thinker_cuda_graph` /
  `use_talker_cuda_graph`, the two loader lines, and the deploy
  YAML entries. Adds a focused test pinning the new shape
  (`tests/test_deploy_config_no_cuda_graph_fields.py`).
- `8798493` — drop the relay in `engine/stage_runner.py:68`;
  refactor `ThinkerStage` and `SmolVLMStage` to set
  `cfg.enforce_eager = True` directly with a one-line rationale
  comment.

`bench/` keeps its own copy of the comparison knob (the CLI flag
`args.use_thinker_cuda_graph` on `nanovllm_omni.bench.__main__`)
and continues to flow it as kwargs to `Omni(...)` /
`run_n_full(...)`. Removing the field from `DeployConfig` does
not affect bench.

## Considered alternatives

- **Keep `DeployConfig.use_*_cuda_graph` as user-tunable flags**
  (status quo). Rejected because production never had a reason to
  toggle them; only bench did. The public surface was a leak.
- **Move the flags to `OmniEngineArgs`** (private API) instead of
  removing them. Rejected because the same principle applies: the
  consumer of these flags is bench, not production. `OmniEngineArgs`
  is already a public-facing dataclass and would carry the same
  leak.
- **Replace the boolean with a richer "graph mode" enum**
  (`"eager"`, `"cuda-graph"`, `"auto"`). Rejected because each
  stage has exactly one capability, not a knob. The enum would
  re-introduce a comparison surface.
- **Move both bench locations (`nanovllm_omni/bench/` and
  `tools/bench_*.py`) into one location as part of this ADR.**
  Deferred — that is a separate decision (one bench location vs
  two) and the principle here does not depend on it.

## Consequences

- Stages own their graph-vs-eager decision; the rationale lives
  next to the stage code that depends on it (one-line comment per
  stage).
- Adding a new stage that wants a runtime-tunable graph mode would
  require re-opening this ADR. The default assumption is: a
  stage's decision is static, not user-tunable.
- If a future contributor proposes "let's add `use_*_cuda_graph`
  back to `DeployConfig`", they should re-read this ADR first.

## References

- `nanovllm_omni/config/registry.py:245,250` — fields removed
  (commit `4a4502f`).
- `nanovllm_omni/config/registry.py:423-424` — loader lines
  removed.
- `nanovllm_omni/deploy/minimind_omni.yaml` — top-level flags
  removed.
- `nanovllm_omni/engine/stage_runner.py:68` — relay removed
  (commit `8798493`).
- `nanovllm_omni/models/minimind_omni/stage.py:ThinkerStage.__init__`
  — sets `cfg.enforce_eager = True` directly.
- `nanovllm_omni/models/smolvlm/stage.py:SmolVLMStage.__init__`
  — same pattern.
- `tests/test_deploy_config_no_cuda_graph_fields.py` — pins the
  new shape.
- ADR-006: talker uses fork `Context` directly;
  `enforce_eager` does not apply.
- ADR-0001: profile-analysis protocol — its "Phase 1 baseline
  profile (eager mode)" cell stays correct since the eager
  baseline is a stage-level default, not a deploy-level flag.