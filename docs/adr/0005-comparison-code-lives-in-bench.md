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
  Stage sets `cfg.enforce_eager = not extra["use_thinker_cuda_graph"]`,
  so eager is the default and the bench harness can still measure
  the graph cell.
- **talker** (`minimind_omni/stage.py:TalkerStage`): bypasses
  `ModelRunner` and uses fork `Context` directly — `enforce_eager`
  does not apply (see ADR-006).
- **smolvlm** (`smolvlm/stage.py:SmolVLMStage`): the stage owns the
  choice. It sets `cfg.enforce_eager = not extra["use_cuda_graph"]`,
  so the graph is the default and `extra["use_cuda_graph"] = False`
  turns it off (measured 2026-10-20 on a Colab T4: 3.16x end to end,
  see ADR-0003).

`DeployConfig` carried a comparison knob that only the bench
harness ever flipped (`nanovllm_omni/bench/__main__.py`). Production
never had a reason to toggle it; the public surface was a leak.

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

The flag reaches the stage as `OmniEngineArgs.extra`
(`use_thinker_cuda_graph`), because unknown kwargs land in `extra`.
`--use-talker-cuda-graph` was removed: the talker has no graph
path, so the flag could only ever be a no-op.

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

## Amendment (2026-10-20)

The decision above stands: stages own the graph-vs-eager choice, and
`DeployConfig` keeps no comparison knob. Two facts changed.

1. **The fork no longer dies on a failed capture.** `third_party/nano-vllm`
   catches the capture failure, prints a warning, and runs eager (fork
   commit `7cd0d4a`). It also bounds the captured KV window
   (`Config.graph_max_seq_len`, fork commit `f80d644`) and the captured
   batch size (`Config.max_capture_bs`, fork commit `4990060`). A stage
   can default to the graph without knowing whether the device captures
   it. The eager fallback compares the request against the window the
   graph actually captured (fork commit `719c066`).
2. **smolvlm now defaults to the graph.** Commit `951aaf0` turned
   `cfg.enforce_eager = True` into
   `cfg.enforce_eager = not extra.get("use_cuda_graph", True)`. The
   nsys profile on a Colab T4 showed the decode loop was launch-bound
   (GPU busy 24 % of the span), and the graph cut the end-to-end wall
   time from 586.0 ms to 185.7 ms (3.16x, ADR-0003).

The thinker keeps its eager default. The RTX 3050 capture failure was
measured on the pre-fork bench stack, and the current fork path has no
measurement on that card, so the default stays until a measurement
exists.

## Amendment (2026-10-30)

The measurement the previous amendment asked for now exists, so the
thinker default flips to the graph.

1. **Speedup on the card.** With the thinker graph on, the stage drops
   from ~2250 ms to ~595 ms per request, and a full request from 2692 ms
   to 990 ms (ADR-0007, `--runs 5`, seed 7, six prompts).
2. **No output change.** Graph and eager produce byte-identical WAV for
   the same prompt and seed, across two prompts
   (`tests/test_minimind_graph_parity.py`, `@pytest.mark.smoke`, one
   subprocess per cell). This is the gate the previous amendment implied
   but did not require; it is required now.
3. **The decision stays at the stage.** `DeployConfig` still carries no
   graph knob: `ThinkerStage.__init__` sets
   `cfg.enforce_eager = not extra.get("use_thinker_cuda_graph", True)`,
   so the graph is the default and `extra[...] = False` is how the bench
   measures the eager cell. Nothing about this amendment re-opens the
   rejected alternative of putting the flag back in `DeployConfig`.

The same rule applies to any future stage that wants to turn its graph on
by default: a measurement plus a parity test, not a config flag.

The ADR-006 reference below points at the decision written up in
`docs/dev/nanovllm-omni-paged-engine.md` (its own ADR-006 section).

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